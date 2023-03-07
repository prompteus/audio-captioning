import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F

from datasets.audiodataset import get_test_set, get_training_set
from models.cnn import get_model
from models.mel import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.mixup import mixup
from helpers.lr_schedule import exp_warmup_linear_down


class SimpleDCASELitModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # model to preprocess waveform into mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config.n_mels,
                                  sr=config.resample_rate,
                                  win_length=config.window_size,
                                  hopsize=config.hop_size,
                                  n_fft=config.n_fft,
                                  freqm=config.freqm,
                                  timem=config.timem,
                                  fmin=config.fmin,
                                  fmax=config.fmax,
                                  fmin_aug_range=config.fmin_aug_range,
                                  fmax_aug_range=config.fmax_aug_range
                                  )

        self.model = get_model(in_channels=config.in_channels,
                               n_classes=config.n_classes,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier
                               )

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def forward(self, x):
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        schedule_lambda = \
            exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
                                   self.config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        bs = x.size(0)
        x = self.mel_forward(x)

        if args.mixup_alpha:
            rn_indices, lam = mixup(bs, args.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(bs, 1, 1, 1) + \
                x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
            y_hat = self.model(x)
            samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                            F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                    1. - lam.reshape(bs)))

        else:
            y_hat = self.model(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        loss = samples_loss.mean()
        results = {"loss": loss}
        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log_dict({'loss': avg_loss})

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = self.mel_forward(x)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log_dict({'val_loss': avg_loss})


def train(config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="DCASE23",
        notes="Test pipeline for DCASE23 tasks.",
        tags=["DCASE23"],
        config=config,
        name=config.experiment_name,
        log_model=True
    )

    # dataloader
    train_dl = DataLoader(dataset=get_training_set(config.cache_path, config.resample_rate, config.roll),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    # test loader
    val_dl = DataLoader(dataset=get_test_set(config.cache_path, config.resample_rate),
                        worker_init_fn=worker_init_fn,
                        num_workers=config.num_workers,
                        batch_size=config.batch_size)

    pl_module = SimpleDCASELitModule(config)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(max_epochs=config.n_epochs, logger=wandb_logger, accelerator='gpu', devices=1,
                         callbacks=[lr_monitor])
    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="DCASE23")
    parser.add_argument('--num_workers', type=int, default=12)

    # dataset
    parser.add_argument('--cache_path', type=str, default="datasets/example_data/cached")
    parser.add_argument('--roll', default=False, action='store_true')

    # model
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--channels_multiplier', type=int, default=2)

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--mixup_alpha', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=3e-4)  # learning rate + schedule
    parser.add_argument('--warm_up_len', type=int, default=6)
    parser.add_argument('--ramp_down_start', type=int, default=20)
    parser.add_argument('--ramp_down_len', type=int, default=10)
    parser.add_argument('--last_lr_value', type=float, default=0.01)  # relative to '--lr'

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)
    parser.add_argument('--fmax_aug_range', type=int, default=1000)

    args = parser.parse_args()

    train(args)
