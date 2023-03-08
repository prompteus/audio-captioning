import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F

from datasets.audiodataset import get_val_set, get_training_set
from models.cnn import get_model
from models.mel import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.mixup import mixup
from helpers.lr_schedule import exp_warmup_linear_down


class SimpleDCASELitModule(pl.LightningModule):
    """
    This is a Pytorch Lightening Module.
    It has several convenient abstractions, e.g. we don't have to specify all parts of the
    training loop (optimizer.step(), loss.backward()) ourselves.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse and contains all configurations for our experiment
        # model to preprocess waveforms into log mel spectrograms
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

        # our model to be trained on the log mel spectrograms
        self.model = get_model(in_channels=config.in_channels,
                               n_classes=config.n_classes,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier
                               )

    def mel_forward(self, x):
        """
        @param x: a batch of raw signals (waveform)
        return: a batch of log mel spectrograms
        """
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])  # for calculating mel spectrograms we remove the channel dimension
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])  # batch x channels x mels x time-frames
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
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
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx: will likely not be used at all
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """
        x, y = train_batch  # we get a batch of raw audio signals and labels as defined by our dataset
        bs = x.size(0)
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if args.mixup_alpha:
            # Apply Mixup, a very common data augmentation method
            rn_indices, lam = mixup(bs, args.mixup_alpha)  # get shuffled indices and mixing coefficients
            # send mixing coefficients to correct device and make them 4-dimensional
            lam = lam.to(x.device).reshape(bs, 1, 1, 1)
            # mix two spectrograms from the batch
            x = x * lam + x[rn_indices] * (1. - lam)
            # generate predictions for mixed log mel spectrograms
            y_hat = self.model(x)
            # mix the prediction targets using the same mixing coefficients
            samples_loss = (
                    F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                    F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(bs))
            )

        else:
            y_hat = self.model(x)
            # cross_entropy is used for multiclass problems
            # be careful when choosing the correct loss functions
            # read the documentation what input your loss function expects, e.g. for F.cross_entropy:
            # the logits (no softmax!) and the prediction targets (class indices)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        loss = samples_loss.mean()
        results = {"loss": loss}
        return results

    def training_epoch_end(self, outputs):
        """
        :param outputs: contains the items you log in 'training_step'
        :return: a dict containing the metrics you want to log to Weights and Biases
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log_dict({'loss': avg_loss})

    def validation_step(self, val_batch, batch_idx):
        # similar to 'training_step' but without any data augmentation
        # pytorch lightening takes care of 'with torch.no_grad()' and 'model.eval()'
        x, y = val_batch
        x = self.mel_forward(x)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # log validation metric to weights and biases
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log_dict({'val_loss': avg_loss})


def train(config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="DCASE23",
        notes="Test pipeline for DCASE23 tasks.",
        tags=["DCASE23"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    # train dataloader
    train_dl = DataLoader(dataset=get_training_set(config.cache_path, config.resample_rate, config.roll),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    # test loader
    val_dl = DataLoader(dataset=get_val_set(config.cache_path, config.resample_rate),
                        worker_init_fn=worker_init_fn,
                        num_workers=config.num_workers,
                        batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = SimpleDCASELitModule(config)
    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto',
                         callbacks=[lr_monitor])
    # start training and validation
    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    # simplest form of specifying hyperparameters using argparse
    # IMPORTANT: log hyperparameters to be able to reproduce you experiments!
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="DCASE23")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for dataloaders

    # dataset
    # location to store resample waveform
    parser.add_argument('--cache_path', type=str, default="datasets/example_data/cached")
    parser.add_argument('--roll', default=False, action='store_true')  # rolling waveform over time

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    # spectrograms have 1 input channel (RGB images would have 3)
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--channels_multiplier', type=int, default=2)

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--mixup_alpha', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # learning rate + schedule
    # phases:
    #  1. exponentially increasing warmup phase (for 'warm_up_len' epochs)
    #  2. constant lr phase using value specified in 'lr' (for 'ramp_down_start' - 'warm_up_len' epochs)
    #  3. linearly decreasing to value 'las_lr_value' * 'lr' (for 'ramp_down_len' epochs)
    #  4. finetuning phase using a learning rate of 'last_lr_value' * 'lr' (for the rest of epochs up to 'n_epochs')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warm_up_len', type=int, default=6)
    parser.add_argument('--ramp_down_start', type=int, default=20)
    parser.add_argument('--ramp_down_len', type=int, default=10)
    parser.add_argument('--last_lr_value', type=float, default=0.01)  # relative to 'lr'

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)  # in samples (corresponds to 25 ms)
    parser.add_argument('--hop_size', type=int, default=320)  # in samples (corresponds to 10 ms)
    parser.add_argument('--n_fft', type=int, default=1024)  # length (points) of fft, e.g. 1024 point FFT
    parser.add_argument('--n_mels', type=int, default=128)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=0)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram bins
    parser.add_argument('--fmin', type=int, default=0)  # mel bins are created for freqs. between 'fmin' and 'fmax'
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)  # data augmentation: vary 'fmin' and 'fmax'
    parser.add_argument('--fmax_aug_range', type=int, default=1000)

    args = parser.parse_args()
    train(args)
