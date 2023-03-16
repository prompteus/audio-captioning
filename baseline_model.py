
import torch
from torch.nn import Linear, LayerNorm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from transformers import BertConfig, BertForCondtionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartAttention

from models.cnn import BasicBlock
from models.mel import AugmentMelSTFT



def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class CNN14(pl.LightningModule):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.b0 = nn.BatchNorm2d(64)

        self.cnn = nn.Sequential(
                            # Conv2D block1
                            BasicBlock(1, 64),
                            nn.AvgPool2d(kernel_size=2),
                            nn.Dropout(p=0.2),
                
                            # Conv2D block2
                            BasicBlock(64, 128)
                            nn.AvgPool2d(kernel_size=2),
                            nn.Dropout(p=0.2),
                
                            # Conv2D block3
                            BasicBlock(128, 256),
                            nn.AvgPool2d(kernel_size=2),
                            nn.Dropout(p=0.2),
                
                            # Conv2D block4
                            BasicBlock(256, 512),
                            nn.AvgPool2d(kernel_size=2),
                            nn.Dropout(p=0.2),
                
                            # Conv2D block5
                            BasicBlock(512, 1024),
                            nn.AvgPool2d(kernel_size=2),
                            nn.Dropout(p=0.2),
                
                            # Conv2D block6
                            BasicBlock(1024, 2048),
                            nn.AvgPool2d(kernel_size=2),
                            nn.Dropout(p=0.2)
                            )
        
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Linear(2048, config["out_dim"], bias=True)

        self.bn0.apply(init_weights)
        self.cnn.apply(init_weights)
        self.fc.apply(init_weights)
        self.fc2.apply(init_weights)

    def forward(self, x, skip_fc=False):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, embed_dim).
        """
        x = x.unsqueeze(1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.cnn(x)
        x = torch.mean(x, dim=3) # (N, 2048, T/64)
        
        if skip_fc:
            return x.transpose(1,2) # b,t,h
        else:
            (x1, _) = torch.max(x, dim=2)  # max across time
            x2 = torch.mean(x, dim=2)  # average over time
            x = x1 + x2  # (N, 2048)

            x = self.fc(x)  # (N, 2048)
            x = self.fc2(x)  # (N, embed_dim)

            return x

class BaselineModule(pl.LightningModule):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

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

        self.audio_enc = CNN14(config)

        self.bart = BartConfig(vocab_size=config.vocab_size,
                                encoder_layers=config.encoder_layers,
                                encoder_ffn_dim=config.encoder_ffn_dim,
                                encoder_attention_heads=config.encoder_attention_heads,
                                decoder_layers=config.decoder_layers,
                                decoder_ffn_dim=config.decoder_ffn_dim,
                                decoder_attention_heads=config.decoder_attention_heads,
                                activation_function=config.activation_function,
                                d_model=config.d_model,
                                dropout=config.dropout,
                                attention_dropout=config.attention_dropout,
                                activation_dropout=config.activation_dropout,
                                classifier_dropout=config.classifier_dropout,
                                max_length=config.max_length,
                                min_length=config.min_length,
                                early_stopping=config.early_stopping,
                                num_beams=config.num_beams,
                                length_penalty=config.length_penalty,
                                no_repeat_ngram_size=config.no_repeat_ngram_size)

    def forward(self,
                inputs=None,
                cond_tokens=None,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
            ):
        
        audio_features = self.mel(inputs)
        audio_embs = self.audio_enc(audio_features, skip_fc=True)
        
        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_embs)
        else:
            audio_embs = audio_features
        
        # Encoder pass
        encoder_outputs = self.bart.model.encoder(
                    input_ids=None,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=audio_embs,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True)['last_hidden_state']
        
        encoder_outputs = [encoder_outputs]
        
        # Decoder-only pass
        outputs = self.bart(input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
        )
