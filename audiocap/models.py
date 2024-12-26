from typing import Optional, Tuple, Union

import transformers
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE


class WhisperForAudioCaptioning(transformers.WhisperForConditionalGeneration):

    def forward(
        self,
        forced_ac_decoder_ids: Optional[torch.LongTensor] = None, # added to be ignored when passed from trainer
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return super().forward(**kwargs)
    
    # copy-pasted and adapted from transformers.WhisperForConditionalGeneration.generate
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        forced_ac_decoder_ids: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        return_timestamps=None,
        task="transcribe",
        language="english",
        **kwargs,
    ):
        if generation_config is None:
            generation_config = self.generation_config

        if return_timestamps is not None:
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set."
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`."
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )

            generation_config.return_timestamps = return_timestamps
        else:
            generation_config.return_timestamps = False

        if language is not None:
            generation_config.language = language
        if task is not None:
            generation_config.task = task

        forced_decoder_ids = []
        if task is not None or language is not None:
            if hasattr(generation_config, "language"):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                else:
                    raise ValueError(
                        f"Unsupported language: {language}. Language should be one of:"
                        f" {list(TO_LANGUAGE_CODE.keys()) if generation_config.language in TO_LANGUAGE_CODE.keys() else list(TO_LANGUAGE_CODE.values())}."
                    )
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))  # automatically detect the language

            if hasattr(generation_config, "task"):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(
                        f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                    )
            else:
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
            if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        # Legacy code for backward compatibility
        elif hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif (
            hasattr(self.generation_config, "forced_decoder_ids")
            and self.generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = self.generation_config.forced_decoder_ids

        if generation_config.return_timestamps:
            logits_processor = [WhisperTimeStampLogitsProcessor(generation_config)]

        decoder_input_ids = None

        if len(forced_decoder_ids) > 0:
            # get the token sequence coded in forced_decoder_ids
            forced_decoder_ids.sort()
            if min(forced_decoder_ids)[0] != 0:
                forced_decoder_ids = [(0, self.config.decoder_start_token_id)] + forced_decoder_ids 

            position_indices, decoder_input_ids = zip(*forced_decoder_ids)
            assert tuple(position_indices) == tuple(range(len(position_indices))), "forced_decoder_ids is not a (continuous) prefix, we can't handle that"

            device = self.get_decoder().device

            if forced_ac_decoder_ids is None:
                forced_ac_decoder_ids = torch.tensor([[]], device=device, dtype=torch.long)

            # enrich every sample's forced_ac_decoder_ids with Whisper's forced_decoder_ids
            batch_size = forced_ac_decoder_ids.shape[0]
            fluff_len = len(decoder_input_ids)
            decoder_input_ids = torch.tensor(decoder_input_ids, device=device, dtype=torch.long)
            decoder_input_ids = decoder_input_ids.expand((batch_size, fluff_len))
            decoder_input_ids = torch.cat([decoder_input_ids, forced_ac_decoder_ids], dim=1)

        # This attribute is no longer supported by transformers whisper generate
        if hasattr(generation_config, "forced_decoder_ids"):
            generation_config.forced_decoder_ids = None
        # but it's not needed because we force the decoder input ids directly via
        # `decoder_input_ids` parameter below

        return transformers.WhisperPreTrainedModel.generate(
            self,
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            decoder_input_ids=decoder_input_ids,
            **kwargs,
        )