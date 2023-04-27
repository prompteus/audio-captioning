from __future__ import annotations
import math
import json
import pathlib
from typing import Callable

from tqdm.auto import tqdm
import pandas as pd
import transformers
import wandb
import torch
import torch.utils.data
import warnings

import audiocap.models


class PredictionLogger(transformers.TrainerCallback):
    def __init__(
        self,
        log_prefix: str,
        log_every_n_steps: int,
        dataset: torch.utils.data.Dataset | torch.utils.data.IterableDataset,
        collator: Callable,
        skip_special_tokens: bool,
        log_to_wandb: bool,
        log_to_stdout: bool,
        log_to_file: pathlib.Path | str | None = None,
        generate_kwargs: dict | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.log_prefix = log_prefix
        self.logging_steps = log_every_n_steps
        self.dataset = dataset
        self.collator = collator
        self.skip_special_tokens = skip_special_tokens
        self.log_to_stdout = log_to_stdout
        self.log_to_wandb = log_to_wandb

        if isinstance(log_to_file, str):
            log_to_file = pathlib.Path(log_to_file)
        if log_to_file is not None:
            if log_to_file.suffix != ".jsonl":
                warnings.warn(f"log_to_file is expected to have .jsonl suffix, got '{log_to_file}'")
            if log_to_file.exists():
                raise ValueError(f"log_to_file '{log_to_file}' already exists, stopping to avoid ruining it.")
            else:
                log_to_file.parent.mkdir(parents=True, exist_ok=True)
                log_to_file.touch()
        self.log_to_file = log_to_file

        if generate_kwargs is None:
            generate_kwargs = {}
        self.generate_kwargs = generate_kwargs

        if "max_length" not in generate_kwargs:
            warnings.warn(
                "you might have forgot to set `max_length` in `generate_kwargs` "
                f"inside {self.__class__.__name__}"
            )

        self.num_examples = sum(1 for _ in dataset)

    def on_step_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ) -> None:
        if state.global_step % self.logging_steps != 0:
            return
        
        model: audiocap.models.WhisperForAudioCaptioning = kwargs["model"]
        tokenizer: transformers.WhisperTokenizer = kwargs["tokenizer"]
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.collator,
        )

        model.eval()
        
        num_batches = math.ceil(self.num_examples / args.per_device_eval_batch_size)
        progress = tqdm(dataloader, total=num_batches, desc="Generating preds for logging", leave=False)
        all_preds = []
        all_trues = []
        all_forced_ac_decoder_ids = []

        with torch.no_grad():
            for batch in progress:
                for key in ["input_features", "forced_ac_decoder_ids", "attention_mask"]:
                    if key in batch:
                        batch[key] = batch[key].to(args.device)
                
                kwargs = self.generate_kwargs.copy()
                preds = model.generate(
                    inputs=batch["input_features"],
                    forced_ac_decoder_ids=batch["forced_ac_decoder_ids"],
                    **kwargs
                )
                preds_str = tokenizer.batch_decode(preds, skip_special_tokens=self.skip_special_tokens)
                labels_str = tokenizer.batch_decode(batch["labels"], skip_special_tokens=self.skip_special_tokens)
                forced_ac_decoder_ids = tokenizer.batch_decode(batch["forced_ac_decoder_ids"], skip_special_tokens=self.skip_special_tokens)
                all_preds.extend(preds_str)
                all_trues.extend(labels_str)
                all_forced_ac_decoder_ids.extend(forced_ac_decoder_ids)

        if self.log_to_stdout:
            print("=" * 100)
            print(f"WANDB TABLE: {self.log_prefix}, GLOBAL STEP: {state.global_step}")

            for item, pred, label, forced_ac_decoder_ids in zip(iter(self.dataset), all_preds, all_trues, all_forced_ac_decoder_ids):
                    print(
                        f"  FILE_NAME='{item['file_name']}'  WANDB_TABLE={self.log_prefix}  PREFIX='{item['prefix']}'"
                        f"  CAPTION_COLNAME='{item['caption_colname']}  FORCED_AC_DECODER_IDS='{forced_ac_decoder_ids}'"
                    )
                    print(f"  TRUES: '{label}'")
                    print(f"  PREDS: '{pred}'", flush=True)

            print("=" * 100)

        if self.log_to_file is not None:
            logged_df = pd.DataFrame({
                "global_step": [state.global_step] * self.num_examples,
                "file_name": [item["file_name"] for item in self.dataset],
                "trues": all_trues,
                "preds": all_preds,
                "prefix": [item["prefix"] for item in self.dataset],
                "caption_colname": [item["caption_colname"] for item in self.dataset],
                "forced_ac_decoder_ids": all_forced_ac_decoder_ids,
                "wandb_table": [self.log_prefix] * self.num_examples,

            })
            with open(self.log_to_file, "a", encoding="utf-8") as f:
                lines = logged_df.to_json(lines=True, orient="records", force_ascii=False)
                f.write(lines)

        if self.log_to_wandb:
            audios = [wandb.Audio(item["audio_array"], item["sampling_rate"], item["caption"]) for item in self.dataset]
            table = wandb.Table(
                columns=["audio", "truth", "preds", "file_name"],
                data=[
                    (audio, label, pred, item["file_name"])
                    for audio, label, pred, item in zip(audios, all_trues, all_preds, iter(self.dataset))
                ]
            )
            wandb.log({f"{self.log_prefix}_predictions": table}, step=state.global_step)
        

        