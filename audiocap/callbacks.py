import datasets
import transformers
import wandb
import torch
import warnings


class WandbPredictionLogger(transformers.TrainerCallback):
    def __init__(
        self,
        log_prefix: str,
        log_every_n_steps: int,
        dataset: datasets.Dataset,
        collator: transformers.DataCollator,
        generate_kwargs: dict = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.log_prefix = log_prefix
        self.logging_steps = log_every_n_steps
        self.dataset = dataset
        self.collator = collator

        if generate_kwargs is None:
            generate_kwargs = {}
        self.generate_kwargs = generate_kwargs

        if "max_length" not in generate_kwargs:
            warnings.warn(
                "you might forgot to set `max_length` in `generate_kwargs` "
                f"inside {self.__class__.__name__}"
            )

    def on_step_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ) -> None:
        model: transformers.PreTrainedModel = kwargs["model"]
        tokenizer: transformers.PreTrainedTokenizer = kwargs["tokenizer"]
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.collator,
        )

        if state.global_step % self.logging_steps != 0:
            return
        
        model.eval()
        
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                kwargs = self.generate_kwargs.copy()
                if "attention_mask" in batch:
                    kwargs["attention_mask"] = batch["attention_mask"].to(args.device)
                preds = model.generate(batch["input_features"].to(args.device), batch["forced_prefix_ids"], **kwargs)
                preds_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
                all_preds.extend(preds_str)

        audios = [wandb.Audio(item["audio_array"], item["sampling_rate"], item["caption"]) for item in self.dataset]

        table = wandb.Table(
            columns=["audio", "prediction", "label", "filename"],
            data=[
                (audio, pred, item["caption"], item["filename"])
                for audio, pred, item in zip(audios, all_preds, self.dataset)
            ]
        )
        
        wandb.log({f"{self.log_prefix}_predictions": table}, step=state.global_step)
        