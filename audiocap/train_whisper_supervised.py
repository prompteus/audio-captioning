from __future__ import annotations

import pathlib
from typing import Optional, Any

import transformers
import wandb
import torch 
import typer
import yaml
import torchdata.datapipes as dp

import audiocap.metrics
import audiocap.data
import audiocap.callbacks
import audiocap.models
import audiocap.augment


app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    checkpoint_dir_root: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory where checkpoints will be saved"),
    clotho_dir: pathlib.Path = typer.Option(None, dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Clotho dataset"),
    audioset_dir: pathlib.Path = typer.Option(None, dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Audioset dataset"),
    audiocaps_dir: pathlib.Path = typer.Option(None, dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Audiocaps dataset"),
    training_config: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="yaml file with the training config"),
    load_checkpoint: Optional[pathlib.Path] = typer.Option(None, dir_okay=True, file_okay=True, readable=True, help="Path to checkpoint to initialize the model with"),
    wandb_group: Optional[str] = typer.Option(None, help="Wandb group"),
) -> None:
    
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    with open(training_config, "r") as f:
        training_config_dict: dict = yaml.safe_load(f)

    training_args_dict = training_config_dict["hf_training_args"]

    architecture_config = training_config_dict["architecture"]
    architecture_name = architecture_config["name"]
    use_pretrained_encoder = architecture_config["use_pretrained_whisper_encoder"]
    use_pretrained_decoder = architecture_config["use_pretrained_whisper_decoder"]

    early_stopping_config = training_config_dict["early_stopping"]
    should_early_stop = early_stopping_config["should_early_stop"]
    early_stopping_patience = early_stopping_config["early_stopping_patience"]
    early_stopping_threshold = early_stopping_config["early_stopping_threshold"]

    logging_config = training_config_dict["logging"]
    log_preds_every_n_steps = logging_config["log_preds_every_n_steps"]
    log_preds_num_train = logging_config["log_preds_num_train"]
    log_preds_num_valid = logging_config["log_preds_num_valid"]

    dataset_mix_config = training_config_dict["dataset_mix"]
    dataset_weights = dataset_mix_config["weights"]
    datasets_val_limits = dataset_mix_config["limit_val_split"]

    train_fc1_only = training_config_dict.get("train_fc1_only", False)

    if "augment" in training_config_dict:
        augment_config = audiocap.augment.AugmentConfig(**training_config_dict["augment"])
    else:
        augment_config = None


    config = transformers.WhisperConfig.from_pretrained(architecture_name)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture_name, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_name)
    assert isinstance(config, transformers.WhisperConfig)
    model = get_whisper_model(architecture_name, config, load_checkpoint, use_pretrained_encoder, use_pretrained_decoder)

    if train_fc1_only:
        for name, param in model.named_parameters():
            if "fc1" not in name:
                param.requires_grad = False

    tuned_params = sum(p.shape.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.shape.numel() for p in model.parameters())
    print(f"Number of trained parameters: {tuned_params}/{total_params} = {tuned_params/total_params*100:.2f}%")

    dataset, audiofolders, ds_val_alternatives = audiocap.data.load_dataset_mixture(
        clotho_dir,
        audioset_dir,
        audiocaps_dir,
        dataset_weights,
        datasets_val_limits,
        log_preds_num_train,
        log_preds_num_valid,
        tokenizer,
        feature_extractor,
        augment_config,
    )

    for ds in audiofolders:
        for split_name, split in ds.items():
            print(f"{split.source_ds} {split_name}: {len(split)} audio-caption pairs")

    compute_metrics = audiocap.metrics.CaptioningMetrics(tokenizer, ds_val_alternatives)
    collator = audiocap.data.DataCollatorAudioSeq2SeqWithPadding(tokenizer, feature_extractor)

    log_config_dict = {key: val for key, val in training_config_dict.items() if key != "hf_training_args"}
    log_tags = ["supervised", architecture_name, f"trained_params_{tuned_params/total_params*100:.2f}%"]

    if train_fc1_only:
        log_tags.append("fc1_only")
        log_config_dict["trained_params_percent"] = tuned_params / total_params

    wandb.init(
        project="audio-captioning",
        tags=log_tags,
        save_code=True,
        config=log_config_dict,
        group=wandb_group,
    )

    assert wandb.run is not None
    training_args_dict_preset: dict[str, Any] = {
        "output_dir": checkpoint_dir_root / wandb.run.name
    }
    training_args_dict = {**training_args_dict_preset, **training_args_dict}
    training_args = transformers.Seq2SeqTrainingArguments(**training_args_dict)

    callback_log_val_preds = audiocap.callbacks.PredictionLogger(
        log_prefix="val",
        dataset=dataset["val_mini"],
        collator=collator,
        log_every_n_steps=log_preds_every_n_steps,
        skip_special_tokens=False,
        log_to_wandb=True,
        log_to_stdout=True,
        log_to_file=f"logs/preds_during_training/{wandb.run.name}/predictions_val.jsonl",
        generate_kwargs={"max_length": training_args_dict["generation_max_length"]},
    )

    callback_log_train_preds = audiocap.callbacks.PredictionLogger(
        log_prefix="train",
        dataset=dataset["train_mini"],
        collator=collator,
        log_every_n_steps=log_preds_every_n_steps,
        skip_special_tokens=False,
        log_to_stdout=True,
        log_to_wandb=True,
        log_to_file=f"logs/preds_during_training/{wandb.run.name}/predictions_train.jsonl",
        generate_kwargs={"max_length": training_args_dict["generation_max_length"]},
    )

    callbacks: list[transformers.TrainerCallback]
    callbacks = [callback_log_val_preds, callback_log_train_preds]
    
    if should_early_stop:
        if early_stopping_patience is None:
            raise ValueError("early_stopping_patience must be specified if should_early_stop is True")
        early_stopping_kwargs = dict(early_stopping_patience=early_stopping_patience)
        if early_stopping_threshold is not None:
            early_stopping_kwargs["early_stopping_threshold"] = early_stopping_threshold  # type: ignore
        early_stopping = transformers.EarlyStoppingCallback(**early_stopping_kwargs)
        callbacks.append(early_stopping)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        args=training_args,
        callbacks=callbacks,
    )

    print("TRAINING")
    trainer.train()
    trainer.save_model(str(pathlib.Path(trainer.args.output_dir) / "final"))


def get_whisper_model(
    config_name: str,
    config: transformers.WhisperConfig,
    load_checkpoint: pathlib.Path | None,
    use_pretrained_whisper_encoder: bool,
    use_pretrained_whisper_decoder: bool,
) -> audiocap.WhisperForAudioCaptioning:
    
    if load_checkpoint is not None:
        model = audiocap.WhisperForAudioCaptioning.from_pretrained(load_checkpoint)
        assert isinstance(model, audiocap.WhisperForAudioCaptioning)
        return model

    if use_pretrained_whisper_encoder and use_pretrained_whisper_decoder:
        model = audiocap.WhisperForAudioCaptioning.from_pretrained(config_name)
        assert isinstance(model, audiocap.WhisperForAudioCaptioning)
        return model
    
    if not use_pretrained_whisper_encoder and not use_pretrained_whisper_decoder:
        return audiocap.WhisperForAudioCaptioning(config)
    
    model_pretrained = audiocap.WhisperForAudioCaptioning.from_pretrained(config_name)
    assert isinstance(model_pretrained, audiocap.WhisperForAudioCaptioning)
    model = audiocap.WhisperForAudioCaptioning(config)

    if use_pretrained_whisper_encoder:
        model.model.encoder = model_pretrained.get_encoder()

    if use_pretrained_whisper_decoder:
        model.model.decoder = model_pretrained.get_decoder()
    
    del model_pretrained
    return model


if __name__ == "__main__":
    app()
