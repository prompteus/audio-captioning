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

from train_whisper_supervised import get_whisper_model


app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    load_checkpoint: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory where a checkpoint is saved"),
    dataset_dir: pathlib.Path = typer.Option(None, dir_okay=True, file_okay=False, readable=True, help="Path to the directory with audio files"),
    output_dir: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory where the predictions will be saved"),
    generate_config: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="yaml file with the inference config"),
) -> None:
    
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    with open(inference_config, "r") as f:
        inference_config_dict: dict = yaml.safe_load(f)

    generate_args_dict = inference_config_dict["generate_args"]
    architecture_config = inference_config_dict["architecture"]
    architecture_name = architecture_config["name"]
    use_pretrained_encoder = architecture_config["use_pretrained_whisper_encoder"]
    use_pretrained_decoder = architecture_config["use_pretrained_whisper_decoder"]

    data_config = inference_config_dict["data"]
    dataset_name = data_config["dataset_name"]
    task = data_config["task"]
    data_limit = data_config["data_limit"]

    config = transformers.WhisperConfig.from_pretrained(architecture_name)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture_name, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_name)
    assert isinstance(config, transformers.WhisperConfig)
    model = get_whisper_model(architecture_name, config, load_checkpoint, use_pretrained_encoder, use_pretrained_decoder)

    tuned_params = sum(p.shape.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.shape.numel() for p in model.parameters())
    print(f"Number of trained parameters: {tuned_params}/{total_params} = {tuned_params/total_params*100:.2f}%")

    # prepare dataset




    # set eval dataset
    audioset_dir = None
    audiocaps_dir = None
    clotho_dir = None

    if dataset_name == "clotho":
        clotho_dir = dataset_dir
    elif dataset_name == "audioset":
        audioset_dir = dataset_dir
    elif dataset_name == "audiocaps":
        audiocaps_dir = dataset_dir
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    dataset, audiofolders, ds_val_alternatives = audiocap.data.load_dataset_mixture(
        clotho_dir,
        audioset_dir,
        audiocaps_dir,
        dataset_weights,
        datasets_val_limits,
        0,
        0,
        tokenizer,
        feature_extractor,
    )

    compute_metrics = audiocap.metrics.CaptioningMetrics(tokenizer, ds_val_alternatives)
    collator = audiocap.data.DataCollatorAudioSeq2SeqWithPadding(tokenizer, feature_extractor)

    log_config_dict = {key: val for key, val in inference_config_dict.items() if key != "hf_training_args"}
    log_tags = ["supervised", architecture_name, dataset_name, split_name]

    wandb.init(
        project="audio-captioning",
        tags=log_tags,
        save_code=True,
        config=log_config_dict,
        group=wandb_group,
    )

    assert wandb.run is not None
    training_args_dict_preset: dict[str, Any] = {
        "output_dir": output_dir / wandb.run.name
    }
    training_args_dict = {**training_args_dict_preset, **training_args_dict}
    training_args = transformers.Seq2SeqTrainingArguments(**training_args_dict)


    callback_log_val_preds = audiocap.callbacks.PredictionLogger(
        log_prefix="val",
        dataset=dataset["val_mini"],
        collator=collator,
        log_every_n_steps=0,
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
        log_every_n_steps=0,
        skip_special_tokens=False,
        log_to_stdout=True,
        log_to_wandb=True,
        log_to_file=f"logs/preds_during_training/{wandb.run.name}/predictions_train.jsonl",
        generate_kwargs={"max_length": training_args_dict["generation_max_length"]},
    )

    callbacks: list[transformers.TrainerCallback]
    callbacks = [callback_log_val_preds, callback_log_train_preds]
    
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        args=training_args,
        callbacks=callbacks,
    )

    print(f"datasplit info\n##############\n{dataset[split_name].info()}\n##############\n")
    print("EVALUATING")
    trainer.evaluate(dataset[split_name])


if __name__ == "__main__":
    app()
