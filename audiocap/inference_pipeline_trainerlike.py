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
    dataset_dir: pathlib.Path = typer.Option(None, dir_okay=True, file_okay=False, readable=True, help="Path to the dataset directory"),
    output_dir: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory where the predictions will be saved"),
    inference_config: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="yaml file with the inference config"),
) -> None:
    
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    with open(inference_config, "r") as f:
        inference_config_dict: dict = yaml.safe_load(f)

    training_args_dict = inference_config_dict["hf_training_args"]
    architecture_config = inference_config_dict["architecture"]
    architecture_name = architecture_config["name"]
    use_pretrained_encoder = architecture_config["use_pretrained_whisper_encoder"]
    use_pretrained_decoder = architecture_config["use_pretrained_whisper_decoder"]

    data_config = inference_config_dict["data"]
    dataset_name = data_config["dataset_name"]
    split_name = data_config["split_name"]
    data_limit = data_config["data_limit"]
    log_pred_num = data_config["log_pred_num"]

    config = transformers.WhisperConfig.from_pretrained(architecture_name)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture_name, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_name)
    assert isinstance(config, transformers.WhisperConfig)
    model = get_whisper_model(architecture_name, config, load_checkpoint, use_pretrained_encoder, use_pretrained_decoder)

    tuned_params = sum(p.shape.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.shape.numel() for p in model.parameters())
    print(f"Number of trained parameters: {tuned_params}/{total_params} = {tuned_params/total_params*100:.2f}%")

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
    
    dataset_weights = {dataset_name: 1}    
    datasets_val_limits = {dataset_name: None}

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

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        args=training_args,
    )

    print(f"datasplit info\n##############\n{dataset[split_name].info()}\n##############\n")
    print("EVALUATING")
    trainer.evaluate(dataset[split_name])


if __name__ == "__main__":
    app()
