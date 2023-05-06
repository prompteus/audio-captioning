from __future__ import annotations

import pathlib
from typing import Optional, Any

import transformers
import wandb
import torch
from torch.utils.data import DataLoader
import typer
import yaml
import torchdata.datapipes as dp
import json
import time

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
    split_type: str = typer.Option("val", help="Which split to use for evaluation"),
    log_file: pathlib.Path = typer.Option(None, dir_okay=False, file_okay=True, readable=True, help="Path to log file for runtimes"),
) -> int:
    
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    with open(generate_config, "r") as f:
        generate_config_dict: dict = yaml.safe_load(f)

    generate_args_dict = generate_config_dict["generate_args"]
    checkpoint_config = json.load(open(f'{load_checkpoint}/config.json'))
    architecture_name = checkpoint_config['_name_or_path']

    batch_size = generate_config_dict["batch_size"]

    data_config = generate_config_dict["data"]
    dataset_name = data_config["dataset_name"]
    data_limit = data_config["data_limit"]

    run_name = load_checkpoint.parent.name  
    print("run name: ", run_name)

    assert split_type != "train", "Cannot generate captions for training set (it's shuffled)"

    print(">>>>>> LOADING CHECKPOINT AND DATA <<<<<<")

    whisper_config = transformers.WhisperConfig.from_pretrained(architecture_name)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture_name, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_name)
    collator = audiocap.data.DataCollatorAudioSeq2SeqWithPadding(tokenizer, feature_extractor, keep_filename=True)
    assert isinstance(whisper_config, transformers.WhisperConfig)
    model = get_whisper_model(architecture_name, whisper_config, load_checkpoint, False, False)
    model.to(device)

    tuned_params = sum(p.shape.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.shape.numel() for p in model.parameters())
    print(f"Number of trained parameters: {tuned_params}/{total_params} = {tuned_params/total_params*100:.2f}%")

    # set eval dataset
    if dataset_name == "clotho":
        ds = audiocap.data.load_clotho(dataset_dir, tokenizer, feature_extractor, None, 0, 0, 0)
    elif dataset_name == "audioset":
        ds = audiocap.data.load_audioset(dataset_dir, tokenizer, feature_extractor, None, 0, 0, 0)
    elif dataset_name == "audiocaps":
        ds = audiocap.data.load_audiocaps(dataset_dir, tokenizer, feature_extractor, None, 0, 0, 0)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    print(">>>>>> GENERATING CAPTIONS <<<<<<")
    start_time = time.time()
    all_predictions = generate_captions(ds, model, tokenizer, collator, generate_args_dict, split_type, batch_size, data_limit, device)
    total_runtime = time.time() - start_time

    # save predictions as jsonl to output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file_name = generate_config.stem
    preds_file_path = output_dir / f"{run_name}_{dataset_name}_{split_type}_{config_file_name}.jsonl"
    with open(preds_file_path, "w") as f:
        for prediction in all_predictions:
            f.write(json.dumps(prediction) + "\n")
    with open(log_file, "a") as f:
            f.write(f"{preds_file_path.stem}: {total_runtime //60} min, {total_runtime % 60} s\n")

    print(">>>>>> GENERATING DONE <<<<<<")


def generate_captions(ds, model, tokenizer, collator, generate_args_dict, split_type, batch_size=2, data_limit=None, device="cuda"):
    dataloader = DataLoader(ds[split_type].pipe, batch_size=batch_size, collate_fn=collator, num_workers=4)
    all_predictions = []
    for i, batch in enumerate(dataloader):
        if data_limit and i >= data_limit/batch_size:
            break
        filenames = batch.pop("file_name")
        prediction_ids = model.generate(batch["input_features"].to(device), 
                                        forced_ac_decoder_ids=batch["forced_ac_decoder_ids"].to(device),
                                        **generate_args_dict)
        predictions = tokenizer.batch_decode(prediction_ids, skip_special_tokens=False)
        predictions_dict = [{"file_name": fn, "caption": p} for fn, p in zip(filenames, predictions)]
        all_predictions.extend([{"file_name": fn, "caption": p} for fn, p in zip(filenames, predictions)])
        [print(p) for p in predictions_dict]
  
    return all_predictions


if __name__ == "__main__":
    app()
