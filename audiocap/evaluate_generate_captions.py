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

    with open(generate_config, "r") as f:
        generate_config_dict: dict = yaml.safe_load(f)

    generate_args_dict = generate_config_dict["generate_args"]
    architecture_config = generate_config_dict["architecture"]
    architecture_name = architecture_config["name"]
    use_pretrained_encoder = architecture_config["use_pretrained_whisper_encoder"]
    use_pretrained_decoder = architecture_config["use_pretrained_whisper_decoder"]

    only_allowed_tokens = generate_args_dict["only_allowed_tokens"]

    data_config = generate_config_dict["data"]
    dataset_name = data_config["dataset_name"]
    task = data_config["task"]
    dataset_type = data_config["dataset_type"]
    data_limit = data_config["data_limit"]

    assert dataset_type != "train", "Cannot generate captions for training set (it's shuffled)"

    config = transformers.WhisperConfig.from_pretrained(architecture_name)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture_name, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_name)
    collator = audiocap.data.DataCollatorAudioSeq2SeqWithPadding(tokenizer, feature_extractor)
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
        ds = audiocap.data.load_clotho(dataset_dir, tokenizer, feature_extractor, None, 0, 0, 0)
    elif dataset_name == "audioset":
        ds = audiocap.data.load_audioset(dataset_dir, tokenizer, feature_extractor, None, 0, 0, 0)
    elif dataset_name == "audiocaps":
        ds = audiocap.data.load_audiocaps(dataset_dir, tokenizer, feature_extractor, None, 0, 0, 0)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    if only_allowed_tokens:
        # get_list of allowed tokens
        print("TODO: Not working yet - allowing all. Getting list of allowed tokens")
        prefix_allowed_tokens_fn = lambda x: list(tokenizer.get_vocab().values())
    else:
        print("Allowing all tokens")
        prefix_allowed_tokens_fn = lambda x: list(tokenizer.get_vocab().values())




    # generate captions
    predictions = []
    for i, batch in enumerate(DataLoader(ds[dataset_type].pipe)):
        if data_limit and i >= data_limit:
            break

        col_batch = collator([batch])
        prediction_ids = model.generate(col_batch["input_features"][0], 
                                        forced_ac_decoder_ids=col_batch["forced_ac_decoder_ids"], 
                                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, 
                                        **generate_args_dict)
        prediction = tokenizer.batch_decode(prediction_ids, skip_special_tokens=False)[0]
        predictions.append({"file_name": batch["file_name"], "caption": prediction})
        print(batch["file_name"], prediction)

    # save captions to jsonl file
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{dataset_name}_{dataset_type}_beams{generate_args_dict['num_beams']}.jsonl", "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

if __name__ == "__main__":
    app()
