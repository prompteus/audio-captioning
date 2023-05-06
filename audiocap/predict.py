import os
import sys
import math
import pathlib
import json
import csv
import time

import torch
import torch.utils.data
import typer
import transformers
import yaml
import pandas as pd

import audiocap.data
import audiocap.models

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    checkpoint: str = typer.Option(..., dir_okay=True, file_okay=True, readable=True, help="Path to the checkpoint file"),
    data: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=True, readable=True, help="Path to the file / folder with the audio files"),
    output_file: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, exists=False, writable=True, help="Path to the folder where the predictions will be saved"),
    recursive: bool = typer.Option(False, help="Whether to recursively search for audio files in the data directory"),
    config_file: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, exists=True, readable=True, help="config_file"),
    take_first_n: int = typer.Option(None, help="Take only the first n files (for debugging)"),
) -> None:

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    architecture_config = config["architecture"]
    dataset_config = config["dataset"]
    dataloader_config = config["dataloader"]
    batch_size = dataloader_config["batch_size"]
    generate_config = config["generate"]
    runtime_config = config["runtime"]

    source_ds = dataset_config["source_ds"]
    task = dataset_config["task"]

    use_fp16 = runtime_config.get("use_fp16", False)
    device = runtime_config["device"]

    model = audiocap.models.WhisperForAudioCaptioning.from_pretrained(checkpoint)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_config["name"])

    # make mypy happy
    assert isinstance(model, audiocap.models.WhisperForAudioCaptioning)
    assert isinstance(tokenizer, transformers.WhisperTokenizer)
    assert isinstance(feature_extractor, transformers.WhisperFeatureExtractor)

    ds, num_files = audiocap.data.load_audios_for_predition(
        src=data,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        recursive=recursive,
        take_n=take_first_n,
        source_ds=source_ds,
        task=task,
    )

    print(f"Found: {num_files} files")

    dtype = torch.float16 if use_fp16 else torch.float32
    model = model.to(dtype).to(device).eval()

    collator = audiocap.data.DataCollatorAudioSeq2SeqWithPadding(tokenizer, feature_extractor, keep_cols=("file_name",))
    loader = torch.utils.data.DataLoader(ds, **dataloader_config, collate_fn=collator, drop_last=False, shuffle=False)

    log_file = output_file.parent / (output_file.stem + '_log.json')
    log_dict = {
        "checkpoint": checkpoint,
        "data": str(data),
        "output_file": str(output_file),
        "recursive": recursive,
        "config_file": str(config_file),
        "take_first_n": take_first_n,
        "num_files": num_files,
        "config": config,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        "command": " ".join(sys.argv),
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)

    start_time = time.time()

    with torch.no_grad():
        for b, batch in enumerate(loader):
            print("-" * 40)
            print(f"BATCH: {b}/{math.ceil(num_files / batch_size)}")
            preds_tokens = model.generate(
                input_features=batch["input_features"].to(dtype).to(device),
                forced_ac_decoder_ids=batch["forced_ac_decoder_ids"].to(device),
                **generate_config
            )
            preds_raw: list[str] = tokenizer.batch_decode(preds_tokens, skip_special_tokens=False)
            preds = pd.Series(tokenizer.batch_decode(preds_tokens, skip_special_tokens=True))
            preds = preds.apply(lambda x: str(x).split(":", maxsplit=1)[1].strip())

            for file_name, pred_raw in zip(batch["file_name"], preds_raw):
                print("FILE:", file_name)
                print("PRED:", pred_raw)
                print()
                    
            df = pd.DataFrame({"file_name": batch["file_name"], "caption_predicted": preds})
            df.to_csv(output_file, mode='a', header=not output_file.exists(), index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)

    log_dict["wall_time"] = time.time() - start_time

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    app()
