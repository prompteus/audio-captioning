from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import transformers
import datasets
import torch 
import typer
import yaml
import json
import pandas as pd

import audiocap.metrics
import audiocap.preprocess
import audiocap.callbacks

from tqdm import tqdm

app = typer.Typer()

@app.command()
def main(
    #architecture_name: str = typer.Option(..., help="Name of the config to use for the model, e.g.: 'openai/whisper-small'"),
    checkpoint_dir: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory of the checkpoints that should be evaluated"),
    clotho_dir: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Clotho dataset"),
    output_file: str = typer.Option(..., help="Predictions output file name"),
    evaluate_validation: bool = typer.Option(False, help="Evaluate the Validation Dataset"),
    #training_args_config: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="yaml file with the training arguments"),
) -> None:
    
    
    input("Wait for debugger")

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    config = json.load(open(f'{checkpoint_dir}/config.json'))
    architecture_name = config['_name_or_path']
    #config = transformers.WhisperConfig.from_pretrained(architecture_name)
    #model = transformers.WhisperConfig.from_pretrained(architecture_name)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture_name)
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_name)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
    data_files = {}
    data_dir = None
    if evaluate_validation:
        data_dir = clotho_dir / 'validation'
    else:
        data_dir = clotho_dir / 'evaluation'
    data_files['eval'] = f"{data_dir}/*"
    ds_raw = datasets.load_dataset(
        "audiofolder",
        data_files=data_files,
        
        ignore_verifications=True,
    )

    random_gen = np.random.default_rng(seed=1)
    preprocessing = audiocap.preprocess.Preprocess(tokenizer, feature_extractor)

    ds = datasets.IterableDatasetDict()
    for split_name in ds_raw.keys():
        # TODO add augmentations, but only to development split
        split = (ds_raw[split_name]
            .to_iterable_dataset()
            .map(
                audiocap.preprocess.clotho_flatten_captions,
                batched=True,
                batch_size=10,
                remove_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
            )
            .map(
                preprocessing,
                batched=True,
                batch_size=16,
                remove_columns=["audio"],
            )
        )
        ds[split_name] = split

    expected_keys = { 'caption_idx', 'caption', 'path', 'audio_array', 'sampling_rate', 'filename', 'input_features', 'labels' }

    del ds_raw

    collator = audiocap.preprocess.DataCollatorAudioSeq2SeqWithPadding(tokenizer, feature_extractor)
    #compute_metrics = audiocap.metrics.CaptioningMetrics(tokenizer) if compute_metrics and evaluate_validation else None
    
    training_args_dict_preset = {"output_dir": checkpoint_dir}
    
    training_args = transformers.Seq2SeqTrainingArguments(**training_args_dict_preset)
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
        eval_dataset=ds["eval"],
        args=training_args,
    )

    filenames = [f.name for f in data_dir.glob('*.wav')]

    preds_str = []
    for i in tqdm(range(0, len(filenames), 10)):
        take_ds = ds['eval'].skip(i).take(min(10, len(filenames)-i))
        eval_out = trainer.predict(take_ds)
        preds_str.extend(tokenizer.batch_decode(eval_out.label_ids, skip_special_tokens=True))
    
    df = pd.DataFrame({'file_name': filenames,
                       'caption_predicted': preds_str,}
    )
    df.to_csv(f'{output_file}' ,index=False)

if __name__ == "__main__":
    app()
