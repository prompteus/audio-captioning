from __future__ import annotations

import pathlib
from typing import Optional, Any

import numpy as np
import transformers
import datasets
import wandb
import torch 
import typer
import yaml
import pandas as pd

import audiocap.metrics
import audiocap.preprocess
import audiocap.callbacks
from audiocap.models import WhisperForAudioCaptioning


app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    checkpoint_dir_root: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory where checkpoints will be saved"),
    training_phase: str = typer.Option(..., help="Name of the training phase, either pretraining or finetuning"),
    clotho_dir: pathlib.Path = typer.Option(None, dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Clotho dataset"),
    audioset_dir: pathlib.Path = typer.Option(None, dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Audioset dataset"),
    audiocaps_dir: pathlib.Path = typer.Option(None, dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Audiocaps dataset"),
    #limit_train_split_size: Optional[int] = typer.Option(None, help="Limit the dev split size (for debugging purposes)"),
    #limit_valid_split_size: Optional[int] = typer.Option(..., help="Limit the val split size (for debugging purposes)"),
    training_config: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="yaml file with the training config"),
) -> None:
    
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    with open(training_config, "r") as f:
        training_config_dict = yaml.safe_load(f)

    training_args_dict = training_config_dict["hf_training_args"]
    remaining_args_dict = training_config_dict["remaining_training_args"]

    architecture_name = remaining_args_dict["architecture_name"]
    use_pretrained_encoder = remaining_args_dict["use_pretrained_whisper_encoder"]
    use_pretrained_decoder = remaining_args_dict["use_pretrained_whisper_decoder"]

    should_early_stop = remaining_args_dict["should_early_stop"]
    early_stopping_patience = remaining_args_dict["early_stopping_patience"]
    early_stopping_threshold = remaining_args_dict["early_stopping_threshold"]

    config = transformers.WhisperConfig.from_pretrained(architecture_name)
    model = transformers.WhisperConfig.from_pretrained(architecture_name)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture_name, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_name)
    assert isinstance(config, transformers.WhisperConfig)
    model = get_whisper_model(architecture_name, config, use_pretrained_encoder, use_pretrained_decoder)
    

    if training_phase == "pretraining":

        ds_audioset = audiocap.data.load_audioset_small(
            audioset_dir / "audiofolder",
            audioset_dir / "annotations/ontology.json",
            tokenizer,
            feature_extractor
        )

        ds_audiocaps = audiocap.data.load_audiocaps(audiocaps_dir / "audiofolder", tokenizer, feature_extractor)
        
        # TODO:
        # - add probabilities + handle if one ends before the other
        # swap concat with interleave
        ds = {
            "train": datasets.concatenate_datasets([ds_audioset["train"], ds_audiocaps["train"]]),
            "val": datasets.concatenate_datasets([ds_audioset["val"], ds_audiocaps["val"]]),
            "test": datasets.concatenate_datasets([ds_audioset["test"], ds_audiocaps["test"]]),
        }

    elif training_phase == "finetuning":
        # prepare clotho dataset
        ds = audiocap.data.load_clotho(clotho_dir, tokenizer, feature_extractor)

    else:
        raise ValueError(f"training_phase should be either 'pretraining' or 'finetuning', but got {training_phase}")
    

    # TODO REMOVE
    assert isinstance(ds["val"], datasets.IterableDataset)
    ds["val"] = ds["val"].take(10)

    collator = audiocap.preprocess.DataCollatorAudioSeq2SeqWithPadding(tokenizer, feature_extractor)

    # compute_metrics = audiocap.metrics.CaptioningMetrics(
    #     tokenizer, 
    #     expected_captions=expected_captions,
    #     expected_alternatives=expected_alternatives,
    #     ds_captions_size=len(ds["val"])
    # )

    wandb.init(
        project="audio-captioning",
        tags=["supervised", architecture_name],
        save_code=True,
        config={
            "model": architecture_name,
            "use_pretrained_whisper_encoder": use_pretrained_encoder,
            "use_pretrained_whisper_decoder": use_pretrained_decoder,
        },
        # group="", # for organizing runs
        # dir="", # change for some tmp dir if you need
    )

    assert wandb.run is not None
    training_args_dict_preset = {"output_dir": checkpoint_dir_root / wandb.run.name}
    training_args_dict = {**training_args_dict_preset, **training_args_dict}
    training_args = transformers.Seq2SeqTrainingArguments(**training_args_dict)

    # TODO add ds_mini to log predictions

    # callback_log_val_preds = audiocap.callbacks.WandbPredictionLogger(
    #     log_prefix="val",
    #     dataset=ds_mini["val_mini"],
    #     collator=collator,
    #     log_every_n_steps=log_preds_every_n_steps,
    #     generate_kwargs={"max_length": training_args_dict["generation_max_length"]},
    # )

    # callback_log_dev_preds = audiocap.callbacks.WandbPredictionLogger(
    #     log_prefix="dev",
    #     dataset=ds_mini["dev_mini"],
    #     collator=collator,
    #     log_every_n_steps=log_preds_every_n_steps,
    #     generate_kwargs={"max_length": training_args_dict["generation_max_length"]},
    # )

    callbacks: list[transformers.TrainerCallback]
    callbacks = []
    #callbacks = [callback_log_val_preds, callback_log_dev_preds]
    
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
        #compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        args=training_args,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(str(pathlib.Path(trainer.args.output_dir) / "final"))


def get_whisper_model(
    config_name: str,
    config: transformers.WhisperConfig,
    use_pretrained_whisper_encoder: bool,
    use_pretrained_whisper_decoder: bool,
) -> WhisperForAudioCaptioning:
    
    if use_pretrained_whisper_encoder and use_pretrained_whisper_decoder:
        model = WhisperForAudioCaptioning.from_pretrained(config_name)
        assert isinstance(model, WhisperForAudioCaptioning)
        return model
    
    if not use_pretrained_whisper_encoder and not use_pretrained_whisper_decoder:
        return WhisperForAudioCaptioning(config)
    
    model_pretrained = WhisperForAudioCaptioning.from_pretrained(config_name)
    assert isinstance(model_pretrained, WhisperForAudioCaptioning)
    model = WhisperForAudioCaptioning(config)

    if use_pretrained_whisper_encoder:
        model.model.encoder = model_pretrained.get_encoder()

    if use_pretrained_whisper_decoder:
        model.model.decoder = model_pretrained.get_decoder()
    
    del model_pretrained
    return model



# TODO assert keys

    # expected_keys = {'caption_colname',
    #                  'caption',
    #                  'path',
    #                  'audio_array',
    #                  'sampling_rate',
    #                  'filename',
    #                  'input_features',
    #                  'labels',
    #                  'forced_ac_decoder_ids'}


# def get_expected_lists(jsonl_path: pathlib.Path) -> tuple[list[str], list[list[str]]]:
#     df = pd.read_json(jsonl_path, lines=True)
#     expected_captions = df["caption1"].tolist()
#     expected_alternatives = df[[c for c in df.columns if c.startswith("caption")]].values.tolist()

#     return expected_captions, expected_alternatives


if __name__ == "__main__":
    app()
