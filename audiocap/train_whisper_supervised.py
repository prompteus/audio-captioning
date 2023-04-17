from __future__ import annotations

import pathlib
from typing import Optional, Tuple

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
    architecture_name: str = typer.Option(..., help="Name of the config to use for the model, e.g.: 'openai/whisper-small'"),
    checkpoint_dir_root: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory where checkpoints will be saved"),
    training_phase: str = typer.Option(..., help="Name of the training phase, either pretraining or finetuning"),
    use_pretrained_whisper_encoder: bool = typer.Option(..., help="Use the pretrained encoder from OpenAI Whisper"),
    use_pretrained_whisper_decoder: bool = typer.Option(..., help="Use the pretrained decoder from OpenAI Whisper"),
    clotho_dir: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Clotho dataset"),
    audioset_dir: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Audioset dataset"),
    audiocaps_dir: pathlib.Path = typer.Option(..., dir_okay=True, file_okay=False, readable=True, help="Path to the directory with the Audiocaps dataset"),
    num_dev_examples_to_log_preds: int = typer.Option(10, help="Number of development examples to log predictions for during training"),
    num_val_examples_to_log_preds: int = typer.Option(32, help="Number of validation examples to log predictions for during training"),
    log_preds_every_n_steps: int = typer.Option(..., help="Log predictions every n steps"),
    limit_dev_split_size: Optional[int] = typer.Option(None, help="Limit the dev split size (for debugging purposes)"),
    limit_val_split_size: Optional[int] = typer.Option(..., help="Limit the val split size (for debugging purposes)"),
    should_early_stop: bool = typer.Option(False),
    early_stopping_patience: Optional[int] = typer.Option(None),
    early_stopping_threshold: Optional[float] = typer.Option(None),
    training_args_config: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="yaml file with the training arguments"),
) -> None:
    
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    config = transformers.WhisperConfig.from_pretrained(architecture_name)
    model = transformers.WhisperConfig.from_pretrained(architecture_name)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture_name)
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_name)
    model = get_whisper_model(architecture_name, config, use_pretrained_whisper_encoder, use_pretrained_whisper_decoder)
    
    if training_phase == "pretraining":
        # prepare audioset dataset
        ds_audioset, ds_audioset_mini = get_audioset_dataset(audioset_dir,
                                                            tokenizer,
                                                            feature_extractor,
                                                            num_dev_examples_to_log_preds,
                                                            num_val_examples_to_log_preds,
                                                            limit_dev_split_size,
                                                            limit_val_split_size,
                                                            )

        # prepare audiocaps dataset
        ds_audiocaps, ds_audiocaps_mini = get_audiocaps_dataset(audiocaps_dir,
                                                                tokenizer,
                                                                feature_extractor,
                                                                num_dev_examples_to_log_preds,
                                                                num_val_examples_to_log_preds,
                                                                limit_dev_split_size,
                                                                limit_val_split_size,
                                                                )
        expected_captions, expected_alternatives = get_expected_lists(audiocaps_dir / "valid/metadata.jsonl")


        # TODO: MIXING AUDIOSET / AUDIOCAPS DATASETS
        # ??? SHOULDNT WE PUT ALL OF THIS AWAY FROM MAIN AND ONLY PUT CERTAIN DATASETS TO MAIN FOR TRAINING? LIKE
        # DS AND MINI_DS? THIS IS TOO MUCH.. WHERE WE DECIDE ON FINETUNING VS. PRETRAINING?

        # ds, ds_mini = ... # TODO  

    elif training_phase == "finetuning":
        # prepare clotho dataset
        ds, ds_mini = get_clotho_dataset(clotho_dir,
                                                    tokenizer,
                                                    feature_extractor,
                                                    num_dev_examples_to_log_preds,
                                                    num_val_examples_to_log_preds,
                                                    limit_dev_split_size,
                                                    limit_val_split_size,
                                                    )
        expected_captions, expected_alternatives = get_expected_lists(clotho_dir / "validation/metadata.jsonl")



    else:
        raise ValueError(f"training_phase should be either 'pretraining' or 'finetuning', but got {training_phase}")



    collator = audiocap.preprocess.DataCollatorAudioSeq2SeqWithPadding(tokenizer, feature_extractor)
    compute_metrics = audiocap.metrics.CaptioningMetrics(tokenizer, 
                                                         expected_caption=expected_captions,
                                                         expected_alternatives=expected_alternatives)

    wandb.init(
        project="audio-captioning",
        tags=["supervised", architecture_name],
        save_code=True,
        config={
            "model": architecture_name,
            "use_pretrained_whisper_encoder": use_pretrained_whisper_encoder,
            "use_pretrained_whisper_decoder": use_pretrained_whisper_decoder,
        },
        # group="", # for organizing runs
        # dir="", # change for some tmp dir if you need
    )

    training_args_dict_preset = {"output_dir": checkpoint_dir_root / wandb.run.name}
    with open(training_args_config, "r") as f:
        training_args_dict = yaml.safe_load(f)
    training_args_dict = {**training_args_dict_preset, **training_args_dict}
    training_args = transformers.Seq2SeqTrainingArguments(**training_args_dict)

    callback_log_val_preds = audiocap.callbacks.WandbPredictionLogger(
        log_prefix="val",
        dataset=ds_mini["val_mini"],
        collator=collator,
        log_every_n_steps=log_preds_every_n_steps,
        generate_kwargs={"max_length": training_args_dict["generation_max_length"]},
    )

    callback_log_dev_preds = audiocap.callbacks.WandbPredictionLogger(
        log_prefix="dev",
        dataset=ds_mini["dev_mini"],
        collator=collator,
        log_every_n_steps=log_preds_every_n_steps,
        generate_kwargs={"max_length": training_args_dict["generation_max_length"]},
    )

    callbacks = [callback_log_val_preds, callback_log_dev_preds]
    
    if should_early_stop:
        if early_stopping_patience is None:
            raise ValueError("early_stopping_patience must be specified if should_early_stop is True")
        early_stopping_kwargs = dict(early_stopping_patience=early_stopping_patience)
        if early_stopping_threshold is not None:
            early_stopping_kwargs["early_stopping_threshold"] = early_stopping_threshold
        early_stopping = transformers.EarlyStoppingCallback(**early_stopping_kwargs)
        callbacks.append(early_stopping)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        train_dataset=ds["dev"],
        eval_dataset=ds["val"],
        args=training_args,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(trainer.args.output_dir / "final")


def get_whisper_model(
    config_name: str,
    config: transformers.WhisperConfig,
    use_pretrained_whisper_encoder: bool,
    use_pretrained_whisper_decoder: bool,
) -> WhisperForAudioCaptioning:
    
    if use_pretrained_whisper_encoder and use_pretrained_whisper_decoder:
        return WhisperForAudioCaptioning.from_pretrained(config_name)
    
    if not use_pretrained_whisper_encoder and not use_pretrained_whisper_decoder:
        return WhisperForAudioCaptioning(config)
    
    model_pretrained = WhisperForAudioCaptioning.from_pretrained(config_name)
    model = WhisperForAudioCaptioning(config)

    if use_pretrained_whisper_encoder:
        model.model.encoder = model_pretrained.get_encoder()

    if use_pretrained_whisper_decoder:
        model.model.decoder = model_pretrained.get_decoder()
    
    del model_pretrained
    return model

def get_expected_lists(jsonl_path: pathlib.Path) -> Tuple[List[str], List[str]]:
    df = pd.read_json(jsonl_path, lines=True)
    expected_captions = df["caption1"].tolist()
    expected_alternatives = df[[c for c in df.columns if c.startswith("caption")]].values.tolist()

    return expected_captions, expected_alternatives

def get_clotho_dataset(clotho_dir: pathlib.Path,
                       tokenizer: transformers.PreTrainedTokenizer,
                       feature_extractor: transformers.PreTrainedFeatureExtractor,
                       num_dev_examples_to_log_preds: int,
                       num_val_examples_to_log_preds: int,
                       limit_dev_split_size: Optional[int],
                       limit_val_split_size: Optional[int],
                       ) -> Tuple[datasets.IterableDatasetDict, datasets.DatasetDict]:
    """Get the Clotho dataset as a dict of datasets.Dataset objects.
    Parameters
    ----------
    clotho_dir : pathlib.Path
        Path to the directory containing the Clotho dataset.
    num_dev_examples_to_log_preds : int
        Number of development examples to log predictions for.
    num_val_examples_to_log_preds : int
        Number of validation examples to log predictions for.
    
    Returns
    -------
    ds : datasets.IterableDatasetDict
        Clotho dataset as a dict of datasets.Dataset objects.
    ds_mini : datasets.DatasetDict
        Clotho dataset as a dict of datasets.Dataset objects, but with only a subset of the examples.
    """

    # create dict string: Dataset
    ds_raw = datasets.load_dataset(
        "audiofolder",
        data_files={
            "dev": f"{clotho_dir}/development/*",
            "val": f"{clotho_dir}/validation/*",
            "test": f"{clotho_dir}/evaluation/*",
            "dev_mini": f"{clotho_dir}/development/*",
            "val_mini": f"{clotho_dir}/validation/*",
        }
    )

    random_gen = np.random.default_rng(seed=1)
    dev_log_indices = random_gen.choice(len(ds_raw["dev"]), size=num_dev_examples_to_log_preds, replace=False)
    val_log_indices = random_gen.choice(len(ds_raw["val"]), size=num_val_examples_to_log_preds, replace=False)

    ds_raw["dev_mini"] = ds_raw["dev"].select(dev_log_indices)
    ds_raw["val_mini"] = ds_raw["val"].select(val_log_indices)

    preprocessing = audiocap.preprocess.Preprocess(tokenizer, feature_extractor)

    ds = datasets.IterableDatasetDict()
    ds_mini = datasets.DatasetDict()
    for split_name in ds_raw.keys():
        # TODO add augmentations, but only to development split
        split = (ds_raw["dev"]
                 .to_iterable_dataset()
                 .map(
                     audiocap.preprocess.flatten_captions,
                     batched=True,
                     batch_size=10,
                     remove_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
                 )
                 .map(
                     preprocessing,
                     batched=True,
                     batch_size=16,
                     remove_columns=["audio", "prefix"],
                     fn_kwargs={"source_ds": "clotho", "task": "caption"},
                 )
                 )

        if "dev" in split_name:
            split = split.shuffle(seed=42, buffer_size=100)

        if "mini" in split_name:
            # there are multiple rows per each audio clip (because of multiple captions)
            # we want to keep only one from each for logging predictions
            ds_mini[split_name] = datasets.Dataset.from_list(list({x["filename"]: x for x in split}.values()))
        else:
            ds[split_name] = split
    
    if limit_dev_split_size is not None:
        ds["dev"] = ds["dev"].take(limit_dev_split_size)
    if limit_val_split_size is not None:
        ds["val"] = ds["val"].take(limit_val_split_size)

    expected_keys = {'caption_idx',
                     'caption',
                     'path',
                     'audio_array',
                     'sampling_rate',
                     'filename',
                     'input_features',
                     'labels',
                     'forced_ac_decoder_ids'}
    assert set(ds_mini["dev_mini"][0].keys()) == expected_keys, f"actual keys: {set(ds_mini['dev_mini'][0].keys())}\nsymetric sifference: {set(ds_mini['dev_mini'][0].keys()).symmetric_difference(expected_keys)}"

    return ds, ds_mini


def get_audioset_dataset(audioset_dir: pathlib.Path,
                         tokenizer: transformers.PreTrainedTokenizer,
                         feature_extractor: transformers.PreTrainedFeatureExtractor,
                         num_dev_examples_to_log_preds: int,
                         num_val_examples_to_log_preds: int,
                         limit_dev_split_size: Optional[int],
                         limit_val_split_size: Optional[int],
                         ) -> Tuple[datasets.IterableDatasetDict, datasets.DatasetDict]:
    """Get the Audioset dataset as a dict of datasets.Dataset objects.
    Parameters
    ----------
    audioset_dir : pathlib.Path
        Path to the directory containing the Audioset dataset.
    num_dev_examples_to_log_preds : int
        Number of development examples to log predictions for.
    num_val_examples_to_log_preds : int
        Number of validation examples to log predictions for.

    Returns
    -------
    ds : datasets.IterableDatasetDict
        Audioset dataset as a dict of datasets.Dataset objects.
    ds_mini : datasets.DatasetDict
        Audioset dataset as a dict of datasets.Dataset objects, but with only a subset of the examples.
    """

    # create dict string: Dataset CHANGE THE FOLDER NAMES!!!
    ds_raw = datasets.load_dataset(
        "audiofolder",
        data_files={
            "dev": f"{audioset_dir}/train_balanced/*",
            "val": f"{audioset_dir}/train_unbalanced/*",
            "test": f"{audioset_dir}/eval/*",
            "dev_mini": f"{audioset_dir}/train_balanced/*",
            "val_mini": f"{audioset_dir}/eval/*",
        }
    )

    random_gen = np.random.default_rng(seed=1)
    dev_log_indices = random_gen.choice(len(ds_raw["dev"]), size=num_dev_examples_to_log_preds, replace=False)
    val_log_indices = random_gen.choice(len(ds_raw["val"]), size=num_val_examples_to_log_preds, replace=False)

    ds_raw["dev_mini"] = ds_raw["dev"].select(dev_log_indices)
    ds_raw["val_mini"] = ds_raw["val"].select(val_log_indices)

    preprocessing = audiocap.preprocess.Preprocess(tokenizer, feature_extractor)

    ds = datasets.IterableDatasetDict()
    ds_mini = datasets.DatasetDict()
    for split_name in ds_raw.keys():
        # TODO add augmentations, but only to development split
        # TODO add convert to captions from Marek's file audioset_tools.py -> AudiosetOntology.audioset_label_ids_to_str
        split = (ds_raw[split_name]
                 .to_iterable_dataset()
                 # .map(
                 #     audiocap.preprocess.audioset_label_ids_to_str???,
                 #     batched=True,
                 #     batch_size=10,
                 # )
                 .map(
                     preprocessing,
                     batched=True,
                     batch_size=16,
                     remove_columns=["audio", "prefix"],
                     fn_kwargs={"ds_name": "audioset", "task": "keywords"},
                 )
                 )

        if "dev" in split_name:
            split = split.shuffle(seed=42, buffer_size=100)

        if "mini" in split_name:
            # there are multiple rows per each audio clip (because of multiple captions)
            # we want to keep only one from each for logging predictions
            ds_mini[split_name] = datasets.Dataset.from_list(list({x["filename"]: x for x in split}.values()))
        else:
            ds[split_name] = split

    if limit_dev_split_size is not None:
        ds["dev"] = ds["dev"].take(limit_dev_split_size)
    if limit_val_split_size is not None:
        ds["val"] = ds["val"].take(limit_val_split_size)

    expected_keys = {'caption',
                     'caption_idx',
                     'path',
                     'audio_array',
                     'sampling_rate',
                     'filename',
                     'input_features',
                     'labels',
                     'forced_ac_decoder_ids'}
    assert set(ds_mini["dev_mini"][0].keys()) == expected_keys, f"actual keys: {set(ds_mini['dev_mini'][0].keys())}\nsymetric sifference: {set(ds_mini['dev_mini'][0].keys()).symmetric_difference(expected_keys)}"

    return ds, ds_mini


def get_audiocaps_dataset(audiocaps_dir: pathlib.Path,
                          tokenizer: transformers.PreTrainedTokenizer,
                          feature_extractor: transformers.PreTrainedFeatureExtractor,
                          num_dev_examples_to_log_preds: int,
                          num_val_examples_to_log_preds: int,
                          limit_dev_split_size: Optional[int],
                          limit_val_split_size: Optional[int],
                          ) -> Tuple[datasets.IterableDatasetDict, datasets.DatasetDict]:
    """Get the Audiocaps dataset as a dict of datasets.Dataset objects.
    Parameters
    ----------
    audiocaps_dir : pathlib.Path
        Path to the directory containing the Audiocaps dataset.
    num_dev_examples_to_log_preds : int
        Number of development examples to log predictions for.
    num_val_examples_to_log_preds : int
        Number of validation examples to log predictions for.
    
    Returns
    -------
    ds : datasets.IterableDatasetDict
        Audiocaps dataset as a dict of datasets.Dataset objects.
    ds_mini : datasets.DatasetDict
        Audiocaps dataset as a dict of datasets.Dataset objects, but with only a subset of the examples.
    """

    # create dict string: Dataset
    ds_raw = datasets.load_dataset(
        "audiofolder",
        data_files={
            "dev": f"{audiocaps_dir}/train/*",
            "val": f"{audiocaps_dir}/valid/*",
            "test": f"{audiocaps_dir}/test/*",
            "dev_mini": f"{audiocaps_dir}/train/*",
            "val_mini": f"{audiocaps_dir}/valid/*",
        }
    )

    random_gen = np.random.default_rng(seed=1)
    dev_log_indices = random_gen.choice(len(ds_raw["dev"]), size=num_dev_examples_to_log_preds, replace=False)
    val_log_indices = random_gen.choice(len(ds_raw["val"]), size=num_val_examples_to_log_preds, replace=False)

    ds_raw["dev_mini"] = ds_raw["dev"].select(dev_log_indices)
    ds_raw["val_mini"] = ds_raw["val"].select(val_log_indices)

    preprocessing = audiocap.preprocess.Preprocess(tokenizer, feature_extractor)

    ds = datasets.IterableDatasetDict()
    ds_mini = datasets.DatasetDict()
    for split_name in ds_raw.keys():
        # TODO add augmentations, but only to development split
        if "dev" in split_name:
            split = (ds_raw[split_name]
                     .to_iterable_dataset()
                     .map(
                        preprocessing,
                        batched=True,
                        batch_size=16,
                        remove_columns=["audio", "prefix"],
                        fn_kwargs={"source_ds": "audiocaps", "task": "caption"},
                     )
                     .shuffle(seed=42, buffer_size=100)
                     )
        else:
            split = (ds_raw[split_name]
                     .to_iterable_dataset()
                     .map(
                         audiocap.preprocess.flatten_captions,
                         batched=True,
                         batch_size=10,
                         remove_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
                     )
                     .map(
                         preprocessing,
                         batched=True,
                         batch_size=16,
                         remove_columns=["audio", "prefix"],
                         fn_kwargs={"source_ds": "audiocaps", "task": "caption"},
                     )
                     )

        if "mini" in split_name:
            # there are multiple rows per each audio clip (because of multiple captions)
            # we want to keep only one from each for logging predictions
            ds_mini[split_name] = datasets.Dataset.from_list(list({x["filename"]: x for x in split}.values()))
        else:
            ds[split_name] = split
    
    if limit_dev_split_size is not None:
        ds["dev"] = ds["dev"].take(limit_dev_split_size)
    if limit_val_split_size is not None:
        ds["val"] = ds["val"].take(limit_val_split_size)

    expected_keys = {'caption_idx',
                     'caption',
                     'path',
                     'audio_array',
                     'sampling_rate',
                     'filename',
                     'input_features',
                     'labels',
                     'forced_ac_decoder_ids'}
    assert set(ds_mini["dev_mini"][0].keys()) == expected_keys, f"actual keys: {set(ds_mini['dev_mini'][0].keys())}\nsymetric sifference: {set(ds_mini['dev_mini'][0].keys()).symmetric_difference(expected_keys)}"

    return ds, ds_mini



if __name__ == "__main__":
    app()
