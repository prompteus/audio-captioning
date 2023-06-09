from __future__ import annotations

import sys
import pathlib
import functools
import dataclasses
import collections
import multiprocessing
from typing import Literal, Callable

import librosa
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torchdata.datapipes as dp
import transformers

import audiocap


def set_cols(colname: str | tuple[str, ...] | list[str], func: Callable) -> Callable:
    def _func(row: dict):
        row = row.copy()
        if isinstance(colname, str):
            row[colname] = func(row)
        else:
            for col, val in zip(colname, func(row)):
                row[col] = val
        return row
    return _func


def del_cols(*args: str | tuple[str, ...] | list[str]) -> Callable:
    colnames = []
    for arg in args:
        if isinstance(arg, str):
            colnames.append(arg)
        elif isinstance(arg, tuple) or isinstance(arg, list):
            colnames.extend(arg)
    def _func(row: dict):
        return {
            col: val for col, val in row.items()
            if col not in colnames
        }
    return _func


def rename_col(mapper: dict[str, str]) -> Callable:
    def _func(row: dict):
        row = row.copy()
        for old_colname, new_colname in mapper.items():
            row[new_colname] = row.pop(old_colname)
        return row
    return _func


def explode_col(colnames: list[str], new_name: str, name_keep_in: str) -> Callable:
    def _func(row: dict):
        row = row.copy()
        caption_cols = {colname: row.pop(colname) for colname in colnames}
        return [
            {**row, name_keep_in: colname, new_name: caption }
            for colname, caption in caption_cols.items()
        ]
    return _func
    

class PrepareLabels:
    def __init__(self, tokenizer: transformers.WhisperTokenizer) -> None:
        self.tokenizer = tokenizer
    
    def __call__(self, prefix: str, caption: str) -> tuple[list[int], list[int]]:
        forced_ac_decoder_ids = self.tokenizer("", text_target=prefix, add_special_tokens=False).labels
        *fluff_tokens, eos = self.tokenizer("", text_target="", add_special_tokens=True).labels
        labels = self.tokenizer("", text_target=caption, add_special_tokens=False).labels
        labels = fluff_tokens + forced_ac_decoder_ids + labels + [eos]
        return labels, forced_ac_decoder_ids


class PreprocessAudio:
    def __init__(self, feature_extractor: transformers.WhisperFeatureExtractor) -> None:
        self.feature_extractor = feature_extractor
        self.num_features = feature_extractor.feature_size

    def __call__(self, audio_array: np.ndarray, sampling_rate: int) -> torch.Tensor:
        features: torch.Tensor = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features
        return features.reshape(self.num_features, -1)


def librosa_load_safe(path: pathlib.Path, sr: int | None, mono: bool) -> tuple[np.ndarray, int] | tuple[None, int | None]:
    try:
        return librosa.load(path, sr=sr, mono=mono) # type: ignore
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr, flush=True)
        return None, sr


def create_prefix(source_ds: str, task: str) -> str:
    return source_ds + " > " + task + ": "



@dataclasses.dataclass
class AudioFolder:
    path: pathlib.Path | str
    shuffle: bool
    source_ds: str
    task: str
    caption_columns: list[str]
    tokenizer: transformers.WhisperTokenizer
    feature_extractor: transformers.WhisperFeatureExtractor
    handle_multiple_captions: Literal["explode", "keep_first"] | None = None
    prepare_caption: Callable | None = None
    augment_config: audiocap.augment.AugmentConfig | None = None
    shuffle_buffer_size: int = 20
    prefetch: int = 10
    meta_filename: str = "metadata"
    drop_audio_array: bool = True
    sample_n: int | None = None
    seed: int | None = None
    load_as_iterable: bool = True

    meta: pd.DataFrame = dataclasses.field(init=False)
    pipe: dp.iter.IterDataPipe | dp.map.MapDataPipe = dataclasses.field(init=False)
    augmenter: audiocap.augment.Augmenter | None = dataclasses.field(init=False)

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = pathlib.Path(self.path)

        if len(self.caption_columns) > 1 and self.handle_multiple_captions is None:
            raise ValueError(
                "Multiple caption columns found. "
                "Please specify how to handle them using `handle_multiple_captions`."
            )

        self.meta = pd.read_json(self.path / f"{self.meta_filename}.jsonl", lines=True)

        if self.sample_n is not None:
            self.meta = self.meta.sample(n=self.sample_n, random_state=self.seed)
        
        if self.shuffle:
            self.meta = self.meta.sample(frac=1, random_state=self.seed)

        if self.augment_config is not None:
            self.augmenter = audiocap.augment.Augmenter(self.augment_config)
        else: 
            self.augmenter = None

        self.init_pipe()

    def init_pipe(self):
        prepare_labels = PrepareLabels(self.tokenizer)
        extract_features = PreprocessAudio(self.feature_extractor)
        sr = self.feature_extractor.sampling_rate
        prefix = create_prefix(self.source_ds, self.task)

        pipe: dp.iter.IterDataPipe
        pipe = dp.iter.IterableWrapper(self.meta.to_dict("records"), deepcopy=False) # type: ignore

        pipe = (
            pipe
            .sharding_filter()
            .map(set_cols("path", lambda row: self.path / row["file_name"]))
        )

        if self.augmenter is None:
            pipe = (
                pipe
                .map(set_cols(("audio_array", "sampling_rate"), lambda row: librosa_load_safe(row["path"], sr=sr, mono=True)))
                .filter(lambda row: row["audio_array"] is not None)
            )
        else:
            pipe = (
                pipe
                .map(set_cols(("audio_array", "sampling_rate"), lambda row: librosa_load_safe(row["path"], sr=None, mono=True)))
                .filter(lambda row: row["audio_array"] is not None)
                .map(set_cols("audio_array", lambda row: self.augmenter(row["audio_array"], row["sampling_rate"])))
                .map(set_cols("audio_array", lambda row: librosa.resample(row["audio_array"], orig_sr=row["sampling_rate"], target_sr=sr)))
                .map(set_cols("sampling_rate", lambda _: sr))
            )


        pipe = pipe.map(extract_features, ["audio_array", "sampling_rate"], "input_features")
        pipe = pipe.map(del_cols("path"))

        if self.drop_audio_array:
            pipe = pipe.map(del_cols("audio_array"))

        if self.handle_multiple_captions == "explode":
            pipe = pipe.flatmap(explode_col(self.caption_columns, "caption", "caption_colname"))
        else:
            first_col, *rest_cols = self.caption_columns
            pipe = (
                pipe
                .map(rename_col({first_col: "caption"}))
                .map(set_cols("caption_colname", lambda _: first_col))
                .map(del_cols(rest_cols))
            )

        if self.prepare_caption is not None:
            pipe = pipe.map(self.prepare_caption, input_col="caption")

        if self.shuffle:
            pipe = pipe.shuffle(buffer_size=self.shuffle_buffer_size)

        pipe = (
            pipe
            .map(set_cols("prefix", lambda _: prefix))
            .map(set_cols(("labels", "forced_ac_decoder_ids"), lambda row: prepare_labels(prefix, row["caption"])))
        )

        if self.load_as_iterable:
            self.pipe = pipe.prefetch(self.prefetch)
        else:
            self.pipe = pipe.enumerate().to_map_datapipe()


    def __len__(self):
        if len(self.caption_columns) == 1:
            return len(self.meta)
        if self.handle_multiple_captions == "keep_first":
            return len(self.meta)
        if self.handle_multiple_captions == "explode":
            return len(self.meta) * len(self.caption_columns)
        raise ValueError("Invalid value for `handle_multiple_captions`.")
    
    @functools.cached_property
    def alternative_captions(self) -> dict[str, list[str]]:
        if self.handle_multiple_captions == "explode":
            raise NotImplementedError("Cannot return alternative captions when `handle_multiple_captions` is set to `flatten`.")
        caps = self.meta[self.caption_columns]
        if self.prepare_caption is not None:
            caps = caps.applymap(self.prepare_caption)
        caps = caps.set_index(self.caption_columns[0], drop=False)
        # this will drop values in case there are duplicates
        return {
            caption: alternatives
            for caption, *alternatives in caps.itertuples()
        }


def load_audios_for_predition(
    src: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
    source_ds: str,
    task: str,
    recursive: bool,
    suffixes: tuple[str, ...] = ("mp3", "wav"),
    take_n: int | None = None,
    prefetch: int = 10,
) -> tuple[dp.iter.IterDataPipe, int]:
    
    if source_ds not in ("clotho", "audioset", "audiocaps"):
        raise ValueError(f"Unknown value for `source_ds`: {source_ds}")
    if task not in ("caption", "keywords"):
        raise ValueError(f"Unknown value for `task`: {task}")
    
    src = pathlib.Path(src)

    if src.is_file():
        paths = [src]
    elif recursive:
        paths = [path for path in src.glob("**/*") if path.is_file() and path.suffix in suffixes]
    else:
        paths = [path for path in src.iterdir() if path.is_file() and path.suffix.strip(".") in suffixes]
    
    paths.sort()
    if take_n is not None:
        paths = paths[:take_n]

    num_files = len(paths)
    
    prepare_labels = PrepareLabels(tokenizer)
    extract_features = PreprocessAudio(feature_extractor)
    sr = feature_extractor.sampling_rate
    prefix = create_prefix(source_ds, task)
    _, forced_ac_decoder_ids = prepare_labels(prefix, "")

    pipe: dp.iter.IterDataPipe
    pipe = dp.iter.IterableWrapper([{"path": path} for path in paths], deepcopy=False)
    pipe = (pipe
        .sharding_filter()
        .map(set_cols("file_name", lambda x: pathlib.Path(x["path"]).name))
        .map(set_cols(("audio_array", "sampling_rate"), lambda row: librosa_load_safe(row["path"], sr=sr, mono=True)))
        .map(del_cols("path"))
        .filter(lambda row: row["audio_array"] is not None)
        .map(extract_features, ["audio_array", "sampling_rate"], "input_features")
        .map(set_cols("forced_ac_decoder_ids", lambda _: forced_ac_decoder_ids))
        .prefetch(prefetch)
    )

    return pipe, num_files
    

def load_clotho(
    audiofolder_root: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
    limit_val_split: int | None,
    augment_config: audiocap.augment.AugmentConfig,
    train_mini_size: int,
    val_mini_size: int,
    seed: int,
) -> dict[str, AudioFolder]:
    
    if isinstance(audiofolder_root, str):
        audiofolder_root = pathlib.Path(audiofolder_root)

    ds = {}

    common_args = dict(
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        source_ds="clotho",
        task="caption",
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
    )

    ds["train"] = AudioFolder(
        path=audiofolder_root / "development",
        handle_multiple_captions="explode",
        shuffle=True,
        augment_config=augment_config,
        **common_args,
    )

    ds["train_mini"] = AudioFolder(
        path=audiofolder_root / "development",
        handle_multiple_captions="keep_first",
        shuffle=False,
        augment_config=augment_config,
        sample_n=train_mini_size,
        drop_audio_array=False,
        load_as_iterable=False,
        seed=seed,
        **common_args,
    )

    ds["val"] = AudioFolder(
        path=audiofolder_root / "validation",
        handle_multiple_captions="keep_first",
        shuffle=False,
        augment_config=None,
        sample_n=limit_val_split,
        seed=seed,
        **common_args,
    )

    ds["val_mini"] = AudioFolder(
        path=audiofolder_root / "validation",
        handle_multiple_captions="keep_first",
        shuffle=False,
        augment_config=None,
        sample_n=val_mini_size,
        drop_audio_array=False,
        load_as_iterable=False,
        seed=seed,
        **common_args,
    )

    ds["test"] = AudioFolder(
        path=audiofolder_root / "evaluation",
        handle_multiple_captions="keep_first",
        shuffle=False,
        augment_config=None,
        **common_args,
    )

    return ds


def load_audioset(
    audiofolder_root: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
    limit_val_split: int | None,
    augment_config: audiocap.augment.AugmentConfig,
    train_mini_size: int,
    val_mini_size: int,
    seed: int,
) -> dict[str, AudioFolder]:
    
    if isinstance(audiofolder_root, str):
        audiofolder_root = pathlib.Path(audiofolder_root)

    ontology = audiocap.audioset_tools.AudiosetOntology.from_json_file(audiofolder_root / "ontology.json")

    ds = {}

    common_args = dict(
        caption_columns=["labels"],
        source_ds="audioset",
        task="keywords",
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        prepare_caption=ontology.audioset_label_ids_to_str,
    )

    ds["train"] = AudioFolder(
        path=audiofolder_root / "train",
        shuffle=True,
        augment_config=augment_config,
        **common_args,
    )

    ds["train_mini"] = AudioFolder(
        path=audiofolder_root / "train",
        shuffle=False,
        augment_config=augment_config,
        sample_n=train_mini_size,
        drop_audio_array=False,
        load_as_iterable=False,
        seed=seed,
        **common_args,
    )

    ds["val"] = AudioFolder(
        path=audiofolder_root / "valid",
        shuffle=False,
        augment_config=None,
        sample_n=limit_val_split,
        seed=seed,
        **common_args,
    )

    ds["val_mini"] = AudioFolder(
        path=audiofolder_root / "valid",
        shuffle=False,
        augment_config=None,
        sample_n=val_mini_size,
        drop_audio_array=False,
        load_as_iterable=False,
        seed=seed,
        **common_args,
    )

    ds["test"] = AudioFolder(
        path=audiofolder_root / "test",
        shuffle=False,
        augment_config=None,
        **common_args,
    )

    return ds


def load_audiocaps(
    audiofolder_root: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
    limit_val_split: int | None,
    augment_config: audiocap.augment.AugmentConfig,
    train_mini_size: int,
    val_mini_size: int,
    seed: int,
) -> dict[str, AudioFolder]:
    
    if isinstance(audiofolder_root, str):
        audiofolder_root = pathlib.Path(audiofolder_root)

    ds = {}

    common_args = dict(
        source_ds="audiocaps",
        task="caption",
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
    )

    ds["train"] = AudioFolder(
        path=audiofolder_root / "train",
        caption_columns=["caption"],
        shuffle=True,
        augment_config=augment_config,
        **common_args,
    )

    ds["train_mini"] = AudioFolder(
        path=audiofolder_root / "train",
        caption_columns=["caption"],
        shuffle=False,
        augment_config=augment_config,
        sample_n=train_mini_size,
        drop_audio_array=False,
        load_as_iterable=False,
        seed=seed,
        **common_args,
    )

    ds["val"] = AudioFolder(
        path=audiofolder_root / "valid",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        handle_multiple_captions="keep_first",
        shuffle=False,
        augment_config=None,
        sample_n=limit_val_split,
        seed=seed,
        **common_args,
    )

    ds["val_mini"] = AudioFolder(
        path=audiofolder_root / "valid",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        handle_multiple_captions="keep_first",
        shuffle=False,
        augment_config=None,
        sample_n=val_mini_size,
        drop_audio_array=False,
        load_as_iterable=False,
        seed=seed,
        **common_args,
    )

    ds["test"] = AudioFolder(
        path=audiofolder_root / "test",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        handle_multiple_captions="keep_first",
        shuffle=False,
        augment_config=None,
        **common_args,
    )

    return ds


def load_dataset_mixture(
    clotho_dir: pathlib.Path,
    audioset_dir: pathlib.Path,
    audiocaps_dir: pathlib.Path,
    dataset_weights: dict[str, float],
    datasets_val_limits: dict[str, int | None],
    log_preds_num_train: int,
    log_preds_num_valid: int,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
    augment_config: audiocap.augment.AugmentConfig,
):
    audiofolders: list[dict[str, audiocap.data.AudioFolder]] = []

    if clotho_dir is not None and dataset_weights["clotho"] > 0.000001:
        audiofolders.append(
            audiocap.data.load_clotho(clotho_dir, tokenizer, feature_extractor, datasets_val_limits["clotho"], augment_config, log_preds_num_train, log_preds_num_valid, seed=0)
        )

    if audioset_dir is not None and dataset_weights["audioset"] > 0.000001:
        audiofolders.append(
            audiocap.data.load_audioset(audioset_dir, tokenizer, feature_extractor, datasets_val_limits["audioset"], augment_config, log_preds_num_train, log_preds_num_valid, seed=0)
        )

    if audiocaps_dir is not None and dataset_weights["audiocaps"] > 0.000001:
        audiofolders.append(
            audiocap.data.load_audiocaps(audiocaps_dir, tokenizer, feature_extractor, datasets_val_limits["audiocaps"], augment_config, log_preds_num_train, log_preds_num_valid, seed=0)
        )

    if len(audiofolders) == 0:
        raise ValueError("No dataset specified")

    dataset = {}

    dataset["train"] = dp.iter.SampleMultiplexer({
        af["train"].pipe.cycle(): dataset_weights[af["train"].source_ds]
        for af in audiofolders
    })

    for split in ["val", "test"]:
        dataset[split] = dp.iter.Concater(*[af[split].pipe for af in audiofolders])
    for split in ["train_mini", "val_mini"]:
        dataset[split] = dp.map.Concater(*[af[split].pipe for af in audiofolders])


    ds_val_alternatives = {
        (af["val"].source_ds, af["val"].task) : af["val"].alternative_captions
        for af in audiofolders
    }

    return dataset, audiofolders, ds_val_alternatives


class DataCollatorAudioSeq2SeqWithPadding:

    def __init__(
        self,
        tokenizer: transformers.WhisperTokenizer,
        feature_extractor: transformers.WhisperFeatureExtractor,
        keep_cols: tuple[str, ...] = tuple(),
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.keep_cols = keep_cols

    def __call__(
        self,
        orig_batch: list[dict],
    ) -> collections.UserDict:
        
        batch_features = [{"input_features": x["input_features"]} for x in orig_batch]
        batch_forced_ac_decoder_ids = [x["forced_ac_decoder_ids"] for x in orig_batch]

        batch = self.feature_extractor.pad(batch_features, return_tensors="pt")
        batch["forced_ac_decoder_ids"] = torch.tensor(batch_forced_ac_decoder_ids)

        if "labels" in orig_batch[0]:
            batch_labels = [{"input_ids": x["labels"]} for x in orig_batch]
            batch_labels = self.tokenizer.pad(batch_labels, return_tensors="pt")
            # replace padding with -100 to ignore loss correctly
            labels = batch_labels["input_ids"].masked_fill(batch_labels.attention_mask != 1, -100)
            if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels

        for col in self.keep_cols:
            batch[col] = torch.utils.data.default_collate([x[col] for x in orig_batch])

        return batch


def find_corrupted_audios(folder: pathlib.Path | str, extension: str, num_workers: int) -> list[pathlib.Path]:
    folder = pathlib.Path(folder)
    corrupted = []
    with multiprocessing.Pool(num_workers) as pool:
        files = list(folder.glob(f"**/*.{extension}"))
        print("found total files:", len(files))
        for path in files:
            if path.is_file():
                try:
                    pool.apply_async(librosa.load, args=(path,), kwds={"sr": None})
                except:
                    corrupted.append(path)
    print("found corrupted files:", len(corrupted))
    return corrupted

