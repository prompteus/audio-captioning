from __future__ import annotations

import pathlib
import functools
import dataclasses
from typing import Any, Literal, Callable, Iterable

import datasets
import librosa
import pandas as pd
import numpy as np
import torchdata.datapipes as dp
import transformers

import audiocap


class OurInterleave:
    def __init__(
        self,
        ds_list: list[datasets.IterableDataset],
        stop_on_first_end: bool,
        seed: int | None = None,
        probs: list[float] | np.ndarray | None = None,
    ) -> None:
        self.ds_list = ds_list

        if probs is None:
            probs = np.ones(len(ds_list), dtype=np.float64) / len(ds_list)
        self.probs = np.array(probs, dtype=np.float64)

        if len(self.probs) != len(self.ds_list):
            raise ValueError("probs and datasets must have the same length")
        
        self.stop_on_first_end = stop_on_first_end
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def __iter__(self):
        iterators = [iter(ds) for ds in self.ds_list]
        going = np.array([True] * len(self.ds_list))
        probs = self.probs
        while True:
            if not np.any(going):
                return
            idx = self.rng.choice(len(self.ds_list), p=probs)
            try:
                yield next(iterators[idx])
            except StopIteration:
                if self.stop_on_first_end:
                    return
                going[idx] = False
                probs = self.probs * going
                probs /= probs.sum()

    def __call__(self) -> Iterable:
        return self
    
    def to_iterable_dataset(self) -> datasets.IterableDataset:
        return datasets.IterableDataset.from_generator(self)


def interleave_datasets(
    ds_list: list[datasets.IterableDataset | datasets.Dataset],
    stop_on_first_end: bool,
    seed: int | None = None,
    probs: list[float] | np.ndarray | None = None,
) -> datasets.IterableDataset:
    iterable_datasets = [ds if isinstance(ds, datasets.IterableDataset) else ds.to_iterable_dataset() for ds in ds_list]
    return OurInterleave(iterable_datasets, stop_on_first_end, seed, probs).to_iterable_dataset()




def add_cols(out_col: str | list[str] | dict, func = None, input_cols: str | list[str] | None = None):
    def _func(row):
        nonlocal input_cols

        if isinstance(out_col, dict):
            assert func is None
            assert input_cols is None
            row = row.copy()
            for col, val in out_col.items():
                row[col] = val
            return row

        assert func is not None
        if input_cols is None:
            outputs = func(row)
        elif isinstance(input_cols, str):
            outputs = func(row[input_cols])
        else:
            outputs = func(*[row[c] for c in input_cols])

        if isinstance(out_col, str):
            row[out_col] = outputs
        else:
            for col, output in zip(out_col, outputs):
                row[col] = output
        return row
    return _func


def rename_cols(mapper: dict[str, str]):
    def _func(row: dict):
        row = row.copy()
        for old_colname, new_colname in mapper.items():
            row[new_colname] = row.pop(old_colname)
        return row
    return _func


def delete_cols(colnames: list[str]):
    def _func(row: dict):
        row = row.copy()
        for colname in colnames:
            del row[colname]
        return row
    return _func


def flatten_captions(row: dict, colnames: list[str]) -> list[dict]:
    row = row.copy()
    caption_cols = {colname: row.pop(colname) for colname in colnames}
    return [
        {**row, "caption_colname": colname, "caption": caption }
        for colname, caption in caption_cols.items()
    ]


class Preprocessing:
    def __init__(
        self,
        tokenizer: transformers.WhisperTokenizer,
        feature_extractor: transformers.WhisperFeatureExtractor,
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.num_features = feature_extractor.feature_size

    def __call__(self, row: dict) -> dict:
        row = row.copy()
        features = self.feature_extractor(row["audio_array"], sampling_rate=row["sampling_rate"], return_tensors="pt")
        row["input_features"] = features.input_features.reshape(self.num_features, -1)
        row["prefix"] = row["source_ds"] + " > " + row["task"] + ": "
        row["forced_ac_decoder_ids"] = self.tokenizer("", text_target=row["prefix"], add_special_tokens=False).labels
        *fluff_tokens, eos = self.tokenizer("", text_target="", add_special_tokens=True).labels
        labels = self.tokenizer("", text_target=row["caption"], add_special_tokens=False).labels
        row["labels"] = fluff_tokens + row["forced_ac_decoder_ids"] + labels + [eos]
        return row


@dataclasses.dataclass
class AudioFolder:
    path: pathlib.Path | str
    shuffle: bool
    source_ds: str
    task: str
    tokenizer: transformers.WhisperTokenizer
    feature_extractor: transformers.WhisperFeatureExtractor
    handle_multiple_captions: str | None = None
    caption_columns: list[str] | None = None
    prepare_caption: Callable | None = None
    shuffle_buffer_size: int = 10 # TODO
    meta_filename: str = "metadata"

    meta: pd.DataFrame = dataclasses.field(init=False)
    pipe: dp.iter.IterDataPipe = dataclasses.field(init=False)

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = pathlib.Path(self.path)

        self.meta = pd.read_json(self.path / f"{self.meta_filename}.jsonl", lines=True)

        if self.caption_columns is None:
            self.caption_columns = [c for c in self.meta.columns if str(c).startswith("caption")]

        if len(self.caption_columns) > 1 and self.handle_multiple_captions is None:
            raise ValueError("Multiple caption columns found. Please specify how to handle them using `handle_multiple_captions`.")

        self.init_pipe()

    def init_pipe(self):
        self.preprocess = Preprocessing(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
        )

        sampling_rate = self.feature_extractor.sampling_rate
        assert self.caption_columns is not None

        #rows = (row._asdict() for row in self.meta.itertuples(index=False))

        pipe: dp.iter.IterDataPipe
        pipe = dp.iter.IterableWrapper(self.meta.to_dict("records")) # type: ignore
        pipe = pipe.sharding_filter()
        pipe = pipe.map(add_cols("path", lambda row: self.path / row["file_name"]))
        pipe = pipe.map(add_cols(["audio_array", "sampling_rate"], lambda row: librosa.load(row["path"], sr=sampling_rate, mono=True)))

        if self.handle_multiple_captions == "flatten":
            pipe = pipe.flatmap(lambda row: flatten_captions(row, self.caption_columns))
        else:
            first, *rest = self.caption_columns
            pipe = pipe.map(rename_cols({first: "caption"}))
            pipe = pipe.map(add_cols({"caption_colname": first}))
            pipe = pipe.map(delete_cols(rest))

        if self.prepare_caption is not None:
            pipe = pipe.map(self.prepare_caption, input_col="caption")

        if self.shuffle:
            pipe = pipe.shuffle(buffer_size=self.shuffle_buffer_size)

        pipe = pipe.map(add_cols({"source_ds": self.source_ds, "task": self.task}))
        pipe = pipe.map(self.preprocess)
        pipe = pipe.map(delete_cols(["audio_array", "path", "file_name", "source_ds", "task"]))
        self.pipe = pipe

    def __len__(self):
        if self.handle_multiple_captions == "keep_first":
            return len(self.meta)
        elif self.handle_multiple_captions == "flatten":
            assert self.caption_columns is not None
            return len(self.meta) * len(self.caption_columns)
        raise ValueError("Invalid value for `handle_multiple_captions`.")
    
    @functools.cached_property
    def alternative_captions(self) -> list[list[str]]:
        if self.handle_multiple_captions == "flatten":
            raise NotImplementedError("Cannot return alternative captions when `handle_multiple_captions` is set to `flatten`.")
        caps = self.meta[self.caption_columns]
        if self.prepare_caption is not None:
            caps = caps.applymap(self.prepare_caption)
        return caps.values.tolist()
    

def load_clotho(
    audiofolder_root: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
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
        handle_multiple_captions="flatten",
        shuffle=True,
        **common_args,
    )

    ds["val"] = AudioFolder(
        path=audiofolder_root / "validation",
        handle_multiple_captions="keep_first",
        shuffle=False,
        **common_args,
    )

    ds["val"] = AudioFolder(
        path=audiofolder_root / "evaluation",
        handle_multiple_captions="keep_first",
        shuffle=False,
        **common_args,
    )

    return ds


def load_audioset(
    audiofolder_root: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
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
        **common_args,
    )

    ds["val"] = AudioFolder(
        path=audiofolder_root / "valid",
        shuffle=False,
        **common_args,
    )

    ds["test"] = AudioFolder(
        path=audiofolder_root / "test",
        shuffle=False,
        **common_args,
    )

    return ds


def load_audiocaps(
    audiofolder_root: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
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
        **common_args,
    )

    ds["val"] = AudioFolder(
        path=audiofolder_root / "valid",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        handle_multiple_captions="keep_first",
        shuffle=False,
        **common_args,
    )

    ds["test"] = AudioFolder(
        path=audiofolder_root / "test",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        handle_multiple_captions="keep_first",
        shuffle=False,
        **common_args,
    )

    return ds
