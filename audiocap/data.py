from __future__ import annotations

import pathlib
from typing import Any, Literal, Callable

import datasets
import transformers

import audiocap


def load_ds_split(
    folder: pathlib.Path | str,
    load_as_iterable: bool,
    handle_multiple_captions: Literal["flatten", "keep_first"],
    caption_columns: list[str],
    augment: bool,
    shuffle: bool,
    preprocessing_kwargs: dict[str, Any],
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
    shuffle_seed: int | None = None,
    shuffle_buffer_size: int = 100,
    prepare_caption: Callable | None = None,
    take_first_n: int | None = None,
) -> datasets.IterableDataset | datasets.Dataset:

    if isinstance(folder, str):
        folder = pathlib.Path(folder)

    ds = datasets.load_dataset(
        "audiofolder",
        data_files={folder.stem: f"{folder}/*"}
    )

    assert isinstance(ds, (datasets.DatasetDict))

    ds = ds[folder.stem]

    if load_as_iterable:
        # TODO does this shuffle the dataset because of multiprocessing or sth?
        ds = ds.to_iterable_dataset()

    assert isinstance(ds, (datasets.Dataset, datasets.IterableDataset))

    if handle_multiple_captions == "flatten":
        ds = ds.map(
            audiocap.preprocess.flatten_captions,
            batched=True,
            batch_size=10,
            remove_columns=caption_columns,
            fn_kwargs={"caption_columns": caption_columns},
        )
    elif handle_multiple_captions == "keep_first":
        if caption_columns[0] != "caption":
            ds = ds.rename_column(caption_columns[0], "caption")
        ds = ds.remove_columns(caption_columns[1:])
        ds = ds.map(lambda x: {"caption_colname": caption_columns[0]})
    else:
        raise ValueError(f"Unknown handle_multiple_captions value: {handle_multiple_captions}") 
    
    if prepare_caption is not None:
        ds = ds.map(lambda x: {"caption": prepare_caption(x["caption"])})

    if augment:
        # TODO
        pass

    if shuffle:
        if isinstance(ds, datasets.IterableDataset):
            ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
        else:
            ds = ds.shuffle(seed=shuffle_seed)

    preprocessing = audiocap.preprocess.Preprocess(tokenizer, feature_extractor)

    ds = ds.map(
        preprocessing,
        batched=True,
        batch_size=16,
        fn_kwargs=preprocessing_kwargs,
        remove_columns=["audio", "prefix"],
    )

    if take_first_n is not None:
        if isinstance(ds, datasets.IterableDataset):
            ds = ds.take(take_first_n)
        elif isinstance(ds, datasets.Dataset):
            ds = ds.select(range(take_first_n))

    return ds



def load_clotho(
    audiofolder_root: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
) -> dict[str, datasets.IterableDataset | datasets.Dataset]:
    
    if isinstance(audiofolder_root, str):
        audiofolder_root = pathlib.Path(audiofolder_root)

    ds = {}

    preprocessing_kwargs = {"source_ds": "clotho", "task": "caption"}

    ds["dev"] = audiocap.data.load_ds_split(
        folder=audiofolder_root / "development",
        handle_multiple_captions="flatten",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        augment=True,
        shuffle=True,
        load_as_iterable=True,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        preprocessing_kwargs=preprocessing_kwargs,
    )

    ds["val"] = audiocap.data.load_ds_split(
        folder=audiofolder_root / "validation",
        handle_multiple_captions="keep_first",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        augment=False,
        shuffle=False,
        load_as_iterable=True,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        preprocessing_kwargs=preprocessing_kwargs,
    )

    ds["test"] = audiocap.data.load_ds_split(
        folder=audiofolder_root / "evaluation",
        handle_multiple_captions="keep_first",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        augment=False,
        shuffle=False,
        load_as_iterable=True,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        preprocessing_kwargs=preprocessing_kwargs,
    )

    return ds


def load_audioset_small(
    audiofolder_root: pathlib.Path | str,
    audioset_ontology_json: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
) -> dict[str, datasets.IterableDataset | datasets.Dataset]:
    
    if isinstance(audiofolder_root, str):
        audiofolder_root = pathlib.Path(audiofolder_root)

    if isinstance(audioset_ontology_json, str):
        audioset_ontology_json = pathlib.Path(audioset_ontology_json)

    ds = {}

    preprocessing_kwargs = {"source_ds": "audioset", "task": "keywords"}
    
    ontology = audiocap.audioset_tools.AudiosetOntology.from_json_file(audioset_ontology_json)

    ds["dev"] = audiocap.data.load_ds_split(
        folder=audiofolder_root / "train",
        handle_multiple_captions="keep_first",
        caption_columns=["labels"],
        augment=True,
        shuffle=True,
        load_as_iterable=True,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        preprocessing_kwargs=preprocessing_kwargs,
        prepare_caption=ontology.audioset_label_ids_to_str,
    )

    ds["val"] = audiocap.data.load_ds_split(
        folder=audiofolder_root / "valid",
        handle_multiple_captions="keep_first",
        caption_columns=["labels"],
        augment=False,
        shuffle=False,
        load_as_iterable=True,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        preprocessing_kwargs=preprocessing_kwargs,
        prepare_caption=ontology.audioset_label_ids_to_str,
    )

    ds["test"] = audiocap.data.load_ds_split(
        folder=audiofolder_root / "test",
        handle_multiple_captions="keep_first",
        caption_columns=["labels"],
        augment=False,
        shuffle=False,
        load_as_iterable=True,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        preprocessing_kwargs=preprocessing_kwargs,
        prepare_caption=ontology.audioset_label_ids_to_str,
    )

    return ds


def load_audiocaps(
    audiofolder_root: pathlib.Path | str,
    tokenizer: transformers.WhisperTokenizer,
    feature_extractor: transformers.WhisperFeatureExtractor,
) -> dict[str, datasets.IterableDataset | datasets.Dataset]:
    
    if isinstance(audiofolder_root, str):
        audiofolder_root = pathlib.Path(audiofolder_root)
        
    ds = {}

    preprocessing_kwargs = {"source_ds": "audiocaps", "task": "caption"}
    
    ds["dev"] = audiocap.data.load_ds_split(
        folder=audiofolder_root / "train",
        handle_multiple_captions="keep_first",
        caption_columns=["caption"],
        augment=True,
        shuffle=True,
        load_as_iterable=True,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        preprocessing_kwargs=preprocessing_kwargs,
    )

    ds["val"] = audiocap.data.load_ds_split(
        folder=audiofolder_root / "valid",
        handle_multiple_captions="keep_first",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        augment=False,
        shuffle=False,
        load_as_iterable=True,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        preprocessing_kwargs=preprocessing_kwargs,
    )

    ds["test"] = audiocap.data.load_ds_split(
        folder=audiofolder_root / "test",
        handle_multiple_captions="keep_first",
        caption_columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
        augment=False,
        shuffle=False,
        load_as_iterable=True,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        preprocessing_kwargs=preprocessing_kwargs,
    )

    return ds