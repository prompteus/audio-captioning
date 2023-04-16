from __future__ import annotations

import pathlib

import pandas as pd
import torch
import torchaudio
import transformers
import librosa


# TODO maybe TypedDict hints
def clotho_flatten_captions(orig_batch: dict) -> dict:
    batch_df = pd.DataFrame(dict(orig_batch))
    batch_df = batch_df.melt(
        id_vars="audio",
        var_name="caption_idx",
        value_vars=[f"caption_{i}" for i in [1, 2, 3, 4, 5]],
        value_name="caption",
    )
    batch_df["source_ds"] = ["clotho"] * len(batch_df)
    batch_df["task"] = ["captions"] * len(batch_df)
    batch: dict = batch_df.to_dict(orient="list")
    assert set(["audio", "caption_idx", "caption", "source_ds", "task"]) == set(batch.keys())
    return batch


def audioset_set_columns(orig_batch: dict) -> dict:
    batch_df = pd.DataFrame(dict(orig_batch))
    batch_df["source_ds"] = ["audioset"] * len(batch_df)
    batch_df["task"] = ["keywords"] * len(batch_df)
    batch_df["caption_idx"] = [0] * len(batch_df)
    batch: dict = batch_df.to_dict(orient="list")
    assert set(["audio", "caption_idx", "caption", "source_ds", "task"]) == set(batch.keys())
    return batch


class Preprocess:

    def __init__(
        self,
        tokenizer: transformers.WhisperTokenizer,
        feature_extractor: transformers.WhisperFeatureExtractor,
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    # TODO maybe TypedDict hints
    def __call__(self, orig_batch: dict) -> dict:
        audios = pd.DataFrame(orig_batch.pop("audio")).rename(columns={"array": "audio_array"})
        assert set(["path", "audio_array", "sampling_rate"]) == set(audios.keys())

        # popping due huggingface column handling in map function
        rest = pd.DataFrame({k: orig_batch.pop(k) for k in list(orig_batch.keys())}) 
        assert set(["caption_idx", "caption", "source_ds", "task"]) == set(rest.keys())

        assert orig_batch == {}

        assert len(audios) == len(rest)
        batch = pd.concat([audios, rest], axis="columns")
        assert set(["path", "audio_array", "sampling_rate", "caption_idx", "caption", "source_ds", "task"]) == set(batch.keys())

        batch["prefix"] = batch["source_ds"] + ">" + batch["task"]
        batch["forced_prefix_ids"] = batch["prefix"].apply(self.tokenizer.encode)
        batch["caption"] = batch["prefix"] + batch["caption"]
        batch["filename"] = batch["path"].apply(lambda path: pathlib.Path(path).name)

        # check suspicious shape and convert to mono
        test_sample = batch["audio_array"].iloc[0]
        if len(test_sample.shape) > 1 and test_sample.shape[0] > 10:
            print(f"WARNING: audio might have bad shape (switched channels and time), \
                  shape: {test_sample.shape}, \
                  filename: {batch['filename'].iloc[0]} \
                  ds_source: {batch['source_ds'].iloc[0]}")  
        batch["audio_array"] = batch["audio_array"].apply(librosa.to_mono)

        # TODO ensure MONO audio
        assert (batch["audio_array"].apply(lambda x: x.ndim) == 1).all()

        def resample(row) -> torch.Tensor:
            return torchaudio.functional.resample(
                torch.tensor(row["audio_array"]),
                row["sampling_rate"],
                self.feature_extractor.sampling_rate
            ).cpu().numpy()

        batch["audio_array"] = batch.apply(resample, axis=1)
        batch["sampling_rate"] = self.feature_extractor.sampling_rate

        assert batch["sampling_rate"].nunique() == 1
        features = self.feature_extractor(
            batch["audio_array"].tolist(),
            sampling_rate=batch["sampling_rate"][0],
            return_tensors="np",
        )
        batch["input_features"] = list(features["input_features"])

        assert features["input_features"].shape[0] == len(batch)
        assert batch["input_features"][0].shape == features["input_features"][0].shape

        batch["labels"] = self.tokenizer("", text_target=batch["caption"].tolist()).labels

        return batch.to_dict(orient="list")


class DataCollatorAudioSeq2SeqWithPadding:

    def __init__(
        self,
        tokenizer: transformers.WhisperTokenizer,
        feature_extractor: transformers.WhisperFeatureExtractor,
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    # TODO maybe TypedDict hints
    def __call__(
        self,
        orig_batch: list[dict],
    ) -> dict:
        
        batch_features = [{"input_features": x["input_features"]} for x in orig_batch]
        batch_labels = [{"input_ids": x["labels"]} for x in orig_batch]
        batch_forced_prefix_ids = [x["forced_prefix_ids"] for x in orig_batch]

        batch = self.feature_extractor.pad(batch_features, return_tensors="pt")
        batch_labels = self.tokenizer.pad(batch_labels, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = batch_labels["input_ids"].masked_fill(batch_labels.attention_mask != 1, -100)

        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["forced_prefix_ids"] = batch_forced_prefix_ids
        batch["labels"] = labels
        return batch
