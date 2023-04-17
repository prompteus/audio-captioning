from __future__ import annotations

import pathlib

import pandas as pd
import torch
import torchaudio
import transformers
import librosa


# TODO maybe TypedDict hints
def flatten_captions(orig_batch: dict) -> dict:
    batch_df = pd.DataFrame(dict(orig_batch))
    batch_df = batch_df.melt(
        id_vars="audio",
        var_name="caption_idx",
        value_vars=[f"caption_{i}" for i in [1, 2, 3, 4, 5]],
        value_name="caption",
    )
    batch: dict = batch_df.to_dict(orient="list")
    assert set(["audio", "caption_idx", "caption"]) == set(batch.keys())
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
    def __call__(self, orig_batch: dict, source_ds: str, task: str) -> dict:
        audios = pd.DataFrame(orig_batch.pop("audio")).rename(columns={"array": "audio_array"})
        assert set(["path", "audio_array", "sampling_rate"]) == set(audios.keys())

        # popping due huggingface column handling in map function
        rest = pd.DataFrame({k: orig_batch.pop(k) for k in list(orig_batch.keys())}) 
        assert set(["caption_idx", "caption"]) == set(rest.keys())

        assert orig_batch == {}

        assert len(audios) == len(rest)
        batch = pd.concat([audios, rest], axis="columns")
        assert set(["path", "audio_array", "sampling_rate", "caption_idx", "caption"]) == set(batch.keys())

        batch["prefix"] = source_ds + " > " + task + ": "
        batch["forced_ac_decoder_ids"] = batch["prefix"].apply(lambda x: self.tokenizer("", text_target=x, add_special_tokens=False).labels)
        # batch["caption"] = batch["prefix"] + batch["caption"]
        batch["filename"] = batch["path"].apply(lambda path: pathlib.Path(path).name)

        # check suspicious shape and convert to mono
        test_sample = batch["audio_array"].iloc[0]
        if len(test_sample.shape) > 1 and test_sample.shape[0] > 10:
            print(f"WARNING: audio might have bad shape (switched channels and time), \
                  shape: {test_sample.shape}, \
                  filename: {batch['filename'].iloc[0]} \
                  ds_source: {source_ds}")  
        batch["audio_array"] = batch["audio_array"].apply(librosa.to_mono)

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

        prefix_ids = self.tokenizer("", text_target="", add_special_tokens=True).labels[:-1]  # prefix fluff without eos

        batch["labels"] = self.tokenizer("", text_target=batch["caption"].tolist(), add_special_tokens=False).labels
        batch["labels"] = [prefix_ids + fdi + label + [self.tokenizer.eos_token_id] for fdi, label in zip(batch["forced_ac_decoder_ids"], batch["labels"])]
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
        batch_forced_ac_decoder_ids = [x["forced_ac_decoder_ids"] for x in orig_batch]

        batch = self.feature_extractor.pad(batch_features, return_tensors="pt")
        batch_labels = self.tokenizer.pad(batch_labels, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = batch_labels["input_ids"].masked_fill(batch_labels.attention_mask != 1, -100)

        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["forced_ac_decoder_ids"] = torch.tensor(batch_forced_ac_decoder_ids)
        batch["labels"] = labels
        return batch
