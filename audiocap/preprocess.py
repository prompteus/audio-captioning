from __future__ import annotations

import pathlib

import pandas as pd
import torch
import torchaudio
import transformers


# TODO maybe TypedDict hints
def clotho_flatten_captions(orig_batch: dict) -> dict:
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
    def __call__(self, orig_batch: dict) -> dict:
        audios = pd.DataFrame(orig_batch.pop("audio")).rename(columns={"array": "audio_array"})
        assert set(["path", "audio_array", "sampling_rate"]) == set(audios.keys())

        rest = pd.DataFrame({k: orig_batch.pop(k) for k in list(orig_batch.keys())})
        assert set(["caption_idx", "caption"]) == set(rest.keys())

        assert orig_batch == {}

        assert len(audios) == len(rest)
        batch = pd.concat([audios, rest], axis="columns")
        assert set(["path", "audio_array", "sampling_rate", "caption_idx", "caption"]) == set(batch.keys())

        batch["filename"] = batch["path"].apply(lambda path: pathlib.Path(path).name)

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
