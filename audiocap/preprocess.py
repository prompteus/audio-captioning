from __future__ import annotations

import pandas as pd


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

