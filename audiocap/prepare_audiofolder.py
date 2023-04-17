import os
import pathlib
import shutil

import joblib
import typer
import rich
import pandas as pd

from tqdm.auto import tqdm

app = typer.Typer()


def print_suggestion(path: pathlib.Path) -> None:
    print(f"{path} is prepared for loading with audiofolder. ")
    print("To avoid accidental changes of the files inside the folder, run the following command:")
    print(f"  chmod u-x '{path}'")


@app.command()
def prepare_clotho_audiofolder(
    clotho_path: pathlib.Path = typer.Argument(..., help="Path to the Clotho dataset")
) -> None:

    expected_paths = [
        clotho_path / "development/",
        clotho_path / "evaluation/",
        clotho_path / "validation/",
        clotho_path / "test/",
        clotho_path / "clotho_captions_development.csv",
        clotho_path / "clotho_captions_evaluation.csv",
        clotho_path / "clotho_captions_validation.csv",
        clotho_path / "clotho_metadata_development.csv",
        clotho_path / "clotho_metadata_evaluation.csv",
        clotho_path / "clotho_metadata_validation.csv",
        clotho_path / "clotho_metadata_test.csv",
    ]
    
    for path in expected_paths:
        if not path.exists():
            print("your folder structure should contain: ")
            rich.print(expected_paths)
            print("but it does not contain: " + str(path))
            raise FileNotFoundError(path)

    for split in ["development", "evaluation", "validation"]:
        df_captions = pd.read_csv(clotho_path / f"clotho_captions_{split}.csv", engine="python")
        df = df_captions
        df.to_json(clotho_path / split / "metadata.jsonl", orient="records", force_ascii=False, lines=True)

    print_suggestion(clotho_path)


@app.command()
def prepare_audioset_small_audiofolder(
    audioset_small_path: pathlib.Path = typer.Option(..., help="Path to the AudioSet small dataset"),
    audioset_full_path: pathlib.Path = typer.Option(..., help="Path to the AudioSet full dataset"),
    audio_format: str = typer.Option(..., help="Extension of the audio files (mp3, wav, ...)"),
) -> None:

    expected_paths = [
        audioset_small_path / "annotations/train.jsonl",
        audioset_small_path / "annotations/valid.jsonl",
        audioset_small_path / "annotations/test.jsonl",
    ]
    
    for path in expected_paths:
        if not path.exists():
            print("your folder structure should contain: ")
            rich.print(expected_paths)
            print("but it does not contain: " + str(path))
            raise FileNotFoundError(path)

    os.makedirs(audioset_small_path / "audiofolder", exist_ok=True)
    for split in ["train", "valid", "test"]: 
        os.makedirs(audioset_small_path / f"audiofolder/{split}", exist_ok=True)
        df = pd.read_json(audioset_small_path / f"annotations/{split}.jsonl", lines=True)
        df.insert(loc=0, column="file_name", value = df["youtube_id"] + ("." + audio_format))
        df.drop(columns=["youtube_id", "start_seconds", "end_seconds"], inplace=True)

        pool = joblib.Parallel(n_jobs=-1)
        queue = (
            (
                audioset_full_path / "audios" / row.orig_split / row.file_name,
                audioset_small_path / "audiofolder" / split / row.file_name
            )
            for row in df[["file_name", "orig_split"]].itertuples(index=False)
            if not (audioset_small_path / "audiofolder" / split / row.file_name).exists()
        )

        tasks = (joblib.delayed(shutil.copy)(source_path, target_path) for source_path, target_path in tqdm(queue, total=len(df)))
        pool(tasks)

        df.drop(columns=["orig_split"], inplace=True)
        df.to_json(audioset_small_path / f"audiofolder/{split}/metadata.jsonl", orient="records", force_ascii=False, lines=True)

    print_suggestion(audioset_small_path)


@app.command()
def prepare_audiocaps_audiofolder(
    audiocaps_path: pathlib.Path = typer.Option(..., help="Path to the AudioCaps dataset"),
    audioset_path: pathlib.Path = typer.Option(..., help="Path to the AudioSet full dataset"),
    audio_format: str = typer.Option(..., help="Extension of the audio files (mp3, wav, ...)"),
) -> None:
    expected_paths = [
        audiocaps_path / "csvs/train.csv",
        audiocaps_path / "csvs/valid.csv",
        audiocaps_path / "csvs/test.csv",
    ]
    
    for path in expected_paths:
        if not path.exists():
            print("your folder structure should contain: ")
            rich.print(expected_paths)
            print("but it does not contain: " + str(path))
            raise FileNotFoundError(path)

    os.makedirs(audiocaps_path / "audiofolder", exist_ok=True)
    for split in ["train", "valid", "test"]: 
        os.makedirs(audiocaps_path / f"audiofolder/{split}", exist_ok=True)
        df = pd.read_csv(audiocaps_path / f"csvs/{split}.csv")
        df = (
            df
            .drop(columns=["audiocap_id", "start_time"])
            .groupby("youtube_id")
            .agg(list)["caption"]
            .apply(pd.Series)
            .rename(columns=lambda x: "caption_" + str(x+1))
            .reset_index()
        )
        if split == "train":
            df.rename(columns={"caption_1": "caption"}, inplace=True)

        df.insert(loc=0, column="file_name", value = df["youtube_id"] + ("." + audio_format))
        df.drop(columns=["youtube_id"], inplace=True)
    
        df["src"] = df["file_name"].apply(lambda x: audioset_path / "audios/train_unbalanced" / x)
        df = df[df["src"].apply(pathlib.Path.exists)]
        df["tgt"] = df["file_name"].apply(lambda x: audiocaps_path / f"audiofolder/{split}" / x)

        pool = joblib.Parallel(n_jobs=-1)
        queue = [(src, tgt) for (src, tgt) in zip(df["src"], df["tgt"]) if not tgt.exists()]
        tasks = (joblib.delayed(shutil.copy)(source_path, target_path) for source_path, target_path in tqdm(queue))
        pool(tasks)

        df.drop(columns=["src", "tgt"], inplace=True)
        df.to_json(audiocaps_path / f"audiofolder/{split}/metadata.jsonl", orient="records", force_ascii=False, lines=True)

    print_suggestion(audiocaps_path)
    

if __name__ == "__main__":
    app()
