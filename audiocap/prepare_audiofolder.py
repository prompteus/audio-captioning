import os
import pathlib

import typer
import rich

app = typer.Typer()

@app.command()
def prepare_clotho_audiofolder(
    clotho_path: pathlib.Path = typer.Argument(..., help="Path to the Clotho dataset")
) -> None:
    import pandas as pd

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

    print("Clotho prepared for loading with audiofolder. ")
    print("To avoid accidental changes of the files inside the folder, run the following command:")
    print(f"  chmod u-x '{clotho_path}'")


if __name__ == "__main__":
    app()
