# imports
import pandas as pd
from audiocap.metrics import CiderMetric, SpiceMetric, CocoTokenizer
import evaluate
import pathlib
import typer
import json

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    predictions_path: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="Path to the jsonl file with caption predictions"),
    labels_path: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="file with the ground truth captions"),
) -> None:
    
    print(">>>>>> COMPUTING METRICS <<<<<<")
    # load predictions and gt
    df_preds = pd.read_csv(predictions_path, sep=",")

    if labels_path.suffix == ".csv":
        df_labels = pd.read_csv(labels_path)
    elif labels_path.suffix == ".jsonl":
        df_labels = pd.read_json(labels_path, lines=True)
    else:
        raise ValueError(f"labels_path must be a csv or jsonl file, got {labels_path.suffix}")
    
    # join predictions and labels
    df = df_preds.merge(df_labels, on="file_name")

    # join multiple gts to a list and drop original cols
    df["all_labels"] = df.apply(lambda x: [x.caption_1, x.caption_2, x.caption_3, x.caption_4, x.caption_5], axis=1)
    df = df.drop(columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"])

    # init metrics
    sacrebleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    spice = SpiceMetric()
    cider = CiderMetric()

    preds_str = df["caption_predicted"].tolist()
    references = df["all_labels"].tolist()
    # compute metrics
    sacrebleu_score = sacrebleu.compute(predictions=preds_str, references=references)
    meteor_score = meteor.compute(predictions=preds_str, references=references)

    # coco metrics
    tokenizer = CocoTokenizer(preds_str, references)
    tokens = tokenizer.tokenize()
    spice_score = spice.compute(predictions=preds_str, references=references, tokens=tokens)
    cider_score = cider.compute(predictions=preds_str, references=references, tokens=tokens)
    spider_score = 0.5 * (spice_score['average_score'] + cider_score['score'])

    output_dict = {"metric_computation": {
                        "predictions file": str(predictions_path),
                        "ground truth file": str(labels_path),
                        "computed metrics": {
                            "sacrebleu": sacrebleu_score["score"],
                            "meteor": meteor_score["meteor"],
                            "spice": spice_score['average_score'],
                            "cider": cider_score['score'],
                            "spider": spider_score
                        }
                    }
                  }
    print(json.dumps(output_dict, indent=4, sort_keys=False))

    log_file = predictions_path.parent / (predictions_path.stem + '_log.json')
    with open(log_file, "r+") as f:
        try:
            log_dict = json.load(f)
            log_dict["metric_computation"] = output_dict["metric_computation"]
        except json.decoder.JSONDecodeError:
            print("No log_file => Creating new log file")
            log_dict = output_dict

    with open(log_file, "w") as f:
        json.dump(log_dict, open(log_file, "w"), indent=2, ensure_ascii=False)
    with open(log_file.parent / "all_spiders", "a") as f:
        f.write(str(predictions_path.stem) 
                + ":" 
                + " " * (80 - len(str(predictions_path.stem))) 
                + f"{spider_score:.4f}"
                + "\n")

if __name__ == "__main__":
    app()
