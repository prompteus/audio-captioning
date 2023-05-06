
# imports
import pandas as pd
from audiocap.metrics import CiderMetric, SpiceMetric, CocoTokenizer
import evaluate
import pathlib
import typer

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    predictions_path: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="Path to the jsonl file with caption predictions"),
    labels_path: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="file with the ground truth captions"),
    save_stats_path: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="file where to save the metrics stats"),
) -> None:
    
    print(">>>>>> COMPUTING METRICS <<<<<<")
    
    # load predictions and gt
    df_preds = pd.read_json(predictions_path, lines=True)
    df_labels = pd.read_csv(labels_path)

    # cut prefixes
    df_preds["clean_pred"] = df_preds["caption"].apply(lambda x: x.split(": ")[1].split("<|endoftext|>")[0])

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

    preds_str = df["clean_pred"].tolist()
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

    output_str = f"predictions file: {predictions_path}\n" + \
                  "ground truth file: {labels_path}\n" + \
                  "###### COMPUTED METRICS ######\n" + \
                  "sacrebleu: {sacrebleu_score}\n" +  \
                  "meteor: {meteor_score}\n" + \
                  "spice: {spice_score['average_score']}\n" + \
                  "cider: {cider_score['score']}\n" + \
                  "spider: {spider_score}\n\n\n"

    print(output_str)
    with open(save_stats_path, "a") as f:
        f.write(output_str)


if __name__ == "__main__":
    app()
