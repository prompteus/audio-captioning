from __future__ import annotations

import evaluate
import numpy as np
import transformers


class CaptioningMetrics:

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> None:
        self.sacrebleu = evaluate.load("sacrebleu")
        self.rouge = evaluate.load("rouge")
        self.exact_match = evaluate.load("exact_match")
        self.tokenizer = tokenizer

    def __call__(self, eval_preds: transformers.EvalPrediction) -> dict[str, float]:
        preds = eval_preds.predictions
        trues = eval_preds.label_ids

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        trues = np.where(trues != -100, trues, self.tokenizer.pad_token_id)
        
        preds_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        trues_str = self.tokenizer.batch_decode(trues, skip_special_tokens=True)

        sacrebleu_score = self.sacrebleu.compute(predictions=preds_str, references=trues_str)
        rouge_scores = self.rouge.compute(predictions=preds_str, references=trues_str)
        exact_match_score = self.exact_match.compute(predictions=preds_str, references=trues_str)

        pred_num_tokens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]

        logged_dict = {
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "rougeLsum": rouge_scores["rougeLsum"],
            "sacrebleu": sacrebleu_score["score"],
            "exact_match": exact_match_score["exact_match"],
            "num_tokens": float(np.mean(pred_num_tokens)),
        }

        return logged_dict
