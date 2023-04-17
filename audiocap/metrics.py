from __future__ import annotations
from typing import Dict, Any, Union, List
from pathlib import Path
import json

import evaluate
import numpy as np
import pandas as pd
import transformers
import datasets

from .evaluation_tools.coco_caption.pycocotools.coco import COCO
from .evaluation_tools.coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .evaluation_tools.coco_caption.pycocoevalcap.cider.cider import Cider
from .evaluation_tools.coco_caption.pycocoevalcap.spice.spice import Spice

def write_json(data: Union[List[Dict[str, Any]], Dict[str, Any]],
               path: Path) \
        -> None:
    """ Write a dict or a list of dicts into a JSON file
    :param data: Data to write
    :type data: list[dict[str, any]] | dict[str, any]
    :param path: Path to the output file
    :type path: Path
    """
    with path.open("w") as f:
        json.dump(data, f)

def reformat_to_coco(predictions: List[str],
                     ground_truths: List[List[str]],
                     ids: Union[List[int], None] = None) \
        -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ Reformat annotations to the MSCOCO format
    :param predictions: List of predicted captions
    :type predictions: list[str]
    :param ground_truths: List of lists of reference captions
    :type ground_truths: list[list[str]]
    :param ids: List of file IDs. If not given, a running integer\
                is used
    :type ids: list[int] | None
    :return: Predictions and reference captions in the MSCOCO format
    :rtype: list[dict[str, any]]
    """
    # Running number as ids for files if not given
    if ids is None:
        ids = range(len(predictions))

    # Captions need to be in format
    # [{
    #     "audio_id": : int,
    #     "caption"  : str
    # ]},
    # as per the COCO results format.
    pred = []
    ref = {
        'info': {'description': 'Clotho reference captions (2019)'},
        'audio samples': [],
        'licenses': [
            {'id': 1},
            {'id': 2},
            {'id': 3}
        ],
        'type': 'captions',
        'annotations': []
    }
    cap_id = 0
    for audio_id, p, gt in zip(ids, predictions, ground_truths):
        p = p[0] if isinstance(p, list) else p
        pred.append({
            'audio_id': audio_id,
            'caption': p
        })

        ref['audio samples'].append({
            'id': audio_id
        })

        for cap in gt:
            ref['annotations'].append({
                'audio_id': audio_id,
                'id': cap_id,
                'caption': cap
            })
            cap_id += 1

    return pred, ref

class SpiceMetric(evaluate.Metric):
    
    def _info(self):
        return evaluate.MetricInfo(
            description="SPICE Metric",
            citation="https://arxiv.org/abs/1607.08822",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/tylin/coco-caption"],
            reference_urls=["https://arxiv.org/abs/1607.08822"],
        )
    
    def _compute(self, predictions, references, tokens=None):
        res, gts = tokens
        _spice = Spice()
        (average_score, scores) = _spice.compute_score(gts, res)
        return {"average_score": average_score, "scores": scores}

class CiderMetric(evaluate.Metric):
    
    def __init__(self, n=4, sigma=6.0):
        super().__init__()
        self.cider = Cider(n=n, sigma=sigma)
        self.n = n
        self.sigma = sigma
    
    def _info(self):
        return evaluate.MetricInfo(
            description="CIDEr Metric",
            citation="http://arxiv.org/abs/1411.5726",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/tylin/coco-caption"],
            reference_urls=["http://arxiv.org/abs/1411.5726"],
        )
    
    def _compute(self, predictions, references, tokens=None):
        res, gts = tokens
        (score, scores) = self.cider.compute_score(gts, res)
        return {"score": score, "scores": scores}

class CocoTokenizer:
    def __init__(self, preds_str, trues_str):
        self.evalAudios = []
        self.eval = {}
        self.audioToEval = {}
        
        pred, ref = reformat_to_coco(preds_str, trues_str)
        
        tmp_dir = Path('tmp')

        if not tmp_dir.is_dir():
            tmp_dir.mkdir()

        self.ref_file = tmp_dir.joinpath('ref.json')
        self.pred_file = tmp_dir.joinpath('pred.json')

        write_json(ref, self.ref_file)
        write_json(pred, self.pred_file)

        self.coco = COCO(str(self.ref_file))
        self.cocoRes = self.coco.loadRes(str(self.pred_file))

        self.params = {'audio_id': self.coco.getAudioIds()}
        
    def __del__(self):
        # Delete temporary files
        self.ref_file.unlink()
        self.pred_file.unlink()

    def tokenize(self):
        audioIds = self.params['audio_id']
        # audioIds = self.coco.getAudioIds()
        gts = {}
        res = {}
        for audioId in audioIds:
            gts[audioId] = self.coco.audioToAnns[audioId]
            res[audioId] = self.cocoRes.audioToAnns[audioId]

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        return res, gts

    def setEval(self, score, method):
        self.eval[method] = score

    def setAudioToEvalAudios(self, scores, audioIds, method):
        for audioId, score in zip(audioIds, scores):
            if not audioId in self.audioToEval:
                self.audioToEval[audioId] = {}
                self.audioToEval[audioId]["audio_id"] = audioId
            self.audioToEval[audioId][method] = score

    def setEvalAudios(self):
        self.evalAudios = [eval for audioId, eval in self.audioToEval.items()]


def keyword_metrics_single(*, y_pred: str, y_true: str):
    y_pred = set(label.strip().strip('"').strip('') for label in y_pred.split(",")) - {""}
    y_true = set(label.strip().strip('"').strip('') for label in y_true.split(",")) - {""}
    
    intersection = y_true & y_pred
    union = y_true | y_pred 
    
    if len(y_pred) == 0:
        precision = 0.0
    else:
        precision = len(intersection) / len(y_pred)

    if len(y_true) == 0:
        recall = 0.0
    else:
        recall = len(intersection) / len(y_true)

    if len(union) == 0:
        jaccard = 0.0
    else:
        jaccard = len(intersection) / len(union)
    
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.0

    return {
        "keywords_num_generated": len(y_pred),
        "keywords_num_true": len(y_true),
        "keywords_precision": precision,
        "keywords_recall": recall,
        "keywords_f1": f1,
        "keywords_jaccard": jaccard,
    }

def keyword_metrics_batch(*, y_pred: list[str], y_true: list[str]):
    """
    >>> keyword_metrics_batch(
    ...     y_pred = ["hello, darkness, my old friend", "a, b, c, d, e"],
    ...     y_true = ["hello, world,", "a, b, c, d"],
    ... )
    {'keywords_num_generated': 4.0,
     'keywords_num_true': 3.0,
     'keywords_precision': 0.5666666666666667,
     'keywords_recall': 0.75,
     'keywords_f1': 0.6444444444444445,
     'keywords_jaccard': 0.525}
"""

    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length")
    
    batch_size = len(y_pred)
    metrics = pd.DataFrame([
        keyword_metrics_single(y_pred=y_pred[i], y_true=y_true[i])
        for i in range(batch_size)
    ])
    
    return metrics.mean().to_dict()


class CaptioningMetrics:

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        expected_captions: list[str],
        expected_alternatives: list[list[str]],
        ds_captions_size: int,
    ) -> None:
        self.sacrebleu = evaluate.load("sacrebleu")
        self.meteor = evaluate.load("meteor")
        self.spice = SpiceMetric()
        self.cider = CiderMetric()
        self.tokenizer = tokenizer
        self.expected_captions = expected_captions
        self.expected_alternatives = expected_alternatives
        self.ds_captions_size = ds_captions_size
        
    def __call__(self, eval_preds: transformers.EvalPrediction) -> dict[str, float]:
        preds = eval_preds.predictions
        trues = eval_preds.label_ids

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        trues = np.where(trues != -100, trues, self.tokenizer.pad_token_id)

        preds_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        trues_str = self.tokenizer.batch_decode(trues, skip_special_tokens=True)

        # split on captions and keywords
        preds_str_captions = preds_str[:self.ds_captions_size]
        preds_str_keywords = preds_str[self.ds_captions_size:]

        # check if trues_str are expected labels
        # TODO bacha na nas fluff -> expected_captions jsou bez fluffu, jak jsou na tom trues_str?
        assert trues_str[:self.ds_captions_size] == self.expected_captions, f"Expected labels: predicitons are different than expected."         

        sacrebleu_score = self.sacrebleu.compute(predictions=preds_str_captions, references=self.expected_alternatives)
        meteor_score = self.meteor.compute(predictions=preds_str_captions, references=self.expected_alternatives)

        # coco metrics
        tokenizer = CocoTokenizer(preds_str_captions, self.expected_alternatives)
        tokens = tokenizer.tokenize()
        spice_score = self.spice.compute(predictions=preds_str_captions, references=self.expected_alternatives, tokens=tokens)
        cider_score = self.cider.compute(predictions=preds_str_captions, references=self.expected_alternatives, tokens=tokens)
        spider_score = 0.5 * (spice_score['average_score'] + cider_score['score'])

        pred_num_tokens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds[:self.ds_captions_size]]

        logged_dict = {
            "sacrebleu_captions": sacrebleu_score['score'],
            "meteor_captions": meteor_score['meteor'],
            "spice_captions": spice_score['average_score'],
            "cider_captions": cider_score['score'],
            "spider_captions": spider_score,
            "num_tokens_captions": float(np.mean(pred_num_tokens)),
        }

        # if keywords are present, compute the metrics for them
        if len(preds_str_keywords) > 0:
            trues_str_keywords = trues_str[self.ds_captions_size:]
            keywords_metrics = keyword_metrics_batch(
                y_pred=preds_str_keywords,
                y_true=trues_str_keywords
            )
            logged_dict.update(keywords_metrics)

            sacrebleu_score_keywords = self.sacrebleu.compute(predictions=preds_str_keywords, references=trues_str_keywords)
            meteor_score_keywords = self.meteor.compute(predictions=preds_str_keywords, references=trues_str_keywords)

            # coco metrics
            tokenizer = CocoTokenizer(preds_str_keywords, trues_str_keywords)
            tokens = tokenizer.tokenize()
            spice_score_keywords = self.spice.compute(predictions=preds_str_keywords, references=trues_str_keywords, tokens=tokens)
            cider_score_keywords = self.cider.compute(predictions=preds_str_keywords, references=trues_str_keywords, tokens=tokens)
            spider_score_keywords = 0.5 * (spice_score_keywords['average_score'] + cider_score_keywords['score'])

            pred_num_tokens_keywords = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds[:self.ds_captions_size]]

            logged_dict.update({
                "sacrebleu_keywords": sacrebleu_score_keywords['score'],
                "meteor_keywords": meteor_score_keywords['meteor'],
                "spice_keywords": spice_score_keywords['average_score'],
                "cider_keywords": cider_score_keywords['score'],
                "spider_keywords": spider_score_keywords,
                "num_tokens_keywords": float(np.mean(pred_num_tokens_keywords)),
            })

        return logged_dict
