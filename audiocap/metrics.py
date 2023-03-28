from __future__ import annotations

import evaluate
import numpy as np
import transformers
import datasets
import json
from pathlib import Path

from .coco_caption.pycocotools.coco import COCO
from .coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .coco_caption.pycocoevalcap.cider.cider import Cider
from .coco_caption.pycocoevalcap.spice.spice import Spice

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

class CaptioningMetrics:

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> None:
        self.sacrebleu = evaluate.load("sacrebleu")
        self.meteor = evaluate.load("meteor")
        self.exact_match = evaluate.load("exact_match")
        self.spice = SpiceMetric()
        self.cider = CiderMetric()
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
        meteor_score = self.meteor.compute(predictions=preds_str, references=trues_str)
        exact_match_score = self.exact_match.compute(predictions=preds_str, references=trues_str)

        # coco metrics
        tokenizer = CocoTokenizer(preds_str, trues_str)
        tokens = tokenizer.tokenize()
        spice_score = self.spice.compute(predictions=preds_str, references=trues_str, tokens=tokens)
        cider_score = self.cider.compute(predictions=preds_str, references=trues_str, tokens=tokens)

        pred_num_tokens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]

        logged_dict = {
            "sacrebleu": sacrebleu_score['score'],
            "exact_match": exact_match_score["exact_match"],
            "meteor": meteor_score['meteor'],
            "spice": spice_score['average_score'],
            "cider": cider_score['score'],
            "num_tokens": float(np.mean(pred_num_tokens)),
        }

        return logged_dict
