from __future__ import annotations
from .base import Evaluator
import evaluate
import torch

# 필요 시 CUDA 사용:
device = "cuda" if torch.cuda.is_available() else "cpu"

class BleurtEvaluator(Evaluator):

    def __init__(self, bleurt_model_name):
        super().__init__()  # self.CFG 로드
        self._bleurt = evaluate.load("bleurt", checkpoint=bleurt_model_name)
        self.metric_key = "BLEURT"

    def score_once(self, data: dict) -> float:
        ref = self._normalize(self._get_field(data, "reference", "answer")).strip()
        gen = self._normalize(self._get_field(data, "generated", "gen_answer")).strip()

        out = self._bleurt.compute(predictions=[gen], references=[ref])
        return float(out["scores"][0])