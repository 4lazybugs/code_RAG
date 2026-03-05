from __future__ import annotations
from .base import Evaluator, get_config
import evaluate

CFG = get_config()

# 필요 시 CUDA 사용:
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

class BleurtEvaluator(Evaluator):
    # 원칙적으로 bleurt-base-128의 치역은 제한이 없으나 논문 보고 기준 -1.5 ~ 1.5
    def __init__(self, model_name: str = "bleurt-base-128"):
        super().__init__()
        # BLEURT 로드 (Hugging Face evaluate)
        self._bleurt = evaluate.load("bleurt", checkpoint=model_name)

    def score_once(self, data: dict) -> float:
        ref = self._normalize(self._get_field(data, "reference", "answer")).strip()
        gen = self._normalize(self._get_field(data, "generated", "gen_answer")).strip()

        out = self._bleurt.compute(predictions=[gen], references=[ref])
        return float(out["scores"][0])