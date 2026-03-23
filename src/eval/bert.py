from __future__ import annotations

from typing import Dict, Any

import torch
from bert_score import score as bert_score
from .base import Evaluator, get_config

device = "cuda" if torch.cuda.is_available() else "cpu"

def _clean_for_bert(x: str, max_chars: int = 3000) -> str:
    if x is None:
        return ""
    return str(x).strip()[:max_chars]


class BERTEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def score_once(self, data: Dict[str, Any]) -> float:
        reference = _clean_for_bert(self._normalize(self._get_field(data, "reference", "answer")))
        generated = _clean_for_bert(self._normalize(self._get_field(data, "generated", "gen_answer")))

        # 둘 다 비어 있으면 완벽 일치로 처리(정책은 상황에 맞게 변경 가능)
        if not reference and not generated:
            return 1.0
        if not reference or not generated:
            return 0.0

        with torch.no_grad():
            P, R, F1 = bert_score(
                [generated],                  # ✅ list[str] 형태로 전달 권장
                [reference],
                lang="ko",
                model_type=self.CFG.bert_model_name,
                device=device,
                batch_size=1,
                rescale_with_baseline=False,
            )

        # F1은 텐서 shape [1] -> 스칼라 float로 변환
        return float(F1[0].cpu().item())
