import re
from typing import Any, Dict
from .base import Evaluator

class EMEvaluator(Evaluator):

    def __init__(self, remove_spaces: bool = True):
        super().__init__()
        self.remove_spaces = remove_spaces

    def _prep(self, x: Any) -> str:
        s = self._normalize(x)
        return s.replace(" ", "") if self.remove_spaces else s

    def score_once(self, data: Dict[str, Any]) -> float:
        ref_norm = self._prep(self._get_field(data, "reference", "answer"))
        gen_norm = self._prep(self._get_field(data, "generated", "gen_answer"))

        # 둘 다 비어있으면 정답 처리
        if not ref_norm and not gen_norm:
            return 1.0

        # 순수 exact match
        return 1.0 if ref_norm == gen_norm else 0.0