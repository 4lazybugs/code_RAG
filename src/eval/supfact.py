from typing import Any, Dict
from .base import Evaluator, add_metric_key


@add_metric_key("supfact")
class SupportingEMEvaluator(Evaluator):
    """
    supporting_fact_gen이 supporting_fact_ref 또는 supporting_fact_comp에
    포함되는지(부분문자열) 체크.

    점수:
      - 포함되면 1.0
      - 아니면 0.0
    """

    def __init__(self, remove_spaces: bool = True, case_insensitive: bool = True):
        super().__init__()
        self.remove_spaces = remove_spaces
        self.case_insensitive = case_insensitive

    def _prep(self, x: Any) -> str:
        s = self._normalize(x)
        if self.case_insensitive:
            s = s.lower()
        if self.remove_spaces:
            s = s.replace(" ", "")
        return s

    def score_once(self, data: Dict[str, Any]) -> float:
        """
        data에서 직접 꺼내는 것을 전제로 함.
        필요한 키:
          - supporting_fact_gen
          - supporting_fact_ref
          - supporting_fact_comp
        """
        sf_gen = self._prep(data.get("supporting_fact_gen", ""))
        sf_ref = self._prep(data.get("supporting_fact_ref", ""))
        sf_comp = self._prep(data.get("supporting_fact_comp", ""))

        # gen이 비어있으면 실패(정책은 필요 시 변경 가능)
        if not sf_gen:
            return 0.0

        ok = (sf_gen in sf_ref) or (sf_gen in sf_comp)
        return 1.0 if ok else 0.0
