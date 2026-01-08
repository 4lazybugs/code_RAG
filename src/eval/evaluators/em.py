import re
from typing import Any, Dict
from .base import Evaluator, add_metric_key, get_config

_LEADING_CHOICE_RE = re.compile(r"^\s*(\d+)\s*(?:[).]|:)?")  # "1)", "1.", "1 :", "1" 등


def _leading_number(text: str):
    if not text:
        return None
    m = _LEADING_CHOICE_RE.match(text)
    return m.group(1) if m else None


@add_metric_key("em")
class ExactMatchEvaluator(Evaluator):

    def __init__(self, remove_spaces: bool = True, mcq: bool = True):
        super().__init__()
        self.remove_spaces = remove_spaces
        self.mcq = mcq  # True면 "선두 숫자 있으면 객관식 모드" 자동 적용

    def _prep(self, x: Any) -> str:
        s = self._normalize(x)
        return s.replace(" ", "") if self.remove_spaces else s

    def score_once(self, data: Dict[str, Any]) -> float:
        ref_norm = self._prep(self._get_field(data, "reference", "answer"))
        gen_norm = self._prep(self._get_field(data, "generated", "gen_answer"))

        # 둘 다 비어있으면 정답 처리
        if not ref_norm and not gen_norm:
            return 1.0

        # 1) 객관식 모드(선두 숫자) 자동 판정
        if self.mcq:
            ref_choice = _leading_number(ref_norm)
            gen_choice = _leading_number(gen_norm)

            # 둘 중 하나라도 선두 숫자가 잡히면 객관식으로 판단
            if ref_choice is not None or gen_choice is not None:
                # 숫자 추출 실패한 쪽은 오답
                if ref_choice is None or gen_choice is None:
                    return 0.0
                return 1.0 if ref_choice == gen_choice else 0.0

        # 2) 그 외: 전체 문자열 exact match
        return 1.0 if ref_norm == gen_norm else 0.0
