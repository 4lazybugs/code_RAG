# evaluators/rougel.py
from typing import Dict, Any, List
from .base import Evaluator, add_metric_key


def _lcs_length(a: List[str], b: List[str]) -> int:
    """
    LCS 길이 (토큰 시퀀스 기준)
    메모리 O(min(n,m))로 최적화한 DP
    """
    if not a or not b:
        return 0

    # b를 더 짧게 두면 메모리 절약에 유리
    if len(a) < len(b):
        a, b = b, a

    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[-1]


@add_metric_key("rougel")
class RougeLEvaluator(Evaluator):
    """
    ROUGE-L F1 (토큰 단위)
    - reference/generated를 공백 기준 토큰화
    - LCS 기반 precision/recall -> F1
    - gen_docs/ref_docs는 data dict에 있어도 무시하거나(기본) 추후 확장 가능
    """

    def score_once(self, data: Dict[str, Any]) -> float:
        ref_text = self._normalize(data.get("reference"))
        gen_text = self._normalize(data.get("generated"))

        ref_tokens = [t for t in ref_text.split() if t]
        gen_tokens = [t for t in gen_text.split() if t]

        if not ref_tokens and not gen_tokens:
            return 1.0
        if not gen_tokens:
            return 0.0
        if not ref_tokens:
            return 0.0

        lcs = _lcs_length(ref_tokens, gen_tokens)

        prec = lcs / len(gen_tokens) if gen_tokens else 0.0
        rec = lcs / len(ref_tokens) if ref_tokens else 0.0

        return 0.0 if (prec + rec) == 0 else float(2 * prec * rec / (prec + rec))
