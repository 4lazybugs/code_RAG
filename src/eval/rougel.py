# evaluators/rougel.py
from typing import Dict, Any, List

from .base import Evaluator
from .rouge_token import get_tokenizer


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
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


class RougeLEvaluator(Evaluator):
    """
    ROUGE-L F1 (한국어 형태소 단위)
    - Kiwi로 형태소 토크나이징
    - LCS 기반 precision/recall -> F1
    """

    def __init__(self, tokenizer_name: str):
        super().__init__()  
        self._tokenize = get_tokenizer(tokenizer_name)

    def score_once(self, data: Dict[str, Any]) -> float:
        ref_text = self._normalize(self._get_field(data, "reference", "answer"))
        gen_text = self._normalize(self._get_field(data, "generated", "gen_answer"))

        ref_tokens = self._tokenize(ref_text)
        gen_tokens = self._tokenize(gen_text)

        if not ref_tokens and not gen_tokens:
            return 1.0
        if not ref_tokens or not gen_tokens:
            return 0.0

        lcs = _lcs_length(ref_tokens, gen_tokens)

        prec = lcs / len(gen_tokens)
        rec  = lcs / len(ref_tokens)

        return 0.0 if (prec + rec) == 0 else float(2 * prec * rec / (prec + rec))