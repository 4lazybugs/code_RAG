# evaluators/rouge1.py
from typing import Dict, Any, List
from kiwipiepy import Kiwi
from collections import Counter
from .base import Evaluator

_kiwi = Kiwi()

def _tokenize_ko(text: str) -> List[str]:
    return [token.form for token in _kiwi.tokenize(text)]


class Rouge1Evaluator(Evaluator):
    """
    ROUGE-1 F1 (한국어 형태소 단위)
    - Kiwi로 형태소 토크나이징
    - unigram overlap 기반 precision/recall -> F1
    """

    def __init__(self):
        super().__init__()

    def score_once(self, data: Dict[str, Any]) -> float:
        ref_text = self._normalize(self._get_field(data, "reference", "answer"))
        gen_text = self._normalize(self._get_field(data, "generated", "gen_answer"))

        ref_tokens = _tokenize_ko(ref_text)
        gen_tokens = _tokenize_ko(gen_text)

        if not ref_tokens and not gen_tokens:
            return 1.0
        if not ref_tokens or not gen_tokens:
            return 0.0

        ref_counts = Counter(ref_tokens)
        gen_counts = Counter(gen_tokens)

        overlap = sum((ref_counts & gen_counts).values())
        prec = overlap / len(gen_tokens)
        rec  = overlap / len(ref_tokens)

        return 0.0 if (prec + rec) == 0 else float(2 * prec * rec / (prec + rec))