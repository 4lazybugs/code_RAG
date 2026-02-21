from .base import Evaluator, add_metric_key
from collections import Counter

@add_metric_key("rouge1")
class Rouge1Evaluator(Evaluator):

    def score_once(self, data: dict) -> float:
        ref = self._normalize(self._get_field(data, "reference", "answer")).replace(" ", "")
        gen = self._normalize(self._get_field(data, "generated", "gen_answer")).replace(" ", "")

        if not ref and not gen:
            return 1.0

        ref_counts = Counter(ref)
        gen_counts = Counter(gen)

        overlap = sum((ref_counts & gen_counts).values())
        prec = overlap / len(gen) if len(gen) > 0 else 0.0
        rec  = overlap / len(ref) if len(ref) > 0 else 0.0

        return 0.0 if (prec + rec) == 0 else float(2 * prec * rec / (prec + rec))