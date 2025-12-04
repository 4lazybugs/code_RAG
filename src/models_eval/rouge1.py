from .base import BaseEvaluator

def _normalize(text: str) -> str:
    if text is None:
        return ""
    return text.strip()

class Rouge1Evaluator(BaseEvaluator):
    metric_key = 'rouge1'

    def __init__(self, expert, qa_data_path, sample_size=None):
        super().__init__(expert, qa_data_path, sample_size)

    def compute_scores(self, references: list, generated: list) -> list:
        scores = []
        for ref, gen in zip(references, generated):
            ref_norm = _normalize(ref).replace(" ", "")
            gen_norm = _normalize(gen).replace(" ", "")

            if not ref_norm and not gen_norm:
                scores.append(1.0)
                continue

            # char 단위 unigram 카운트
            from collections import Counter
            ref_counts = Counter(ref_norm)
            gen_counts = Counter(gen_norm)

            overlap = sum((ref_counts & gen_counts).values())
            prec = overlap / len(gen_norm) if len(gen_norm) > 0 else 0.0
            rec  = overlap / len(ref_norm) if len(ref_norm) > 0 else 0.0

            if prec + rec == 0:
                f1 = 0.0
            else:
                f1 = 2 * prec * rec / (prec + rec)

            scores.append(float(f1))
        return scores
