from .base import BaseEvaluator
import sacrebleu

class BleuEvaluator(BaseEvaluator):
    metric_key = 'bleu'

    def __init__(self, expert, qa_data_path, sample_size=None):
        super().__init__(expert, qa_data_path, sample_size)

    def compute_scores(self, references: list, generated: list) -> list:
        """
        sacrebleu의 sentence_bleu로 문장 단위 BLEU (0~1) 계산
        """
        scores = []
        for ref, gen in zip(references, generated):
            # sacrebleu의 sentence_bleu는 score가 0~100 범위
            s = sacrebleu.sentence_bleu(gen, [ref]).score / 100.0
            scores.append(float(s))
        return scores
