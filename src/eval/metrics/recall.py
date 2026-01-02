from .base import Evaluator

class RecallEvaluator(Evaluator):
    metric_key = "recall"

    def __init__(self):
        super().__init__()

    @staticmethod
    def _canon(name: str) -> str:
        """
        문서명 정규화.
        현재 예시는 GT와 retrieved 모두 '[text]xxx.md' 형태로 동일하므로
        기본은 strip만 적용. 필요하면 여기서 lower()나 prefix 제거 등을 추가.
        """
        return (name or "").strip()

    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs) -> list:
        """
        gen_docs: 샘플 1개의 retrieved 결과 (list[dict])
                 예) [{"rank":1,"filename":"..."}, {"rank":2,"filename":"..."}, ...]
        ref_docs: 샘플 1개의 GT reference_docs
                 예) ["[text]a.md"] 또는 ["[text]a.md","[text]b.md"] 또는 (드물게) "[text]a.md"

        반환: [Recall@K] (샘플 단위이므로 길이 1 리스트)
        """

        # 1) GT 문서들을 set으로 정규화
        if isinstance(ref_docs, str):
            gt_set = {self._canon(ref_docs)}
        else:
            gt_set = {self._canon(x) for x in (ref_docs or [])}

        # 2) retrieved를 rank 기준 정렬 후 top-k filename 추출
        retrieved_sorted = sorted(gen_docs or [], key=lambda x: x.get("rank", 10**9))
        topk_names = [self._canon(x.get("filename", "")) for x in retrieved_sorted]

        # 3) hit 여부로 recall@k 계산 (0/1)
        score = 1.0 if any(name in gt_set for name in topk_names) else 0.0
        return [score]
