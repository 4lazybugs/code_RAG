from .base import Evaluator

class MRREvaluator(Evaluator):
    metric_key = "mrr"

    def __init__(self):
        super().__init__()

    @staticmethod
    def _canon(name: str) -> str:
        return (name or "").strip()

    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs) -> list:
        """
        gen_docs: 샘플 1개의 retrieved 결과 (list[dict])
                 예) [{"rank":1,"filename":"..."}, {"rank":2,"filename":"..."}, ...]
        ref_docs: 샘플 1개의 GT reference_docs
                 예) ["[text]a.md"] 또는 ["[text]a.md","[text]b.md"] 또는 (드물게) "[text]a.md"

        반환: [Reciprocal Rank@K] (샘플 단위)
        - 정답이 top-k에 없으면 0.0
        - 정답이 top-k에서 처음 등장한 위치가 r이면 1.0 / r  (r은 1-based)
        """

        # 1) GT 문서 set 정규화
        if isinstance(ref_docs, str):
            gt_set = {self._canon(ref_docs)}
        else:
            gt_set = {self._canon(x) for x in (ref_docs or [])}

        # 2) retrieved 정렬 + top-k
        retrieved_sorted = sorted(gen_docs or [], key=lambda x: x.get("rank", 10**9))
        topk = retrieved_sorted

        # 3) 첫 정답의 위치(1-based)를 찾아 역수 반환
        #    - rank 필드가 신뢰 가능하면 rank로 계산해도 되지만,
        #      "top-k 내부에서의 위치" 기준으로 계산하는 게 일반적이라 index 기반을 권장.
        for idx, item in enumerate(topk, start=1):
            name = self._canon(item.get("filename", ""))
            if name in gt_set:
                return [1.0 / idx]

        return [0.0]
