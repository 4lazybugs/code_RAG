from typing import Dict, Any
from .base import Evaluator, add_metric_key

@add_metric_key("mrr")
class MRREvaluator(Evaluator):

    @staticmethod
    def _canon(name: str) -> str:
        return (name or "").strip()

    def score_once(self, data: Dict[str, Any]) -> float:
        """
        data:
          - gen_docs: retrieved 결과 (list[dict])  예) [{"rank":1,"filename":"..."}, ...]
          - ref_docs: GT 문서 (list[str] or str)   예) ["[text]a.md"] or "[text]a.md"

        반환:
          - top-k에 정답 없으면 0.0
          - top-k에서 처음 등장한 위치가 idx(1-based)면 1.0/idx
        """
        gen_docs = data.get("gen_docs")
        ref_docs = data.get("ref_docs")

        # 1) GT 문서 set 정규화
        if isinstance(ref_docs, str):
            gt_set = {self._canon(ref_docs)}
        else:
            gt_set = {self._canon(x) for x in (ref_docs or [])}

        # 2) retrieved 정렬
        retrieved_sorted = sorted(gen_docs or [], key=lambda x: x.get("rank", 10**9))

        # 3) 첫 정답 위치(1-based) 찾아 역수 반환
        for idx, item in enumerate(retrieved_sorted, start=1):
            name = self._canon(item.get("filename", ""))
            if name in gt_set:
                return 1.0 / idx

        return 0.0
