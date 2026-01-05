from typing import Dict, Any, List
from .base import Evaluator, add_metric_key

@add_metric_key("recall")
class RecallEvaluator(Evaluator):

    @staticmethod
    def _canon(name: str) -> str:
        return (name or "").strip()

    def score_once(self, data: Dict[str, Any]) -> float:
        """
        data:
          - reference: (옵션) reference 텍스트
          - generated: (옵션) generated 텍스트
          - gen_docs: retrieved 결과 (list[dict])  예) [{"rank":1,"filename":"..."}, ...]
          - ref_docs: GT 문서 (list[str] or str)   예) ["[text]a.md"] or "[text]a.md"

        반환: recall@k를 0.0/1.0 float로 반환(샘플 단위)
        """

        gen_docs = data.get("gen_docs")   # list[dict] 예상
        ref_docs = data.get("ref_docs")   # list[str] or str 예상

        # 1) GT 문서 set 정규화
        if isinstance(ref_docs, str):
            gt_set = {self._canon(ref_docs)}
        else:
            gt_set = {self._canon(x) for x in (ref_docs or [])}

        # 2) retrieved 정렬 후 filename 추출
        retrieved_sorted = sorted(gen_docs or [], key=lambda x: x.get("rank", 10**9))
        topk_names = [self._canon(x.get("filename", "")) for x in retrieved_sorted]

        # 3) hit 여부(0/1)
        return 1.0 if any(name in gt_set for name in topk_names) else 0.0
