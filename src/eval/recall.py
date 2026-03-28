from typing import Dict, Any, List
from .base import Evaluator

class RecallEvaluator(Evaluator):

    def __init__(self):
        self.metric_key = "Recall"

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

        ref_doc_name = data.get("ref_docs")
        ret_docs = data.get("gen_docs")

        # 1) GT 문서 set 정규화
        if isinstance(ref_doc_name, str):
            gt_set = {self._canon(ref_doc_name)}
        else:
            gt_set = {self._canon(x) for x in (ref_doc_name or [])}

        # 2) retrieved 정렬 후 filename 추출
        retrieved_sorted = sorted(ret_docs or [], key=lambda x: x.get("rank", 10**9))
        topk_names = [self._canon(x.get("filename") or x.get("rel_path", "")) for x in retrieved_sorted]

        # 3) hit 여부(0/1)
        return 1.0 if any(name in gt_set for name in topk_names) else 0.0
