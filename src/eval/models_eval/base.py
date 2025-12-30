from abc import ABC, abstractmethod
from typing import Any

class BaseEvaluator(ABC):
    """
    오프라인 평가용 베이스:
    - Runner/Repository가 references/generate/gen_docs/ref_docs를 이미 제공한다는 가정
    - 여기서는 compute_scores()만 강제한다
    """

    @staticmethod
    def _normalize(text) -> str:
        """하위 evaluator에서 공통으로 쓸 정규화 유틸(선택)"""
        if text is None:
            return ""
        if hasattr(text, "content"):
            text = text.content
        if isinstance(text, dict):
            for k in ("content", "text", "answer", "output"):
                if k in text:
                    text = text[k]
                    break
        return str(text).strip()

    @abstractmethod
    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs: list) -> list:
        """
        references: 정답 문자열 리스트 (len=N)
        generated : 생성 문자열 리스트 (len=N)
        gen_docs  : retrieved docs (샘플 1개면 list[dict], 배치면 list[list[dict]])
        ref_docs  : GT reference docs (샘플 1개면 list, 배치면 list[list])
        """
        raise NotImplementedError
