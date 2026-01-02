from abc import ABC, abstractmethod
from load_data import FetchData

class Evaluator(ABC):
    """
    오프라인 평가용 베이스:
    - Runner/Repository가 references/generate/gen_docs/ref_docs를 이미 제공한다는 가정
    - 여기서는 compute_scores()만 강제한다
    """

    @staticmethod # 객체(self)도 클래스(cls)도 필요 없는 함수
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

    @abstractmethod # 상속받은 자식 클래스에서 반드시 구현해야 함
    def compute_scores(self, data:FetchData) -> list:
        pass
