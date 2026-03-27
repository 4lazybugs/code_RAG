from abc import ABC, abstractmethod
from typing import Any
from tqdm.auto import tqdm

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
        if isinstance(text, list):
            # content 배열을 문자열로 변환
            text = "\n".join(str(item) for item in text if item)
        return str(text).strip()
    
    @staticmethod
    def _get_field(data: dict, *keys: str) -> Any:
        for k in keys:
            if k in data:
                return data[k]
        return None

    @abstractmethod # 상속받은 자식 클래스에서 반드시 구현해야 함
    def score_once(self, data: dict) -> float:
        """
        단일 샘플 평가
        """
        pass

    def score_all(self, batch_data, *, show_progress=True, desc=None):
        it = batch_data
        if show_progress:
            it = tqdm(
                batch_data,
                desc=getattr(self, "metric_key", "scoring"),
                bar_format="{desc:<10} | {bar:40} | {n}/{total} | {elapsed}s",
                ascii=True,
                colour="cyan",
                leave=True,   # 줄 남김
            )
        return [self.score_once(x) for x in it]