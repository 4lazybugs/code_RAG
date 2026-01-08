from typing import Protocol, Dict, Any, Callable
from langchain_core.prompts import ChatPromptTemplate

class QAMode(Protocol):
    prompt: ChatPromptTemplate
    def build_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...

# QA Mode 등록 및 빌드 함수
QA_MODE_REGISTRY: Dict[str, Callable[[], QAMode]] = {}

def register_qa_mode(key: str):
    def deco(cls_or_factory):
        k = key.strip().upper()
        if k in QA_MODE_REGISTRY:
            raise KeyError(f"Duplicate qa_mode key: {k}")
        # 클래스인 경우 factory로 변환
        if isinstance(cls_or_factory, type):
            QA_MODE_REGISTRY[k] = lambda: cls_or_factory()
        else:
            QA_MODE_REGISTRY[k] = cls_or_factory
        return cls_or_factory
    return deco

def build_qa_mode(key: str) -> QAMode:
    k = key.strip().upper()
    try:
        factory = QA_MODE_REGISTRY[k]
    except KeyError:
        raise KeyError(
            f"Unknown qa_mode key: {k}. Available: {list(QA_MODE_REGISTRY.keys())}"
        ) from None
    return factory()
