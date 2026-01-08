from langchain_openai.chat_models import ChatOpenAI
from typing import Any, Callable, Dict

# 모델 등록 및 빌드 함수 (RAG 타입 레지스트리)
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_model(key: str):
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        k = key.strip()
        if k in MODEL_REGISTRY:
            raise KeyError(f"Duplicate model key: {k}")
        MODEL_REGISTRY[k] = factory
        return factory
    return deco

def build_model(key: str, **kwargs) -> Any: # key는 RAG Type을 말함 ex) naive_rag, naive_llm 등등
    k = key.strip()
    try:
        factory = MODEL_REGISTRY[k]
    except KeyError:
        raise KeyError(
            f"Unknown model key: {k}. Available: {list(MODEL_REGISTRY.keys())}"
        ) from None
    return factory(**kwargs)


class BaseModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = ChatOpenAI(
            base_url=getattr(cfg, "base_url", "http://127.0.0.1:8000/v1"),
            api_key=getattr(cfg, "api_key", "EMPTY"),
            model=cfg.model_name,
            temperature=getattr(cfg, "temperature", 0.0),
        )
    
    @staticmethod
    def to_plain_text(x: Any) -> str:
        if hasattr(x, "content"):
            return str(x.content or "")
        return str(x)