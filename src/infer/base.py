from langchain_openai.chat_models import ChatOpenAI
from typing import Any, Callable, Dict
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.runnables import RunnableLambda

# 모델 등록 및 빌드 함수 (RAG 타입 레지스트리)
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_agent(key: str):
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        k = key.strip()
        if k in MODEL_REGISTRY:
            raise KeyError(f"Duplicate model key: {k}")
        MODEL_REGISTRY[k] = factory
        return factory
    return deco

def build_agent(key: str, **kwargs) -> Any: # key는 RAG Type을 말함 ex) naive_rag, naive_llm 등등
    k = key.strip()
    try:
        factory = MODEL_REGISTRY[k]
    except KeyError:
        raise KeyError(
            f"Unknown model key: {k}. Available: {list(MODEL_REGISTRY.keys())}"
        ) from None
    return factory(**kwargs)


class ExaoneTorchLLM:
    def __init__(self, cfg):
        self.temperature = getattr(cfg, "temperature", 0.0)
        self.max_new_tokens = getattr(cfg, "max_new_tokens", 512)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.model = torch.compile(model) if torch.__version__.startswith("2") else model
        self.model.eval()

    def invoke(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature or None,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class BaseModel:
    def __init__(self, cfg):
        self.cfg = cfg

        if "exaone" in cfg.model_name.lower():
            exa = ExaoneTorchLLM(cfg)
            self.llm = RunnableLambda(lambda x: exa.invoke(x.to_string() if hasattr(x, "to_string") else str(x)))
        
        elif "gpt" in cfg.model_name.lower():
                self.llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=cfg.model_name,
                temperature=getattr(cfg, "temperature", 0.0),
            )

        else:
            # 👉 기본: vLLM / OpenAI-compatible
            self.llm = ChatOpenAI(
                base_url=getattr(cfg, "base_url", "http://127.0.0.1:8000/v1"),
                api_key=getattr(cfg, "api_key", "EMPTY"),
                model=cfg.model_name,
                temperature=getattr(cfg, "temperature", 0.0),
            )
