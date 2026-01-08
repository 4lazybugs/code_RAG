from .base import build_model, register_model  # re-export build_model for external use
from .naive_rag import NaiveRag
from .naive_llm import NaiveLLM
from retriever import MultiCosineRetriever, load_retrievers

# Factory functions for model registration
@register_model("naive_llm")
def make_naive_llm(*, cfg, qa_mode, **_):
    return NaiveLLM(cfg=cfg, qa_mode=qa_mode)

@register_model("naive_rag")
def make_naive_rag(*, cfg, qa_mode, k_each: int, top_k: int, **_):
    single = load_retrievers(ndocs=k_each)
    multi = MultiCosineRetriever(retrievers=single, k_each=k_each, top_k=top_k)
    return NaiveRag(cfg=cfg, retriever=multi, qa_mode=qa_mode)
