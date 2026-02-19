from typing import Any, Dict, Tuple, List
from .base import BaseModel
from ..qa_mode.base import QAMode
from .base import build_agent, register_agent  # re-export build_model for external use

class NaiveRag(BaseModel):
    def __init__(self, cfg, retriever: Any, qa_mode: QAMode):
        super().__init__(cfg)
        self.retriever = retriever
        self.qa_mode = qa_mode
        self.chain = self.qa_mode.prompt | self.llm
        self.last_retrieved_docs: list[Any] = []

    def answer_once(self, payload: Dict[str, Any]) -> Tuple[str, str, str, str]:
        docs = self.retriever.invoke(payload["question"])

        payload = dict(payload)
        payload["retrieved"] = docs

        inputs = self.qa_mode.build_inputs(payload)
        result = self.chain.invoke(inputs)
        return payload["id"], payload["question"], self.to_plain_text(result), docs

    def answer_all(self, payloads: list[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
        return [self.answer_once(p) for p in payloads]

@register_agent("naive_rag")
def make_naive_rag(*, cfg, qa_mode, retriever=None, **_):
    # retriever는 밖에서 주입받는 걸 원칙으로
    if retriever is None:
        raise ValueError("retriever must be provided for naive_rag")
    return NaiveRag(cfg=cfg, retriever=retriever, qa_mode=qa_mode)