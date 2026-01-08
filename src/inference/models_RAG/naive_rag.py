from typing import Any, Dict, Tuple, List
from models_RAG.base import BaseModel
from models_RAG.qa_mode.base import QAMode

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
