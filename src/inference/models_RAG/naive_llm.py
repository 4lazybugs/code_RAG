from models_RAG.base import BaseModel
from typing import Any, Dict
from models_RAG.qa_mode.base import QAMode

class NaiveLLM(BaseModel):
    """
    Retrieval 없이 LLM만 사용하는 Answerer.
    QAMode(prompt + build_inputs)를 그대로 사용한다.
    """

    def __init__(self, cfg, qa_mode: QAMode):
        super().__init__(cfg)
        self.qa_mode = qa_mode
        self.chain = self.qa_mode.prompt | self.llm

    def answer_once(self, payload: Dict[str, Any]) -> str:
        # payload는 그대로 전달, reviews/docs는 없음
        payload = dict(payload)
        payload.setdefault("reviews", "")
        payload.setdefault("docs", [])

        inputs = self.qa_mode.build_inputs(payload)
        result = self.chain.invoke(inputs)
        return self.to_plain_text(result)

    def answer_all(self, payloads: list[Dict[str, Any]]) -> list[str]:
        return [self.answer_once(p) for p in payloads]
