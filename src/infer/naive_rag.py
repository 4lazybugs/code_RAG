from typing import Any, Dict, Tuple, List
from .base import BaseModel
from src.infer.qa_type.base import QAtype
from .base import register_agent

class RAG_agent(BaseModel):
    def __init__(self, cfg, retriever: Any, qa_type: QAtype):
        super().__init__(cfg)
        self.retriever = retriever
        self.qa_type = qa_type
        self.last_retrieved_docs: list[Any] = []

    def answer_once(self, gen_input: Dict[str, Any]) -> Tuple[str, str, str, str]:
        retrieved = self.retriever.invoke(gen_input["question"])
        gen_input["retrieved"] = retrieved

        agent_input = self.qa_type.load_inputs(gen_input)
        gen_ans = self.qa_type.return_result(agent_input)
        print(f"=============== Answering question_{gen_input['id']}.... ==============")
        #breakpoint()
        return gen_input["id"], gen_input["question"], gen_ans, retrieved

    def answer_all(self, gen_inputs: list[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
        return [self.answer_once(p) for p in gen_inputs]
    
    @register_agent("naive_rag")
    def make_naive_rag(*, cfg, qa_mode, retriever, **_):
        # qa_mode를 RAG_agent의 qa_type으로 연결
        return RAG_agent(cfg=cfg, retriever=retriever, qa_type=qa_mode)