from typing import Any, Dict, Tuple, List
from .base import BaseModel
from src.infer.qa_type.base import QAtype
from src.retrieval import build_retrievers  # re-export build_model for external use

class RAG_agent(BaseModel):
    def __init__(self, cfg, retriever: Any, qa_type: QAtype):
        super().__init__(cfg)
        self.retriever = retriever
        self.qa_type = qa_type
        self.last_retrieved_docs: list[Any] = []

    def answer_once(self, ans_input: Dict[str, Any]) -> Tuple[str, str, str, str]:
        retrieved = self.retriever.invoke(ans_input["question"])
        ans_input["retrieved"] = retrieved

        agent_input = self.qa_type.load_inputs(ans_input)
        gen_ans = self.qa_type.return_result(agent_input)
        print(f"=============== Answering question_{ans_input['id']}.... ==============")
        #breakpoint()
        return ans_input["id"], ans_input["question"], gen_ans, retrieved

    def answer_all(self, ans_inputs: list[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
        return [self.answer_once(p) for p in ans_inputs]
    