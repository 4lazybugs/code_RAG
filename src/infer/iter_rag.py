import json
from typing import Any, Dict, Tuple, List

from .base import BaseModel
from src.infer.qa_type.base import QAtype
from src.prompts.infer import iter_prompt

class IterRAG_agent(BaseModel):
    def __init__(self, cfg, retriever: Any, qa_type: QAtype, llm: Any, max_iter: int, prompt=iter_prompt):
        super().__init__(cfg)
        self.retriever = retriever
        self.qa_type = qa_type
        self.llm = llm
        self.chain = prompt | self.llm
        self.max_iter = max_iter
        self.last_retrieved_docs: list[Any] = []

    def answer_once(self, ans_input: Dict[str, Any]) -> Tuple[str, str, str, str]:
        retrieved = self.retriever.invoke(ans_input["question"])
        ans_input["retrieved"] = retrieved

        agent_input = self.qa_type.load_inputs(ans_input)
        gen_ans = self.qa_type.return_result(agent_input)
        #breakpoint()
        return ans_input["id"], ans_input["question"], gen_ans, retrieved
    
    @staticmethod
    # (True(bool), "true", "1", "yes") -> True, 그 외는 False로 간주
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes")
        return False

    def answer_iter(self, ans_input: Dict[str, Any]) -> Tuple[str, str, str, str]:
        print(f"=============== Iteration 1 for question_{ans_input['id']}.... ==============")
        q_id, question, gen_ans, retrieved = self.answer_once(ans_input)
        pure_question = question  # 원래 질문을 보존

        for i in range(self.max_iter - 1): # 이미 위에서 한 번 답변했으므로 max_iter - 1
            iter_decision = self.chain.invoke({"question": question, "answer": gen_ans})
            iter_decision = json.loads(iter_decision.content)[0]
            print(f"====== is_final: {iter_decision.get('is_final')} =====")
            is_final = self.to_bool(iter_decision.get("is_final", True)) # is_final이 없는 경우 loop 탈출 
            
            if is_final: break

            if not is_final:
                ans_input["question"] = pure_question + "\n\n" + iter_decision["improved_prompt"]
                q_id, question, gen_ans, retrieved = self.answer_once(ans_input)
                print(f"=============== Iteration {i+2} for question_{ans_input['id']}.... ==============")
            
        return q_id, question, gen_ans, retrieved

    def answer_all(self, ans_inputs: list[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
        return [self.answer_iter(p) for p in ans_inputs]
    