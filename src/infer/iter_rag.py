import json
from typing import Any, Dict, Tuple, List

from .base import BaseModel
from src.infer.qa_type.base import QAtype
from src.retrieval import build_retrievers

# about llm
from langchain_core.prompts import ChatPromptTemplate

iter_prompt = ChatPromptTemplate.from_template(
"""
당신은 반복 질의응답 개선 시스템이다.

입력:
- 질문
- 기존 답변

목표:
기존 답변이 질문에 대해 충분히 정확하고 완전한지 검토한다.
답변이 부족하거나 모호한 경우, 답변 개선에 필요한 보충 설명을 생성한다.

평가 기준:
1. 답변이 질문의 핵심을 직접적으로 해결하는가
2. 중요한 정보가 누락되거나 불완전하지 않은가
3. 근거 없는 추측, 불필요한 일반화, 외부 지식이 포함되지 않았는가
4. 표현이 명확하고 이해하기 쉬운가

규칙:
- 기존 답변이 충분히 정확하고 완전하다면 is_final을 true로 설정한다.
- 답변이 부족하거나 개선 여지가 있다면 is_final을 false로 설정한다.
- is_final이 false일 경우, 질문과 기존 답변을 더 잘 이해하고 개선하는 데 도움이 되는
  추가 설명 또는 보충 맥락을 improved_prompt에 작성한다.
- 새로운 사실을 만들어내지 말고, 입력된 질문과 답변 범위를 벗어나지 않는다.
- 설명, 해설, 문장 추가 없이 반드시 지정된 JSON 배열만 출력한다.

출력 형식 (JSON 배열만 출력):

[
  {{
    "is_final": true | false,
    "improved_prompt": "question과 더불어 답변하는데 도움되는 보충 설명"
  }}
]

--------------------
질문:
{question}

기존 답변:
{answer}
"""
)

class IterRAG_agent(BaseModel):
    def __init__(self, cfg, retriever: Any, qa_type: QAtype, llm: Any, max_iter: int):
        super().__init__(cfg)
        self.retriever = retriever
        self.qa_type = qa_type
        self.llm = llm
        self.chain = iter_prompt | self.llm
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
    