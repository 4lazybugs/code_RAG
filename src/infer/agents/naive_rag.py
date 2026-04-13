from dataclasses import dataclass, field

from typing import Any, Dict, Tuple, List
from .base import BaseModel
from src.infer.qa_type.base import QAtype

@dataclass
class RAG_agent(BaseModel):
    llm: Any
    qa_type: Any
    retriever: Any
    
    def __post_init__(self):
        self.qa_prompt = self.qa_type.prompt
        self.chain = self.qa_prompt | self.llm

    def answer_once(self, raw_input: Dict[str, Any] = None):
        input_dic = self.qa_type.load_input(raw_input)
        question = input_dic["question"]
        retrieved = self.retriever.invoke(question)
        print("========== retrieved completed! =========")
        input_dic["context"] = "\n\n".join(doc.page_content for doc in retrieved)

        print(f"=============== Answering a question .... ================")
        
        # ↓ 여기가 핵심 변경: .content 대신 전체 객체 받기
        response = self.chain.invoke(input_dic)
        gen_ans = response.content

        # logprob 추출
        token_logprobs = []
        logprobs_meta = response.response_metadata.get("logprobs", {})
        if logprobs_meta and "content" in logprobs_meta:
            token_logprobs = [
                t["logprob"]
                for t in logprobs_meta["content"]
                if t.get("logprob") is not None
            ]

        print("====== Answer generated! ===============") 
        output_dic = self.qa_type.load_output(raw_input, gen_ans, retrieved)
        
        # ↓ logprob을 output_dic에 추가
        output_dic["logprobs"] = token_logprobs

        return output_dic