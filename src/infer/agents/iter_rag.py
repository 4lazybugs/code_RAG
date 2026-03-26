import json
from typing import Any, Dict, Tuple, List
from dataclasses import dataclass

from .base import BaseModel
from .naive_rag import RAG_agent
from src.infer.qa_type.base import QAtype
from src.prompts.qa_type import iter_rag_prompt

@dataclass
class IterRAG_agent(BaseModel):
    rag_agent: RAG_agent
    llm: Any
    qa_type: Any
    max_iter: int = 3

    def __post_init__(self):
        # rag_agent
        self.rag_prompt = self.rag_agent.qa_type.prompt
        self.rag_chain = self.rag_prompt | self.rag_agent.llm
        self.load_rag_input = self.rag_agent.qa_type.load_input
        self.load_rag_output = self.rag_agent.qa_type.load_output

        # iter_agent
        self.iter_prompt = self.qa_type.prompt
        self.iter_chain = self.iter_prompt | self.llm 
        self.load_iter_input = self.qa_type.load_input
        self.load_iter_output = self.qa_type.load_output

        self.retriever = self.rag_agent.retriever 

    def answer_once(self, raw_input: Dict[str, Any] = None) -> Tuple[str, str, str, str]:
        rag_input = self.load_rag_input(raw_input)

        ori_ques = rag_input["question"]
        rag_input["original_question"] = ori_ques

        for i in range(self.max_iter):
            question = rag_input["question"]
            retrieved = self.retriever.invoke(question)
            print("========== retrieved completed! =========")
            rag_input["context"] = "\n\n".join(doc.page_content for doc in retrieved)

            print(f"=============== Answering a question .... ================")
            gen_ans = self.rag_chain.invoke(rag_input).content
            rag_output = self.load_rag_output(rag_input, gen_ans, retrieved) 

            print(f"=============== {i+1}th Iteration for question.... ==============")
            
            iter_result = self.iter_chain.invoke(rag_output)
            iter_output = self.load_iter_output(rag_output, iter_result)
            if iter_output["iter_needed"] is False:
                print("====== No more iteration needed! ===============")
                items = list(iter_output.items())
                items.insert(2, ("original_question", ori_ques)) # original question을 3번째 키로 이동
                iter_output = dict(items)
                return iter_output
            
            print(f"=============== {i+1}th Iteration completed ==============")
            rag_input["question"] = iter_output["question"] 

        print("====== ALL Iterations completed! ===============")
        items = list(iter_output.items())
        items.insert(2, ("original_question", ori_ques)) # original question을 3번째 키로 이동
        iter_output = dict(items)
        return iter_output