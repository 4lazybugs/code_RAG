from dataclasses import dataclass, field

from typing import Any, Dict, Tuple, List
from .base import BaseModel
from src.infer.qa_type.base import QAtype

@dataclass
class RAG_agent(BaseModel):
    """
    Retrieval + LLM.
    """
    llm: Any
    qa_type: Any
    retriever: Any
    
    def __post_init__(self):
        self.prompt = self.qa_type.prompt
        self.chain = self.prompt | self.llm

    def answer_once(self, raw_input: Dict[str, Any] = None) -> Tuple[str, str, str, str]:
        input_dic = self.qa_type.load_input(raw_input)
        #breakpoint()
        question = input_dic["question"]
        retrieved = self.retriever.invoke(question)
        print("========== retrieved completed! =========")
        input_dic["context"] = "\n\n".join(doc.page_content for doc in retrieved)

        print(f"=============== Answering a question .... ================")
        #breakpoint()
        gen_ans = self.chain.invoke(input_dic).content
        #breakpoint()
        print("====== Answer generated! ===============") 
        output_dic = self.qa_type.load_output(input_dic, gen_ans, retrieved)

        return output_dic

    def answer_all(self, input_dics: list[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
        return [self.answer_once(p) for p in input_dics]
