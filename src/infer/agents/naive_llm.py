from dataclasses import dataclass, field
from dataclasses import dataclass, field

from typing import Any, Dict, Tuple, List
from .base import BaseModel
from src.infer.qa_type.base import QAtype

from dataclasses import dataclass, field

from typing import Any, Dict, Tuple, List
from .base import BaseModel
from src.infer.qa_type.base import QAtype

@dataclass
class LLM_agent(BaseModel):
    """
    Retrieval 없이 LLM만 사용.
    QAMode(prompt + build_inputs)를 그대로 사용한다.
    """
    llm: Any
    qa_type: Any
    #retriever: Any  <- retriever는 필요없음
    
    def __post_init__(self):
        self.qa_prompt = self.qa_type.prompt
        self.chain = self.qa_prompt | self.llm

    def answer_once(self, raw_input: Dict[str, Any] = None) -> Tuple[str, str, str, str]:
        input_dic = self.qa_type.load_input(raw_input)
        # quesion이 안쓰이지만 input_dic의 key값인 걸 알려주기 위해 적었음
        question = input_dic["question"]

        print(f"=============== Answering a question .... ================")
        #breakpoint()
        gen_ans = self.chain.invoke(input_dic).content
        #breakpoint()
        print("====== Answer generated! ===============")
        output_dic = self.qa_type.load_output(raw_input, gen_ans)

        return output_dic