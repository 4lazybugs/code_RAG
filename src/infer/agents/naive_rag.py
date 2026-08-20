from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .base import BaseModel


def pick_fields(data: Any, keys: List[str]) -> Any:
    """list[dict] 또는 dict에서 지정된 키만 남긴다."""
    if not data or not keys:
        return data
    if isinstance(data, list):
        return [
            {k: row[k] for k in keys if k in row}
            for row in data
            if isinstance(row, dict)
        ]
    if isinstance(data, dict):
        return {k: data[k] for k in keys if k in data}
    return data

@dataclass
class RAG_agent(BaseModel):
    llm: Any
    qa_type: Any
    retriever: Any
    
    def __post_init__(self):
        self.qa_prompt = self.qa_type.prompt
        self.chain = self.qa_prompt | self.llm

    def answer_once(
        self,
        raw_input: Dict[str, Any] = None,
        flag_paid: bool = False,
        env_context: Any = None,
        nutrient_context: Any = None,
        spectral_context: Any = None,
        env_keys: Optional[List[str]] = None,
        nutrient_keys: Optional[List[str]] = None,
        spectral_keys: Optional[List[str]] = None,
    ):
        input_dic = self.qa_type.load_input(raw_input)
        question = input_dic["question"]
        retrieved = self.retriever.invoke(question)
        print("========== retrieved completed! =========")

        vdb_context = "\n\n".join(doc.page_content for doc in retrieved)

        if flag_paid:
            env_filtered = pick_fields(env_context, env_keys) if env_keys else env_context
            nutrient_filtered = pick_fields(nutrient_context, nutrient_keys) if nutrient_keys else nutrient_context
            spectral_filtered = pick_fields(spectral_context, spectral_keys) if spectral_keys else spectral_context

            rdb_context_parts = []
            if env_filtered:
                rdb_context_parts.append(f"[환경 데이터]\n{env_filtered}")
            if nutrient_filtered:
                rdb_context_parts.append(f"[양액 데이터]\n{nutrient_filtered}")
            if spectral_filtered:
                rdb_context_parts.append(f"[생성요소AI 데이터]\n{spectral_filtered}")
            rdb_context = "\n\n".join(rdb_context_parts)

            input_dic["context"] = "\n\n".join(
                filter(None, [vdb_context, rdb_context])
            )
        else:
            input_dic["context"] = vdb_context

        print(f"=============== Answering a question .... (paid={flag_paid}) ================")

        response = self.chain.invoke(input_dic)
        gen_ans = response.content

        print("====== Answer generated! ===============") 
        output_dic = self.qa_type.load_output(raw_input, gen_ans, retrieved)

        return output_dic