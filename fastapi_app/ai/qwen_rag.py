from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.infer.agents import RAG_agent as _RAG_agent
from src.retrieval import build_retrievers, Multi_Retriever, Embeddor
from src.infer.qa_type.base import QAtype
from src.prompts import rag_consult_prompt
from src.infer.qa_type.load_input import saq_input
from src.infer.qa_type.load_output import gate_rag_output

from .base import AIPlatform


class QwenRAG(AIPlatform):
    def __init__(self):
        load_dotenv()

        cfg_infer = get_config("configs/config_infer.yaml")
        cfg_preproc = get_config("configs/config_preproc.yaml")

        # retriever 구성
        emb = Embeddor(cfg_preproc.embedor_model_name)
        retriever_list = build_retrievers(
            vec_root=Path("db/vector_db"),
            emb=emb,
        )
        multi_retriever = Multi_Retriever(
            retrievers=retriever_list,
            k_each=3,
            top_k=5,
        )

        # LLM
        qwen = ChatOpenAI(
            model=cfg_infer.model_name,
            temperature=0,
            base_url="http://127.0.0.1:8000/v1/",
            api_key="EMPTY",
        )

        # QAtype 빌더
        qa_rag = (QAtype()
            .set_prompt(rag_consult_prompt)
            .set_inputs(saq_input)
            .set_outputs(gate_rag_output)
            .build())

        self.agent = _RAG_agent(
            llm=qwen,
            qa_type=qa_rag,
            retriever=multi_retriever,
        )

    def chat(self, prompt: str) -> str:

        sample = {"id": "api", "question": prompt, "answer": ""}
        out = self.agent.answer_once(sample)
        return out["generated"]