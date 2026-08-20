from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.infer.agents import RAG_agent as _RAG_agent
from src.retrieval import build_retrievers, Multi_Retriever, Embeddor
from src.infer.qa_type.base import QAtype
from src.prompts import rag_consult_prompt, rag_paid_consult_prompt, interpretation_prompt
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




        qa_rag_paid = (QAtype()
            .set_prompt(rag_paid_consult_prompt)
            .set_inputs(saq_input)
            .set_outputs(gate_rag_output)
            .build())

        self.agent_paid = _RAG_agent(
            llm=qwen,
            qa_type=qa_rag_paid,
            retriever=multi_retriever,
        )


    def interpret(self, query: str) -> dict:
        chain = interpretation_prompt | self.agent.llm
        response = chain.invoke({"question": query})
        import json
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"crop": "", "modelType": [], "task": [], "keyword": []}

    def chat(self, prompt: str, flag_paid: bool = False, **_context) -> str:
        agent = self.agent_paid if flag_paid else self.agent

        sample = {"id": "api", "question": prompt, "answer": ""}
        out = agent.answer_once(sample, flag_paid=flag_paid, **_context)
        return out["generated"]