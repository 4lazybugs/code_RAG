import copy
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.infer.agents import LLM_agent, RAG_agent, SimGate_agent
from src.retrieval import build_retrievers, Multi_Retriever, Embeddor
from src.infer.qa_type.base import QAtype
from src.prompts import saq_rag_prompt, saq_llm_prompt, rag_consult_prompt, relv_prompt, faith_prompt
from src.infer.qa_type.load_input import saq_input
from src.infer.qa_type.load_output import gate_rag_output, gate_sota_output


if __name__ == "__main__":
    load_dotenv()
    CFG_infer = get_config("configs/config_infer.yaml")
    CFG_preproc = get_config("configs/config_preproc.yaml")

    ## retrievers
    emb = Embeddor(CFG_preproc.embedor_model_name)
    retriever_list = build_retrievers(
        vec_root=Path("db/vector_db"),
        emb=emb,
    )
    multi_retriever = Multi_Retriever(
        retrievers=retriever_list,
        k_each=3,
        top_k=5,
    )

    #### LM
    qwen = ChatOpenAI(
        model=CFG_infer.model_name,
        temperature=0,
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )

    '''
    <Builder Pattern>
    조립은 외부에서 수행 — prompt/input/output의 다양한 조합을 지원하기 위해
    조합마다 클래스를 따로 만들면 N^3 조합이 생겨 비효율적이므로 빌더 패턴 사용
    '''
    qa_rag = (QAtype()
        .set_prompt(rag_consult_prompt)
        .set_inputs(saq_input)
        .set_outputs(gate_rag_output)
        .build())

    '''
    <Strategy Pattern>
    agent는 고정한 채로, 주입하는 qa_type만 바꿔서
    agent가 llm_mcq(전략1) 또는 rag_saq(전략2)로 동작하게 만들 수 있음.
    '''
    rag_agent = RAG_agent(
        llm=qwen,
        qa_type=qa_rag,
        retriever=multi_retriever
    )

    while True:
        print("q를 누르면 챗봇이 종료됩니다.")
        question = input("Question → ").strip()
        print("\n\n")

        if question == "q":
            break

        sample = {"id": "chat", "question": question, "answer": ""}
        out = rag_agent.answer_once(sample)

        print(f"\nAnswer is.... \n\n{out['generated']}")