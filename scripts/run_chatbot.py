import copy
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from minicheck.minicheck import MiniCheck

from src.config import get_config
from src.infer.agents import LLM_agent, RAG_agent, Gateway_agent
from src.retrieval import build_bm25s, Multi_BM25s, build_retrievers, Multi_Retriever, Embeddor
from src.infer.qa_type.base import QAtype
from src.prompts.qa_type import saq_rag_prompt, saq_llm_prompt, iter_rag_prompt, logprob_prompt
from src.infer.qa_type.load_input import saq_input
from src.infer.qa_type.load_output import router_output, gate_llm_output, gate_rag_output, gate_sota_output


if __name__ == "__main__":
    load_dotenv()
    CFG = get_config("configs/config_infer.yaml")

    ## LM
    qwen = ChatOpenAI(
        model=CFG.model_name,
        temperature=0,
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )
    gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    # MiniCheck는 주어진 문서(context)가 특정 문장(claim 또는 answer)을
    # 실제로 근거로 뒷받침하는지를 판단하는 LLM 기반 검증 모델
    judge_lm = MiniCheck(model_name=CFG.judge_model, cache_dir=CFG.judge_cache_dir)

    ## retrievers
    emb = Embeddor(CFG.embedor_model_name)
    retriever_list = build_retrievers(vec_root=Path("db/vector_db"), emb=emb)
    multi_retriever = Multi_Retriever(retrievers=retriever_list, k_each=3, top_k=5)

    ## QAtype
    qa_router = (QAtype()
        .set_prompt(logprob_prompt)
        .set_inputs(saq_input)
        .set_outputs(router_output)
        .build())

    qa_llm = (QAtype()
        .set_prompt(saq_llm_prompt)
        .set_inputs(saq_input)
        .set_outputs(gate_llm_output)
        .build())

    qa_rag = (QAtype()
        .set_prompt(saq_rag_prompt)
        .set_inputs(saq_input)
        .set_outputs(gate_rag_output)
        .build())

    qa_sota = (QAtype()
        .set_prompt(saq_llm_prompt)
        .set_inputs(saq_input)
        .set_outputs(gate_sota_output)
        .build())

    ## agents
    router_agent = LLM_agent(llm=qwen, qa_type=qa_router)
    llm_agent    = LLM_agent(llm=qwen, qa_type=qa_llm)
    rag_agent    = RAG_agent(llm=qwen, qa_type=qa_rag, retriever=multi_retriever)
    sota_agent   = LLM_agent(llm=gpt,  qa_type=qa_sota)

    gateway_agent = Gateway_agent(
        Router_agent=router_agent,
        LLM_agent=llm_agent,
        RAG_agent=rag_agent,
        SOTA_agent=sota_agent,
        judge_lm=judge_lm,
    )

    print("\n")
    print("==================================================")
    print("==================================================")
    print("     [안녕? 나는 농업 전문가야! 뭐든지 알고있지]")
    print("==================================================")
    print("==================================================")
    print("\n")

    while True:
        print("q를 누르면 챗봇이 종료됩니다.")
        print("\n")
        question = input("Question → ").strip()
        print("\n\n")
        if question == "q":
            break

        out   = gateway_agent.answer_once({"id": "chat", "question": question, "answer": ""})
        route = out.get("agent")

        print(f"\n[Answer | route={route}]\n{out['generated']}")
        print()