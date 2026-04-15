import copy
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.infer.agents import LLM_agent, RAG_agent, SimGate_agent, Judge_LM
from src.retrieval import build_retrievers, Multi_Retriever, Embeddor
from src.infer.qa_type.base import QAtype
from src.prompts import saq_rag_prompt, saq_llm_prompt, relv_prompt, faith_prompt
from src.infer.qa_type.load_input import saq_input
from src.infer.qa_type.load_output import gate_rag_output, gate_sota_output


if __name__ == "__main__":
    load_dotenv()
    CFG_infer = get_config("configs/config_infer.yaml")
    CFG_emb = get_config("configs/config_emb.yaml")

    ## retrievers
    emb = Embeddor(CFG_emb.embedor_model_name)
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

    gpt = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    # 주어진 문서(context)가 특정 문장(claim 또는 answer)을 근거로 뒷받침하는지를 판단하는 LM
    judge_lm = Judge_LM(
        model_name=CFG_infer.judge_model,
        relv_prompt=relv_prompt,
        faith_prompt=faith_prompt
    )

    '''
    <Builder Pattern>
    조립은 외부에서 수행 — prompt/input/output의 다양한 조합을 지원하기 위해
    조합마다 클래스를 따로 만들면 N^3 조합이 생겨 비효율적이므로 빌더 패턴 사용
    '''
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

    sota_agent = LLM_agent(
        llm=gpt,
        qa_type=qa_sota
    )

    simgate_agent = SimGate_agent(
        RAG_agent=rag_agent,
        SOTA_agent=sota_agent,
        judge_lm=judge_lm,
        relv_thre=CFG_infer.relv_thre,
        faith_thre=CFG_infer.faith_thre
    )

    print("\n")
    print("==================================================")
    print("==================================================")
    print("     [안녕? 나는 농업 전문가야! 뭐든지 알고있지]")
    print("==================================================")
    print("==================================================")
    print("\n")

    # -----------------------------
    # mode 선택
    # -----------------------------
    while True:
        mode = input("모드를 선택하세요 (rag / simgate) → ").strip().lower()
        if mode in ["rag", "simgate"]:
            break
        print("올바른 모드를 입력해주세요: rag 또는 simgate\n")

    print(f"\n현재 선택된 모드: {mode}\n")

    while True:
        print("q를 누르면 챗봇이 종료됩니다.")
        print("mode를 입력하면 모드를 다시 선택할 수 있습니다.")
        print("\n")
        question = input("Question → ").strip()
        print("\n\n")

        if question == "q":
            break

        if question.lower() == "mode":
            while True:
                mode = input("모드를 선택하세요 (rag / simgate) → ").strip().lower()
                if mode in ["rag", "simgate"]:
                    break
                print("올바른 모드를 입력해주세요: rag 또는 simgate\n")
            print(f"\n현재 선택된 모드: {mode}\n")
            continue

        sample = {"id": "chat", "question": question, "answer": ""}

        if mode == "rag":
            out = rag_agent.answer_once(sample)
            agent_mode = "rag"
        elif mode == "simgate":
            out = simgate_agent.answer_once(sample)
            agent_mode = out.get("agent", "")

        print(f"\n[Answer | agent_mode={agent_mode}]\n{out['generated']}")
        print()