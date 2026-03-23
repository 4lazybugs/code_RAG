import os
import math
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.config import get_config
from src.infer.agents import LLM_agent, RAG_agent
from src.retrieval import build_retrievers, Multi_Retriever, Embeddor
from src.infer.qa_type.base import QAtype
from src.infer.qa_type.prompts import chatbot_llm_prompt, chatbot_rag_prompt
from src.infer.qa_type.load_input import saq_input
from src.infer.qa_type.load_output import llm_output, rag_output
from src.retrieval import build_bm25s, Multi_BM25s


def get_true_probability(client: OpenAI, model: str, question: str,
                         answer: str, context: str = None) -> float:
    if context:
        # RAG 모드: 답변이 검색된 context에 의해 뒷받침되는지 측정
        # RAGAS의 Faithfulness 구현, antropic 논문은 적절치 못함
        prompt = (
            f"Context: {context}\n\n"
            f"Q: {question}\n"
            f"A: {answer}\n"
            f"Is the answer supported by the context? Answer only Yes or No:"
        )
    else:
        # LLM 모드: 모델 자신의 확신도 측정 (Kadavath et al. 2022)
        #llm 은 LLM은 internal knowledge confidence만 평가
        # P(True)는 모델 자신의 확신도이지, DB 기반 정확도가 아님
        # RAG가 더 정확한 답을 하더라도,
        # 모델이 학습 때 못 본 전문 내용이면 오히려 낮게 평가함
        # 애초에 자기 답변 확신도는 자기 모델 내부 지식과 있냐 없냐 판단하기 때문에 rag 라우팅에 적합x
        prompt = (
            f"Q: {question}\n"
            f"A: {answer}\n"
            f"Is the above answer correct? Answer only Yes or No:"
        )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
        temperature=0.0,
    )
    for candidate in response.choices[0].logprobs.content[0].top_logprobs:
        if candidate.token.strip().lower() == "yes":
            return math.exp(candidate.logprob) # softmax로 구현
    return 0.0


if __name__ == "__main__":
    load_dotenv()
    CFG = get_config("configs/config_infer.yaml")

    llm = ChatOpenAI(
        model=CFG.model_name,
        temperature=CFG.temperature,
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )

    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

    gpt = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    emb = Embeddor(CFG.embedor_model_name)
    retriever_list = build_retrievers(vec_root=Path("db/vector_db"), emb=emb)
    retriever = Multi_Retriever(retrievers=retriever_list, k_each=3, top_k=5)
    retriever_list = build_bm25s(vec_root=Path("db/vector_db"), k_each=4)
    lex_retriever = Multi_BM25s(retrievers=retriever_list, top_k=5)

    qa_type_llm = (QAtype()
        .set_prompt(chatbot_llm_prompt)
        .set_inputs(saq_input)
        .set_outputs(llm_output)
        .build())

    qa_type_rag = (QAtype()
        .set_prompt(chatbot_rag_prompt)
        .set_inputs(saq_input)
        .set_outputs(rag_output)
        .build())

    llm_agent = LLM_agent(llm=llm, qa_type=qa_type_llm)
    rag_agent = RAG_agent(llm=llm, qa_type=qa_type_rag, retriever=lex_retriever)
    gpt_agent = LLM_agent(llm=gpt, qa_type=qa_type_llm)

    print("[사용 가이드]\nrag | RAG 응답\nllm | retrieval 없이 LLM 응답\nq   | 종료")

    while True:
        mode = input("Mode → rag/llm (q to quit): ").strip().lower()
        if mode == "q":
            break
        if mode not in ("rag", "llm"):
            print("Invalid mode.\n")
            continue

        question = input("Question → ").strip()
        payload = {"id": "chat", "question": question, "answer": ""}

        agent = rag_agent if mode == "rag" else llm_agent
        out = agent.answer_once(payload)

        if mode == "rag":
            context = "\n\n".join("\n".join(r["content"]) for r in out["retrieved"])
        else:
            context = None

        prob = get_true_probability(client, CFG.model_name, question, out["generated"], context)
        bar  = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        label = "P(Supported)" if mode == "rag" else "P(True)"

        if prob > 0.6:
            print(f"\n[Answer]\n{out['generated']}")
        else:
            gpt_payload = {"id": "chat", "question": question, "answer": ""}
            gpt_out = gpt_agent.answer_once(gpt_payload)
            print(f"\n[Answer - GPT fallback]\n{gpt_out['generated']}")

        print(f"[{label}] {bar} {prob:.1%}\n")