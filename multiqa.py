from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

def iterative_self_ask_rag_OllamaLLM(
    question: str,
    retriever,
    llm: OllamaLLM,
    max_iter: int = 5
):
    """
    Iterative Self‑Ask + RAG + Sufficiency Judge + Final CoT using OllamaLLM LLM.
    """
    # 1) RetrievalQA 체인
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    # 2) Judge용 PromptTemplate
    judge_prompt = PromptTemplate(
        input_variables=["original_question", "context"],
        template=(
            "Original Question:\n{original_question}\n\n"
            "Collected Q&A so far:\n{context}\n\n"
            "Do you now have enough information to answer the original question? "
            "Answer YES or NO."
        )
    )
    judge_chain = LLMChain(llm=llm, prompt=judge_prompt)

    # 3) Follow‑up 생성용 PromptTemplate
    followup_prompt = PromptTemplate(
        input_variables=["original_question", "context"],
        template=(
            "You are a helpful assistant. If you lack information to answer, "
            "generate exactly one follow-up question.\n\n"
            "Original Question:\n{original_question}\n\n"
            "Collected Q&A so far:\n{context}\n\n"
            "If more info is needed, output one follow-up question.\n\n"
            "Follow-up Question:"
        )
    )
    followup_chain = LLMChain(llm=llm, prompt=followup_prompt)

    # 4) Iterative loop
    context = ""
    qas = []
    for _ in range(max_iter):
        # sufficiency 판단
        judge = judge_chain.run(original_question=question, context=context).strip().lower()
        
        # follow-up 생성
        follow_up = followup_chain.run(original_question=question, context=context).strip()

        # retrieval
        docs = retriever.get_relevant_documents(follow_up)
        combined = "\n".join([d.page_content for d in docs])

        # answer 생성
        answer = qa_chain.run(combined + f"\n\nQ: {follow_up}\nA:").strip()
        qas.append((follow_up, answer))
        context += f"Q: {follow_up}\nA: {answer}\n"

    # 5) 최종 CoT reasoning
    final_prompt = PromptTemplate(
        input_variables=["original_question", "context"],
        template=(
            "Original Question:\n{original_question}\n\n"
            "Step-by-step Q&A collected:\n{context}\n\n"
            "Based on this, provide a concise final answer:"
        )
    )
    final_chain = LLMChain(llm=llm, prompt=final_prompt)
    final_answer = final_chain.run(original_question=question, context=context).strip()

    return {"final_answer": final_answer, "qa_pairs": qas}


# ── 실행 예시 ──
if __name__ == "__main__":
    from src.store_db import load_stores

    # OllamaLLM LLM 세팅 (local OllamaLLM HTTP API 서버가 필요)
    llm = OllamaLLM(
        model="Mistral-nemo",  # OllamaLLM에 로드된 모델 이름
        temperature=0.7,
        top_p=0.9,
        n=1
    )

    # retriever 로드
    retr_qna, retr_crop, retr_soil, _, _ = load_stores(ndocs=3)
    question = "What properties are critical when evaluating soils for building foundations?"

    res = iterative_self_ask_rag_OllamaLLM(
        question=question,
        retriever=retr_soil,
        llm=llm,
        max_iter=3
    )

    print("Final Answer:", res["final_answer"])
    for q, a in res["qa_pairs"]:
        print(f"- Q: {q}\n  A: {a}\n")
