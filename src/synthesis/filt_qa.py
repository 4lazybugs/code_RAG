from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

qa2d_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that converts question-answer pairs into declarative sentences.
Given a question and answer, create a single natural declarative sentence (D) that combines both.
Return ONLY the declarative sentence, nothing else."""),
    ("user", """Question: {question}
Answer: {answer}

Convert to a declarative sentence:""")
])


def qa2d(question: str, answer: str, llm=None) -> str:
    """Q+A → Declarative sentence"""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = qa2d_prompt | llm
    resp = chain.invoke({"question": question, "answer": answer})
    return resp.content.strip()


def qa2d_batch(qa_list: list[dict], llm=None) -> list[dict]:
    """
    qa_list: [{"question": ..., "answer": ...}, ...]
    returns: [{"question": ..., "answer": ..., "declarative": ...}, ...]
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    results = []
    for item in qa_list:
        d = qa2d(item["question"], item["answer"], llm)
        results.append({
            **item,
            "declarative": d,
        })
    return results


# ── 사용 예시 ──────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa_pairs = [
        {"question": "Who called Taylor?", "answer": "Liz"},
        {"question": "Who called Taylor?", "answer": "Ron"},
        {"question": "Who called Taylor?", "answer": "a doctor"},
    ]

    results = qa2d_batch(qa_pairs, llm)
    for r in results:
        print(f"Q: {r['question']}")
        print(f"A: {r['answer']}")
        print(f"D: {r['declarative']}")
        print()