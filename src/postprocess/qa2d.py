from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def qa2d(question: str, answer: str, chain) -> str:
    """Q+A → Declarative sentence"""
    resp = chain.invoke({"question": question, "answer": answer})
    return resp.content.strip()


def qa2d_batch(qa_list: list[dict], chain) -> list[dict]:
    """
    qa_list: [{"question": ..., "answer": ...}, ...]
    returns: [{"question": ..., "answer": ..., "declarative": ...}, ...]
    """
    results = []
    for item in qa_list:
        d = qa2d(item["question"], item["answer"], chain)
        results.append({**item, "declarative": d})
    return results