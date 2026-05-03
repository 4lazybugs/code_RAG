from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

def qa2d(question: str, answer: str, chain) -> str:
    """Q+A → Declarative sentence"""
    resp = chain.invoke({"question": question, "answer": answer})
    return resp.content.strip()

def qa2d_batch(qa_list: list[dict], chain) -> list[dict]:
    inputs = [{"question": item["question"], "answer": item["answer"]} for item in qa_list]
    responses = chain.batch(inputs, config={"max_concurrency": 10})
    return [
        {**item, "declarative": resp.content.strip()}
        for item, resp in tqdm(zip(qa_list, responses), total=len(qa_list), desc="QA2D 변환")
    ]