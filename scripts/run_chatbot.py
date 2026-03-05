import os
import argparse
import yaml
import importlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.infer import build_agent
from src.retrieval.retriever import Embeddor, build_retrievers, Multi_Retriever


def load_cfg(path: str = "configs/config_infer.yaml") -> argparse.Namespace:
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}

    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default=d.get("model_name"))
    p.add_argument("--embedor_model_name", type=str, default=d.get("embedor_model_name"))
    p.add_argument("--base_url", type=str, default=d.get("base_url", "http://127.0.0.1:8000/v1"))
    p.add_argument("--api_key", type=str, default=d.get("api_key", "EMPTY"))
    p.add_argument("--temperature", type=float, default=float(d.get("temperature", 0.0)))
    return p.parse_args()


def build_saq(llm):
    SAQ = getattr(importlib.import_module("src.infer.qa_type.saq"), "SAQ")
    return SAQ(llm=llm)  # ✅ llm 주입해서 chain=None 방지


def main():
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

    load_dotenv()
    cfg = load_cfg()

    llm = ChatOpenAI(
        model=cfg.model_name,
        temperature=cfg.temperature,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
    )

    emb = Embeddor(cfg.embedor_model_name)
    db_pth = Path("db/vector_db")
    retrievers = build_retrievers(vec_root=db_pth, emb=emb)
    multi_retriever = Multi_Retriever(retrievers=retrievers, k_each=5, top_k=5)

    print("[사용 가이드]\nRAG | RAG 응답\nLLM | retrieval 없이 LLM 응답\n(q) | 종료")

    while True:
        mode = input("Mode → RAG/LLM (q to quit): ").strip().lower()
        if mode == "q":
            break
        if mode not in ("rag", "llm"):
            print("Invalid mode.\n")
            continue

        question = input("Question → ").strip()
        if question.lower() == "q":
            break

        qa_mode = build_saq(llm)
        payload = {"id": "chat", "question": question}

        if mode == "rag":
            expert = build_agent("naive_rag", cfg=cfg, qa_mode=qa_mode, retriever=multi_retriever)
        else:
            expert = build_agent("naive_llm", cfg=cfg, qa_mode=qa_mode)

        out = expert.answer_once(payload)
        ans = out[2] if isinstance(out, tuple) and len(out) >= 3 else out
        print(f"\n[Answer]\n{ans}\n")


if __name__ == "__main__":
    main()