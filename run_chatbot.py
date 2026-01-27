import os
import argparse
import yaml
import importlib
from typing import Any, Dict

from langchain_core.messages import BaseMessage

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"


from src.inference import NaiveLLM, build_agent  # 또는 build_model(이름을 그걸로 유지한다면)
from src.inference.qa_mode import build_qa_mode
from src.inference.retriever import load_retrievers, MultiCosineRetriever

# -----------------------------
# Config
# -----------------------------
def load_yaml(path: str = "src/inference/config_infer.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in raw.items()}


def get_config() -> argparse.Namespace:
    d = load_yaml()

    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default=d.get("model_name"))
    p.add_argument("--sbert_model_name", type=str, default=d.get("sbert_model_name"))
    p.add_argument("--bert_model_name", type=str, default=d.get("bert_model_name"))
    p.add_argument("--embedor_model_name", type=str, default=d.get("embedor_model_name"))

    return p.parse_args()


# -----------------------------
# Utils
# -----------------------------
def normalize_answer(res: Any) -> str:
    if isinstance(res, tuple):
        res = res[0]
    if isinstance(res, BaseMessage):
        return (res.content or "").strip()
    if isinstance(res, dict):
        for key in ("result", "answer", "output_text", "content"):
            if key in res:
                return str(res[key]).strip()
    return str(res).strip()


def instantiate_mode(cls, cfg: Any):
    # cfg로 생성 시도 → 안 받으면(TypeError) 무인자 생성
    try:
        return cls(cfg)
    except TypeError:
        return cls()


def build_qa_mode(mode: str, cfg: Any):
    """
    프로젝트에 존재하는 qa_mode 구현체:
    - HotpotMode
    - MCQ
    - SAQ
    """
    base = "src.inference.qa_mode"
    mode = mode.lower()

    if mode in ("hotpot", "hotpotmode"):
        mod = importlib.import_module(f"{base}.hotpot")
        return instantiate_mode(getattr(mod, "HotpotMode"), cfg)

    if mode == "mcq":
        mod = importlib.import_module(f"{base}.mcq")
        return instantiate_mode(getattr(mod, "MCQ"), cfg)

    if mode == "saq":
        mod = importlib.import_module(f"{base}.saq")
        return instantiate_mode(getattr(mod, "SAQ"), cfg)

    raise ValueError("qa_mode는 hotpot/mcq/saq 중 하나여야 합니다.")


def get_help_text() -> str:
    return """
[사용 가이드]
mode      | 설명
----------------------------
RAG       | RAG 응답 
Naive_LLM   | retrieval 없이 LLM 응답
(q)       | 종료
""".strip()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = get_config()
    cfg.EMBED_MODEL = cfg.embedor_model_name

    # RAG 파라미터(프로젝트 팩토리 시그니처에 맞춤)
    K_EACH = 5
    TOP_K = 5

    print(get_help_text())

    while True:
        mode = input("Mode → RAG/LLM (q to quit): ").strip().lower()
        if mode == "q":
            break
        if mode not in ("rag", "llm"):
            print("Invalid mode.\n")
            continue

        qa_mode = build_qa_mode("saq", cfg)

        question = input("Question → ").strip()
        if question.lower() == "q":
            break

        if mode == "rag":
            single = load_retrievers(ndocs=K_EACH)
            retriever = MultiCosineRetriever(retrievers=single, k_each=K_EACH, top_k=TOP_K)

            expert = build_agent(
                "naive_rag",
                cfg=cfg,
                qa_mode=qa_mode,
                retriever=retriever,   # ✅ 반드시 주입
            )
        else:
            expert = build_agent(
                "naive_llm",
                cfg=cfg,
                qa_mode=qa_mode,
            )

        payload = {"id": "chat", "question": question}
        out = expert.answer_once(payload)

        # naive_rag: (id, question, answer, docs)
        # naive_llm: 구현에 따라 (id, question, answer, meta) 또는 유사 튜플일 가능성
        if isinstance(out, tuple) and len(out) >= 3:
            ans = out[2]
        else:
            ans = out

        print(f"\n[Answer]\n{normalize_answer(ans)}\n")


if __name__ == "__main__":
    main()
