# import packages in std
import time
import json
import argparse, os, yaml
from types import SimpleNamespace
from pathlib import Path
from dotenv import load_dotenv
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional

# about llm
from langchain_openai import ChatOpenAI

# import packages in src/
from src.infer import NaiveLLM, build_agent, RAG_agent, IterRAG_agent
from src.infer.qa_type import Hotpot_short
from src.retrieval import build_retrievers, Multi_Retriever, Embeddor


def load_yaml(path):
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # 환경 변수 치환 처리
    config = {}
    for k, v in raw_config.items():
        if isinstance(v, str):
            config[k] = os.path.expandvars(v)
        else:
            config[k] = v

    return config

def get_config(path):
    default_cfg = load_yaml(path=path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=default_cfg.get('model_name'))
    parser.add_argument("--qa_mode", type=str, default=default_cfg.get('qa_mode', 'MCQ'))

    args, _ = parser.parse_known_args()

    # ✅ YAML + argparse 병합 → Namespace
    cfg = {**default_cfg, **vars(args)}
    return SimpleNamespace(**cfg)


def save_predictions_json(out_path: Path, payloads, results, qa_type, model):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_not_rag = isinstance(model, NaiveLLM)

    if is_not_rag:  # NaiveLLM: results는 list[str]
        rows = [
            {
                **payload,
                "gen_answer": gen_ans,
            }
            for payload, gen_ans in zip(payloads, results)
        ]
    else:  # RAG: results는 List[Tuple[str, str, str, str]]
        rows = [
            {
                **payload,
                "id": _id,
                "question": question,
                "gen_answer": gen_ans,
                **qa_type.build_output(retrieved),
            }
            for payload, (_id, question, gen_ans, retrieved) in zip(payloads, results)
        ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    start_time = time.time()
    load_dotenv()
    CFG = get_config("configs/config_infer.yaml")
    emb = Embeddor(CFG.embedor_model_name)

    ## retrievers
    db_pth = Path("/home/jwkim/[code]생성형과제_server/db/vector_db")
    retriever_list = build_retrievers(vec_root=db_pth, emb=emb)
    retriever = Multi_Retriever(retrievers=retriever_list, k_each=3, top_k=5)

    ## llm
    llm = ChatOpenAI(
         model="Qwen/Qwen2.5-32B-Instruct",
         temperature=0,
         base_url="http://127.0.0.1:8000/v1",
         api_key="EMPTY",
    )

    ## agent(infer)
    qa_type = Hotpot_short(llm)  # QAtype 인스턴스 생성
    agent = IterRAG_agent(CFG, retriever, qa_type=qa_type, llm=llm, max_iter=3)

    input_path = Path("db/qa_data/test/HOTPOT_[text]weather_manual.json")
    with open(input_path, "r", encoding="utf-8") as f:
        agent_inputs = json.load(f)

    gen_answers = agent.answer_all(agent_inputs)

    output_path = Path("results/inferenced/test/hotpotqa_test/hotpot_test_iter_rag.json")
    save_predictions_json(output_path, agent_inputs, gen_answers, qa_type, agent)

    print(f"[DONE] {output_path} (n={len(gen_answers)})")
    print(f"\nTOTAL elapsed: {time.time() - start_time:.2f}s")


'''  
     ====== naive_Rag ======
     =======================
    ## agent(infer)
    qa_type = Hotpot_short(llm)  # QAtype 인스턴스 생성
    agent = RAG_agent(CFG, retriever, qa_type=qa_type)

    input_path = Path("db/qa_data/test/HOTPOT_[text]weather_manual.json")
    with open(input_path, "r", encoding="utf-8") as f:
        agent_inputs = json.load(f)

    gen_answers = agent.answer_all(agent_inputs)

    output_path = Path("results/inferenced/test/hotpotqa_test/hotpot_test.json")
    save_predictions_json(output_path, agent_inputs, gen_answers, qa_type, agent)

    print(f"[DONE] {output_path} (n={len(gen_answers)})")
    print(f"\nTOTAL elapsed: {time.time() - start_time:.2f}s")
'''