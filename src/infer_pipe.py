import time
import json
import argparse, os, yaml
from types import SimpleNamespace

from inference import NaiveLLM, build_agent
from inference.qa_mode import build_qa_mode
from inference.retriever import load_retrievers, MultiCosineRetriever

from pathlib import Path
from copy import deepcopy
from dotenv import load_dotenv

def load_yaml(path='src/inference/config_infer.yaml'):
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

def get_config():
    default_cfg = load_yaml()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=default_cfg.get('model_name'))
    parser.add_argument("--qa_mode", type=str, default=default_cfg.get('qa_mode', 'MCQ'))

    args, _ = parser.parse_known_args()

    # ✅ YAML + argparse 병합 → Namespace
    cfg = {**default_cfg, **vars(args)}
    return SimpleNamespace(**cfg)


def save_predictions_json(out_path: Path, payloads, results, qa_mode, model):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # NaiveRag는 (id, question, answer, retrieved) 튜플 반환
    # NaiveLLM은 answer 문자열만 반환
    is_not_rag = isinstance(model, NaiveLLM)
    
    if is_not_rag: # NaiveLLM: results는 list[str] (answer만)
        rows = [
            {
                **payload,
                "gen_answer": gen_ans,
            }
            for payload, gen_ans in zip(payloads, results)
        ]
    else: # Rag: results는 List[Tuple[str, str, str, str]] (id, question, answer, retrieved)
        rows = [
            {
                **payload,
                "id": _id,
                "question": question,
                "gen_answer": gen_ans,
                **qa_mode.build_output(retrieved),   # ✅ 여기서 저장용 output 생성
            }
            for payload, (_id, question, gen_ans, retrieved) in zip(payloads, results)
        ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def run_once(cfg):
    qa_type = build_qa_mode(cfg.qa_mode)

    # 기본 인자
    kwargs = dict(
        cfg=cfg,
        qa_mode=qa_type,
    )

    # ✅ RAG 계열일 때만 retriever를 로드/생성해서 주입
    # (rag_type 문자열은 본인 레지스트리 키에 맞춰 조정)
    if cfg.rag_type != "naive_llm":
        single = load_retrievers(ndocs=cfg.k_each)
        kwargs["retriever"] = MultiCosineRetriever(
            retrievers=single,
            k_each=cfg.k_each,
            top_k=cfg.top_k,
        )

    model = build_agent(cfg.rag_type, **kwargs)

    json_path = Path(cfg.data_path)
    with open(json_path, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    results = model.answer_all(payloads)
    save_predictions_json(Path(cfg.output_path), payloads, results, qa_type, model)

    print(f"[DONE] {cfg.qa_mode} / {cfg.rag_type} -> {cfg.output_path} (n={len(results)})")


if __name__ == "__main__":
    load_dotenv()
    CFG = get_config()
    start_time = time.time()

    runs = getattr(CFG, "runs", None)

    if not runs:
        # runs가 없으면 단일 실행(기존 호환)
        run_once(CFG)
    else:
        for r in runs:
            # base CFG 복사 + run별 override
            cfg_d = deepcopy(vars(CFG))
            cfg_d.update(r)  # rag_type, qa_mode, data_path, output_path 등 덮어쓰기
            cfg_run = SimpleNamespace(**cfg_d)

            print(f"\n[RUN] {cfg_run.name} | qa={cfg_run.qa_mode} rag={cfg_run.rag_type}")
            run_once(cfg_run)

print(f"\nTOTAL elapsed: {time.time() - start_time:.2f}s")