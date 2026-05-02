from __future__ import annotations

import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from itertools import groupby

from src.config import get_config

from src.synthesis.params import Params
from src.synthesis.make_qa import gen_from_chunks
from src.synthesis.qa2d import qa2d_batch
from src.synthesis.nli_filter import NLIFilter

from src.prompts import qa2d_prompt, qa_gen_prompt

def run_qa_generation(chunk_root: Path, params: Params) -> list[dict]:
    """1단계: chunk json → QA 생성"""
    results = gen_from_chunks(chunk_root, params, max_qa=MAX_QA)
    print(f"[INFO] QA 생성 완료: {len(results)}개")
    return results


def run_qa2d(chunk_results: list[dict], chain) -> list[dict]:
    """2단계: Q+A → Declarative sentence 변환"""
    print("[INFO] QA2D 변환 중...")
    results = []
    for i, item in enumerate(chunk_results):
        converted = qa2d_batch([item], chain)[0]
        results.append(converted)
        print(f"  [{i+1}/{len(chunk_results)}] {item['source_file']}")
        print(f"    Q: {item['question']}")
        print(f"    A: {item['answer']}")
        print(f"    D: {converted['declarative']}")
        print(f"  {'-'*50}")
    return results


def run_nli_filter(qa2d_results: list[dict], nli: NLIFilter, threshold: float) -> tuple[list[dict], list[dict]]:
    """3단계: NLI 기반 entailment 판단 → (전체, 필터링된 것) 반환"""
    print(f"[INFO] NLI 필터링 중... (threshold={threshold})")
    all_results, _ = nli.filter_batch(qa2d_results)
    filtered = [
        r for r in all_results
        if r["label"] == "entailment" and r["score"] >= threshold
    ]
    print(f"[INFO] 필터링 결과: {len(filtered)}/{len(all_results)} "
          f"({len(filtered)/max(len(all_results), 1)*100:.1f}%)")
    return all_results, filtered

def save_qa(qa_list: list[dict], chunk_root: Path, out_root: Path, suffix: str) -> None:
    chunk_files = list(chunk_root.rglob("*_chunk*.json"))

    keyfunc = lambda x: x["source_file"]
    for source_file, group in groupby(sorted(qa_list, key=keyfunc), key=keyfunc):
        items = list(group)

        matched = next((f for f in chunk_files if f.name.startswith(Path(source_file).stem.rsplit("_", 1)[0])), None)
        rel_dir  = matched.parent.relative_to(chunk_root) if matched else Path(".")
        save_dir = out_root / suffix / rel_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(source_file).stem
        for qa_idx, item in enumerate(items):   # ← source_file마다 0부터
            filename = f"{stem}_qa{qa_idx+1:03d}.json"
            with open(save_dir / filename, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)


def process_chunk_root(chunk_root: Path, params: Params, qa2d_chain, nli: NLIFilter) -> None:
    print(f"\n[INFO] chunk_root: {chunk_root}")

    chunk_results         = run_qa_generation(chunk_root, params)
    qa2d_results          = run_qa2d(chunk_results, qa2d_chain)
    all_results, filtered = run_nli_filter(qa2d_results, nli, ENTAIL_THRESHOLD)

    save_qa(all_results, chunk_root, OUT_ROOT, suffix="all")      # chunk_root 추가
    print(f"[INFO] 전체 QA {len(all_results)}개 저장 완료 (all)")

    save_qa(filtered, chunk_root, OUT_ROOT, suffix="filtered")    # chunk_root 추가
    print(f"[INFO] 필터링 QA {len(filtered)}개 저장 완료 (filtered)")

    print(f"[INFO] {chunk_root.name} — 완료")

############### load params #######################
CFG = get_config("configs/config_preproc.yaml")

CHUNK_ROOTS      = [Path(d) for d in CFG.chunk_in_dirs]
OUT_ROOT         = Path(CFG.qa_out_dir)
MAX_QA           = getattr(CFG, "max_fnum", 100000)
NLI_MODEL        = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
ENTAIL_THRESHOLD = 0.9
###################################################

if __name__ == "__main__":
    load_dotenv()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    params_chunk = Params(
        global_={"llm": llm},
        per={"prompt_qa": qa_gen_prompt},
    )
    qa2d_chain = qa2d_prompt | llm
    nli        = NLIFilter(model_name=NLI_MODEL, device="cuda")

    for chunk_root in CHUNK_ROOTS:
        process_chunk_root(chunk_root, params_chunk, qa2d_chain, nli)

    print(f"\n[INFO] 전체 완료 → {OUT_ROOT}")