from __future__ import annotations

import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from itertools import groupby

from src.config import get_config

from src.synthesis.params import Params
from src.synthesis.make_qa import gen_from_chunks

from src.postprocess.qa2d import qa2d_batch
from src.postprocess.nli_filter import NLIFilter

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
        print(f"    Q: {item['question']}")
        print(f"    A: {item['answer']}")
        print(f"    D: {converted['declarative']}")
        print(f"  {'-'*50}")
    return results

def save_qa(qa_list: list[dict], chunk_root: Path, out_root: Path, suffix: str) -> None:
    keyfunc = lambda x: x["chunk_name"]
    for chunk_file_str, group in groupby(sorted(qa_list, key=keyfunc), key=keyfunc):
        items      = list(group)
        chunk_file = Path(chunk_file_str)
        rel_dir    = chunk_file.parent.relative_to(chunk_root)
        save_dir   = out_root / suffix / rel_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        for qa_idx, item in enumerate(items):
            filename = f"{chunk_file.stem}_qa{qa_idx+1:03d}.json"
            with open(save_dir / filename, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)

def process_chunk_root(chunk_root: Path, params: Params, qa2d_chain) -> None:
    print(f"\n[INFO] chunk_root: {chunk_root}")

    chunk_results         = run_qa_generation(chunk_root, params)
    qa2d_results          = run_qa2d(chunk_results, qa2d_chain)

    save_qa(qa2d_results, chunk_root, OUT_ROOT, suffix="all")

    print(f"[INFO] {chunk_root.name} — 완료")

############### load params #######################
CFG = get_config("configs/config_gen.yaml")

CHUNK_ROOTS      = [Path(d) for d in CFG.chunk_in_dirs]
OUT_ROOT         = Path(CFG.qa_out_dir)
MAX_QA           = getattr(CFG, "max_fnum", 100000)
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

    for chunk_root in CHUNK_ROOTS:
        process_chunk_root(chunk_root, params_chunk, qa2d_chain)

    print(f"\n[INFO] 전체 완료 → {OUT_ROOT}")