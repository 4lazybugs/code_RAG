from pathlib import Path
from typing import Any, Dict, List
import json
from src.qa_gen.params import Params
from tqdm import tqdm
import random, re

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def yield_single(dir_path: Path, *, min_len: int, seed: int | None = None):
    
    # 입력이 단일 md 파일이면 그대로 사용
    if dir_path.is_file() and dir_path.suffix == ".md":
        files = [dir_path]
    else:
        # 디렉토리면 하위 모든 md 파일 재귀 탐색
        files = sorted(
            [p for p in dir_path.glob("**/*.md") if p.is_file()],
            key=lambda p: p.as_posix()  # 경로 기준 정렬
        )

    # 재현 가능하도록 시드 설정
    if seed is not None:
        random.seed(seed)

    # 파일 순서 랜덤화
    random.shuffle(files)

    # 최소 길이 이상인 파일만 yield
    for md in files:
        text = read_text(md)
        if len(text) < min_len:
            continue
        yield md

def clean_md(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)      # HTML 태그 제거
    text = re.sub(r'\n{3,}', '\n\n', text)   # 연속 빈줄 정리
    text = re.sub(r'[ \t]+', ' ', text)      # 연속 공백 정리
    return text.strip()

def gen_from_md(dir_path: Path, params: Params, mode: str = "qa") -> List[Dict[str, Any]]:
    llm = params.get_llm("llm")
    min_len = params.get_params("min_len", 300)
    start_id = int(0)
    seed = int(42)

    prompt_key = {"q": "prompt_q", "a": "prompt_a", "qa": "prompt_qa"}.get(mode, "prompt_qa")
    prompt = params.get_params(prompt_key)  # "naive" 없애고 바로 조회
    chain = prompt | llm

    results: List[Dict[str, Any]] = []
    cur_id = start_id
    ok_cnt = json_fail_cnt = invoke_fail_cnt = 0

    mds = yield_single(dir_path, min_len=min_len, seed=seed)
    pbar = tqdm(mds, desc=mode, dynamic_ncols=True)

    for idx, md in enumerate(pbar, start=1):
        md_text = read_text(md)

        pbar.set_postfix({
            "pair": idx,
            "id": cur_id,
            "ok": ok_cnt,
            "json_fail": json_fail_cnt,
            "invoke_fail": invoke_fail_cnt,
        })
        tqdm.write(f"id={cur_id}] reference={md.name}")

        try:
            resp = chain.invoke({"md": md_text, "id": cur_id})
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            invoke_fail_cnt += 1
            tqdm.write(f"❌ invoke failed: {type(e).__name__}: {e}")
            cur_id += 1
            continue

        try:
            arr = json.loads(content)
            if not isinstance(arr, list):
                raise ValueError("Response is not a list")
        except Exception:
            json_fail_cnt += 1
            snippet = content[:300].replace("\n", "\\n")
            tqdm.write(f"❌ JSON parse failed raw[:300]={snippet}")
            cur_id += 1
            continue

        ok_cnt += 1
        if arr and isinstance(arr[0], dict):
            preview = str(arr[0].get("question", arr[0].get("answer", "")))[:120].replace("\n", " ")
            tqdm.write(f"✅ parsed preview={preview}")

        for j, item in enumerate(arr):
            record = {
                "id": cur_id + j,
                "ref_doc": md.name,
                "ref_content": clean_md(md_text),  
                "retrival_needed": 1,
            }

            if mode in ("q", "qa"):
                record["question"] = item["question"]
            if mode in ("a", "qa"):
                record["answer"] = item["answer"]

            results.append(record)

        cur_id += len(arr)

    return results