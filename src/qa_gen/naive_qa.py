from pathlib import Path
from typing import Any, Dict, List
import json
from src.qa_gen.params import Params
from tqdm import tqdm
import random

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


def naive_gen(dir_path: Path, params: Params) -> List[Dict[str, Any]]:
    llm = params.get_llm("llm")  # ✅ llm은 global_에서
    min_len = params.get_params("naive", "min_len", 300)
    start_id = int(params.get_params("naive", "start_id", 0))
    seed = int(params.get_params("naive", "seed", 42 ))
    prompt = params.get_params("naive", "prompt")

    chain = prompt | llm

    qas_list: List[Dict[str, Any]] = []
    cur_id = start_id

    # --- 디버그용 카운터 ---
    ok_cnt = 0
    json_fail_cnt = 0
    invoke_fail_cnt = 0

    # --- 진행바 적용 ---
    mds = yield_single(dir_path, min_len=min_len, seed=seed)
    pbar = tqdm(mds, desc="single", dynamic_ncols=True)

    for idx, md in enumerate(pbar, start=1):
        md_text = read_text(md)

        # 진행바 오른쪽 상태 표시
        pbar.set_postfix({
            "pair": idx,
            "id": cur_id,
            "ok": ok_cnt,
            "json_fail": json_fail_cnt,
            "invoke_fail": invoke_fail_cnt,
        })

        #breakpoint()
        # 실시간 로그 (파일명)
        tqdm.write(f"id={cur_id}] refernce={md.name}")

        try:
            resp = chain.invoke({"md": md_text, "id": cur_id})
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            invoke_fail_cnt += 1
            tqdm.write(f"  ❌ invoke failed: {type(e).__name__}: {e}")
            cur_id += 1
            continue

        try:
            arr = json.loads(content)
        except Exception:
            json_fail_cnt += 1
            # raw 응답 일부만 출력(너무 길면 터미널 지옥됨)
            snippet = content[:300].replace("\n", "\\n")
            tqdm.write(f"  ❌ JSON parse failed. raw[:300]={snippet}")
            cur_id += 1
            continue

        # 성공
        ok_cnt += 1
        # question 미리보기(첫 항목만)
        if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
            q_preview = str(arr[0].get("question", ""))[:120].replace("\n", " ")
            tqdm.write(f"  ✅ parsed. question_preview={q_preview}...")

        for j, item in enumerate(arr):
            qas_list.append({
                "id": cur_id + j, #✅ j도입 이유: 여러 샘플을 한 번에 만들 경우 id 중복 발생 가능
                "question": item["question"],
                "answer": item["answer"],
                "ref_doc": md.name, 
            })

        cur_id += len(arr)  #✅ 1이 아니라 len(arr)

    return qas_list