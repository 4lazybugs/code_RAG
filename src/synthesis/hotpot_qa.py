from pathlib import Path
from typing import Any, Dict, List
from src.prompts.qa_gen import hotpot_prompt
import json
from src.qa_gen.params import Params
from tqdm import tqdm
import random

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def yield_pair(dir_path: Path, *, min_len: int, seed: int | None = None):
    files = sorted(
        [p for p in dir_path.glob("**/*.md") if p.is_file()],
        key=lambda p: p.as_posix()
    )

    if seed is not None:
        random.seed(seed)     # 재현 가능하게

    random.shuffle(files)     # ⭐ 여기만 추가

    it = iter(files)

    # zip(it, it)는 같은 iterator를 연속으로 소모해 2개씩 묶으므로,
    # 각 파일은 next()로 한 번만 소비되어 다른 pair에 재사용되지 않는다.
    for a, b in zip(it, it):
        text_a, text_b = read_text(a), read_text(b)
        if len(text_a) < min_len or len(text_b) < min_len: continue
        yield a, b



def hotpot_gen(dir_path: Path, params: Params, prompt= hotpot_prompt) -> List[Dict[str, Any]]:
    llm = params.get_llm("llm")  # ✅ llm은 global_에서
    min_len = params.get_params("hotpot_short", "min_len", 300)
    start_id = int(params.get_params("hotpot_short", "start_id", 0))
    seed = int(params.get_params("hotpot_short", "seed", 42 ))

    chain = prompt | llm

    qas_list: List[Dict[str, Any]] = []
    cur_id = start_id

    # --- 디버그용 카운터 ---
    ok_cnt = 0
    json_fail_cnt = 0
    invoke_fail_cnt = 0

    # --- 진행바 적용 ---
    pairs = yield_pair(dir_path, min_len=min_len, seed=seed)
    pbar = tqdm(pairs, desc="hotpot_short", dynamic_ncols=True)

    for idx, (a, b) in enumerate(pbar, start=1):
        md_a = read_text(a)
        md_b = read_text(b)

        # 진행바 오른쪽 상태 표시
        pbar.set_postfix({
            "pair": idx,
            "id": cur_id,
            "ok": ok_cnt,
            "json_fail": json_fail_cnt,
            "invoke_fail": invoke_fail_cnt,
        })

        # 실시간 로그 (파일명)
        tqdm.write(f"[pair={idx} id={cur_id}] A={a.name} | B={b.name}")

        try:
            resp = chain.invoke({"md_a": md_a, "md_b": md_b, "id": cur_id})
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
                "ref_name_a": a.name,
                "ref_name_b": b.name,
                "supporting_fact_a": item["supporting_fact_a"],
                "supporting_fact_b": item["supporting_fact_b"],
            })

        cur_id += len(arr)  #✅ 1이 아니라 len(arr)

    return qas_list