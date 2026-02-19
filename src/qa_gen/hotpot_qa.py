from pathlib import Path
from itertools import combinations
from typing import Any, Dict, List, Iterator
import json
from src.qa_gen.params import Params
from langchain_core.prompts import ChatPromptTemplate
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

    for a, b in zip(it, it):
        text_a, text_b = read_text(a), read_text(b)
        if len(text_a) < min_len or len(text_b) < min_len:
            continue
        yield a, b

hotpot_prompt = ChatPromptTemplate.from_template(
"""
다음 context에서 HotpotQA 스타일 multi-hop Ground Truth 1개를 생성하라.

[핵심 규칙]
1) 반드시 서로 다른 두 문단(A, B)을 사용해야 하며,
   질문은 A와 B를 모두 읽어야만 답할 수 있어야 한다.

2) link_word는 A와 B에 동시에 등장하는 동일 표기의 단어/구(bridge entity)여야 한다.
   두 문단에 공통 등장하지 않으면 실패다.

3) supporting_fact_a:
   - 문단 A에서 답에 필요한 핵심 근거 1개를 원문 그대로 발췌
   - 문맥상 단독 이해 가능할 만큼 충분히 길 것 (약 200 tokens 이상 권장)

4) supporting_fact_b:
   - 문단 B에서 답에 필요한 핵심 근거 1개를 원문 그대로 발췌
   - 문단 A 내용 포함 금지

5) 질문 규칙:
   - 고유 식별자 포함
   - 지시어/모호한 일반명사 금지
   - 답은 context에 명시된 단일 값

[출력 형식]
설명 없이 JSON 배열 1개만 출력:

[
  {{
    "id": {id},
    "link_word": "...",
    "question": "...",
    "answer": "...",
    "supporting_fact_a": "...",
    "supporting_fact_b": "..."
  }}
]

# context
{md_a}
{md_b}
"""
)

def hotpot(dir_path: Path, params: Params) -> List[Dict[str, Any]]:
    llm = params.get_llm("llm")  # ✅ llm은 global_에서
    min_len = params.get_params("hotpot", "min_len", 300)
    start_id = int(params.get_params("hotpot", "start_id", 0))

    chain = hotpot_prompt | llm

    qas_list: List[Dict[str, Any]] = []
    cur_id = start_id

    # --- 디버그용 카운터 ---
    ok_cnt = 0
    json_fail_cnt = 0
    invoke_fail_cnt = 0

    # --- 진행바 적용 ---
    pairs = yield_pair(dir_path, min_len=min_len)
    pbar = tqdm(pairs, desc="hotpot", dynamic_ncols=True)

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
                "link_word": item["link_word"],
                "question": item["question"],
                "answer": item["answer"],
                "supporting_fact_a": item["supporting_fact_a"],
                "supporting_fact_b": item["supporting_fact_b"],
            })

        cur_id += len(arr)  #✅ 1이 아니라 len(arr)

    return qas_list