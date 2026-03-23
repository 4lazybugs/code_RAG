from __future__ import annotations

import json
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from src.synthesis.params import Params
from src.synthesis.hotpot_qa import hotpot_gen
from src.synthesis.one_md_qa import gen_from_md
from src.synthesis.agri_qa import qa_from_prompt
from src.prompts.qa_gen import agri_gpt_prompt, naive_prompt, consulting_prompt, hotpot_prompt
from src.prompts.ares_gen import q_gen_prompt, ans_gen_prompt

def look4md(root: Path):
    """
    1) '*.pdf' 이름의 디렉토리가 존재하면 → 해당 디렉토리 yield
    2) 없으면 → root에서 '*.md' 파일 yield
    """

    md_dirs = [p for p in root.rglob("*.pdf") if p.is_dir()]

    if md_dirs:
        for p in md_dirs:
            yield p
    else:
        for md in root.rglob("*.md"):
            yield md

if __name__ == "__main__":
    load_dotenv()

    '''
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-32B-Instruct",
        temperature=0,
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )
    '''

    '''
    llm_naive = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
    )
    '''

    llm_md = ChatOpenAI(
        model= "gpt-4o-mini", # 빠르고 싼 가성비 모델
        temperature=0,
    )

    # ===== naive_qa =========================================================
    # ✅ llm은 global_에만
    params_md = Params(
        global_={"llm": llm_md},
        per={
            "min_len": 300,
            "start_id": 0,
            "seed": 42,
            "prompt_q": q_gen_prompt,
        }
    )

    # md들이 들어있는 폴더
    extracted_dir = Path("db/raw_db_extracted/test")

    out_root = Path("db/qa_data/test/q")
    out_root.mkdir(parents=True, exist_ok=True)

    md_dirs = list(look4md(extracted_dir))
    print(f"found md-dirs: {len(md_dirs)}")

    for search_dir in md_dirs:
        # output 파일명: NAIVE_[text]plant_disease_manual.json (pdf 확장자 제거)
        out_name = f"{search_dir.stem}.json"
        output_path = out_root / out_name

        # 기존 스타일 유지: partial로 generate_qa 생성
        generate_qa = partial(gen_from_md, search_dir, params_md, "q")
        result = generate_qa()

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"saved: {output_path} (n={len(result)})")
    






    '''
    out_root = Path("db/qa_data/qa_in_use/without_md")
    out_root.mkdir(parents=True, exist_ok=True)

    for i in range(1100):
        out_name = f"agri_QA_only_via_GPT_{i}.json"
        output_path = out_root / out_name

        result = qa_from_prompt(params_naive, num_samples=1)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"saved: {output_path} (n={len(result)})")
    '''