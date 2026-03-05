from __future__ import annotations

import json
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from src.qa_gen.params import Params
from src.qa_gen.hotpot_qa import hotpot_gen
from src.qa_gen.naive_qa import naive_gen
from src.prompts.qa_gen import naive_prompt, consulting_prompt, hotpot_prompt

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

    llm = ChatOpenAI(
        model= "gpt-4o-mini", # 빠르고 싼 가성비 모델
        temperature=0,
    )
    

    # ===== naive_qa =========================================================
    # ✅ llm은 global_에만
    params = Params(
        global_={"llm": llm},
        per={
            "naive": {
                "min_len": 300,
                "start_id": 0,
                "seed": 42,
                "prompt": consulting_prompt,
            }
        }
    )

    # ⚠️ .md들이 들어있는 폴더
    extracted_dir = Path("db/raw_db_extracted/consulting")

    out_root = Path("db/qa_data/qa_in_use/consulting")
    out_root.mkdir(parents=True, exist_ok=True)

    md_dirs = list(look4md(extracted_dir))
    print(f"found md-dirs: {len(md_dirs)}")

    for search_dir in md_dirs:
        # output 파일명: NAIVE_[text]plant_disease_manual.json (pdf 확장자 제거)
        out_name = f"{search_dir.stem}.json"
        output_path = out_root / out_name

        # 기존 스타일 유지: partial로 generate_qa 생성
        generate_qa = partial(naive_gen, search_dir, params)
        result = generate_qa()

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"saved: {output_path} (n={len(result)})")

    
    #===== naive ======================================================================
    # ✅ llm은 global_에만 둔다
    params = Params(
        global_={"llm": llm},
        per={
            "naive": {
                "min_len": 300,
                "start_id": 0,
                "seed": 42,
                "prompt": naive_prompt,
            }
        }
    )

    # ⚠️ .md들이 들어있는 폴더
    extracted_dir = Path("db/raw_db_extracted/manual")

    out_root = Path("db/qa_data/qa_in_use/manual")
    out_root.mkdir(parents=True, exist_ok=True)

    md_dirs = list(look4md(extracted_dir))
    print(f"found md-dirs: {len(md_dirs)}")

    for search_dir in md_dirs:
        # output 파일명: NAIVE_[text]plant_disease_manual.json (pdf 확장자 제거)
        out_name = f"{search_dir.stem}.json"
        output_path = out_root / out_name

        # 기존 스타일 유지: partial로 generate_qa 생성
        generate_qa = partial(naive_gen, search_dir, params)
        result = generate_qa()

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"saved: {output_path} (n={len(result)})")