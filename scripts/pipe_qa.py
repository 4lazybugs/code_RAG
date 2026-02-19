from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from functools import partial
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from src.qa_gen.params import Params
from src.qa_gen.hotpot_qa import hotpot

if __name__ == "__main__":
    load_dotenv()

    # llm = ChatOpenAI(
    #     model="Qwen/Qwen2.5-32B-Instruct",
    #     temperature=0,
    #     base_url="http://127.0.0.1:8000/v1",
    #     api_key="EMPTY",
    # )

    llm = ChatOpenAI(
        model= "gpt-4o-mini", # 빠르고 싼 가성비 모델
        temperature=0,
    )

    # ⚠️ .md들이 들어있는 폴더
    search_dir = Path("db/raw_db_extracted/manual_book/[text]plant_disease_manual.pdf")

    output_path = Path("db/qa_data/test/HOTPOT_[text]plant_disease_manual.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ llm은 global_에만 둔다
    params = Params(
        global_={"llm": llm},
        per={
            "hotpot": {
                "min_len": 300,
                "start_id": 0,
            }
        }
    )

    generate_qa = partial(hotpot, search_dir, params)
    result = generate_qa()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"saved: {output_path} (n={len(result)})")