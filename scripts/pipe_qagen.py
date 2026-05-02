from __future__ import annotations

import json
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from itertools import groupby

from src.config import get_config
from src.synthesis.params import Params
from src.synthesis.one_md_qa import gen_from_md
from src.synthesis.chunk_qa import gen_from_chunks
from src.synthesis.agri_qa import qa_from_prompt

from src.prompts.synthesis_prompt import agri_gpt_prompt, consulting_prompt, hotpot_prompt
from src.prompts import naive_prompt, qa_gen_prompt


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



############### load params #######################
CFG = get_config("configs/config_preproc.yaml")

CHUNK_ROOTS = [Path(d) for d in CFG.chunk_in_dirs]   # yaml: chunk_in_dirs
OUT_ROOT    = Path(CFG.qa_out_dir)                    # yaml: qa_out_dir
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_QA = getattr(CFG, "max_fnum", 100000)  # yaml: max_fnum (없으면 기본값)                  # yaml: max_fnum (없으면 기본값)
###################################################

if __name__ == "__main__":
    load_dotenv()

    llm_chunk = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    params_chunk = Params(
        global_={"llm": llm_chunk},
        per={"prompt_qa": qa_gen_prompt},
    )

    for chunk_root in CHUNK_ROOTS:
        print(f"\n[INFO] chunk_root: {chunk_root}")

        chunk_results = gen_from_chunks(chunk_root, params_chunk, max_qa=MAX_QA)

        # source_file 기준으로 그룹핑
        keyfunc = lambda x: x["source_file"]
        for source_file_str, group in groupby(
            sorted(chunk_results, key=keyfunc), key=keyfunc
        ):
            qa_list = list(group)

            # source_file 이름 기반으로 저장
            save_dir = OUT_ROOT
            save_dir.mkdir(parents=True, exist_ok=True)

            qa_filename = Path(source_file_str).stem + "_qa.json"
            with open(save_dir / qa_filename, "w", encoding="utf-8") as f:
                json.dump(qa_list, f, ensure_ascii=False, indent=2)

        print(f"[INFO] {chunk_root.name} — QA {len(chunk_results)}개 저장 완료")

    print(f"\n[INFO] 전체 완료 → {OUT_ROOT}")