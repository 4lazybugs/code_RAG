from __future__ import annotations

import json
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from src.synthesis.params import Params
#from src.synthesis.hotpot_qa import hotpot_gen
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


def _save_qa(chunk_file: Path, chunk_root: Path, output_dir: Path, qa_list: list):
    # chunks_for_synthesis/fruit_diagnosis_manual_10/fruit_diagnosis_manual_10_chunk00.json
    # → qa_for_synthesis/fruit_diagnosis_manual_10/fruit_diagnosis_manual_10_qa_chunk00.json
    rel_dir = chunk_file.parent.relative_to(chunk_root)
    save_dir = output_dir / rel_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    qa_filename = chunk_file.name.replace("_chunk", "_qa_chunk")
    with open(save_dir / qa_filename, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)


############### load params #######################

CFG = get_config("configs/config_preproc.yaml")

MD_DIRS    = [Path(d) for d in CFG.md_dirs]
CHUNK_DIR  = Path(CFG.chunk_dir)

chunk_root = Path("db/chunks_for_synthesis")
out_root = Path("db/qa_data/cand_qa_from_agentic_chunks")
out_root.mkdir(parents=True, exist_ok=True)

MAX_QA = 100000
##################################################    




if __name__ == "__main__":
    load_dotenv()

    #================ chunk_qa =========================================================
    llm_chunk = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    params_chunk = Params(
        global_={"llm": llm_chunk},
        per={
            "prompt_qa": qa_gen_prompt,
        }
    )

    chunk_results = gen_from_chunks(chunk_root, params_chunk, max_qa=MAX_QA)

    # chunk_file 기준으로 그룹핑해서 저장
    from itertools import groupby
    keyfunc = lambda x: x["chunk_file"]
    for chunk_file_str, group in groupby(sorted(chunk_results, key=keyfunc), key=keyfunc):
        qa_list = list(group)
        _save_qa(Path(chunk_file_str), chunk_root, out_root, qa_list)

    print(f"total qa: {len(chunk_results)}")
