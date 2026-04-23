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
chunk_root = Path("db/chunks_for_synthesis")
out_root = Path("db/qa_data/cand_qa_from_agentic_chunks")
out_root.mkdir(parents=True, exist_ok=True)

MAX_QA = 100000
##################################################    


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
            #"prompt_q": naive_prompt,
            "prompt_qa": naive_prompt,
        }
    )

    # md들이 들어있는 폴더
    extracted_dir = Path("db/md_for_synthesis")

    out_root = Path("db/qa_data/cand_book")
    out_root.mkdir(parents=True, exist_ok=True)

    md_dirs = list(look4md(extracted_dir))
    print(f"found md-dirs: {len(md_dirs)}")

    MAX_QA = 10000
    remaining = MAX_QA

    for search_dir in md_dirs:
        out_name = f"{search_dir.stem}.json"
        output_path = out_root / out_name

        generate_qa = partial(gen_from_md, search_dir, params_md, "qa", remaining)
        result = generate_qa()

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"saved: {output_path} (n={len(result)})")

        remaining -= len(result)
        if remaining <= 0:
            break
    '''
    
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