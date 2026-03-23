# import packages in std
import time
import json
import argparse, os, yaml
from types import SimpleNamespace
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from src.config import get_config, load_yaml

# agents
from src.infer.agents import LLM_agent, RAG_agent, IterRAG_agent

#retrievals
from src.retrieval import build_retrievers, Multi_Retriever, build_bm25s, Multi_BM25s, Embeddor

# prompts
from src.infer.qa_type.prompts import saq_llm_prompt, saq_rag_prompt
from src.infer.qa_type.prompts import mcq_llm_prompt, mcq_rag_prompt
from src.infer.qa_type.prompts import iter_rag_prompt

from src.infer.qa_type.base import QAtype
# qa_input
from src.infer.qa_type.load_input import mcq_input, saq_input
# qa_output
from src.infer.qa_type.load_output import llm_output, rag_output, iter_output


def save_predictions_json(out_path: Path, inputs, results):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for _input, result in zip(inputs, results):
        # result가 딕셔너리가 아니면 (NaiveLLM처럼 string 반환) 감싸기
        if not isinstance(result, dict):
            result = {"generated": result}

        row = {}
        # _input에서 먼저 채우고
        for key in ["id", "question", "answer"]:
            val = result.get(key) or _input.get(key)
            if val is not None:
                row[key] = val

        # result에 있는 것 다 넣기
        for key, val in result.items():
            if key not in row and val is not None:
                row[key] = val

        rows.append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

## qa json 하나로 합치고 저장하는 함수
def merge_json(json_dir: str | Path, save: bool = False) -> list[dict]:
    json_dir = Path(json_dir)
    items = []
    for json_file in sorted(json_dir.rglob("*.json")):
        data = json.load(json_file.open(encoding="utf-8"))
        for item in (data if isinstance(data, list) else [data]):
            if isinstance(item, dict) and "question" in item:
                items.append(item)

    for i, _dict in enumerate(items):
        _dict["id"] = i

    if save:
        output_path = json_dir / "merged.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    return items


if __name__ == "__main__":
    start_time = time.time()
    load_dotenv()
    CFG = get_config("configs/config_infer.yaml")
    emb = Embeddor(CFG.embedor_model_name)
    llm_model = CFG.model_name

    ## retrievers
    vec_root = Path("/home/jwkim/[code]생성형과제_server/db/vector_db")
    retriever_list = build_retrievers(vec_root=vec_root, emb=emb)
    retriever = Multi_Retriever(retrievers=retriever_list, k_each=3, top_k=5)
    retriever_list = build_bm25s(vec_root=vec_root, k_each=4)
    lex_retriever = Multi_BM25s(retrievers=retriever_list, top_k=5)

    ## llm
    llm = ChatOpenAI(
         model=llm_model,
         temperature=0,
         base_url="http://127.0.0.1:8000/v1",
         api_key="EMPTY",
    )
    
    qa_dir = Path("db/qa_data/test_saq") # json이 들어있는 디렉터리
    json_merged = merge_json(qa_dir, save=True)
    for i, _dict in enumerate(json_merged):
        _dict["id"] = i

    '''
    <Builder Pattern>
    조립은 외부에서 수행 — prompt/input/output의 다양한 조합을 지원하기 위해
    조합마다 클래스를 따로 만들면 N^3 조합이 생겨 비효율적이므로 빌더 패턴 사용
    '''
    qa_type_llm = (QAtype()
        .set_prompt(mcq_llm_prompt)
        .set_inputs(mcq_input)
        .set_outputs(llm_output)
        .build())

    qa_type_rag = (QAtype()
        .set_prompt(saq_rag_prompt)
        .set_inputs(saq_input)
        .set_outputs(rag_output)
        .build())

    qa_type_iter = (QAtype()
        .set_prompt(iter_rag_prompt)
        .set_outputs(iter_output)
        .build())

    '''
    <Strategy Pattern>
    agent는 고정한 채로, 주입하는 qa_type만 바꿔서
    agent가 llm_mcq(전략1) 또는 rag_saq(전략2)로 동작하게 만들 수 있음.
    '''
    llm_agent = LLM_agent(llm=llm, qa_type=qa_type_llm)
    rag_agent = RAG_agent(llm=llm, qa_type=qa_type_rag, retriever=lex_retriever) 
    iter_agent = IterRAG_agent(rag_agent=rag_agent, llm=llm, qa_type=qa_type_iter, max_iter=3)       
    
    #results = llm_agent.answer_all(json_merged)
    results = rag_agent.answer_all(json_merged)
    #results = iter_agent.answer_all(json_merged)

    output_path = Path("results/inferenced/qa_in_used.json")
    save_predictions_json(output_path, json_merged, results)

    print(f"[DONE] {output_path} (n={len(results)})")
    print(f"\nTOTAL elapsed: {time.time() - start_time:.2f}s")