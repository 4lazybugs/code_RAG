# import packages in std
import time
import json
import copy
import argparse, os, yaml
from types import SimpleNamespace
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from minicheck.minicheck import MiniCheck

from src.config import get_config, load_yaml

# agents
from src.infer.agents import LLM_agent, RAG_agent, Gateway_agent, IterRAG_agent

# retrievals
from src.retrieval import build_retrievers, Multi_Retriever, build_bm25s, Multi_BM25s, Embeddor

# prompts
from src.prompts.qa_type import saq_rag_prompt, saq_llm_prompt, iter_rag_prompt, logprob_prompt

from src.infer.qa_type.base import QAtype
# qa_input
from src.infer.qa_type.load_input import mcq_input, saq_input
# qa_output
from src.infer.qa_type.load_output import llm_output, rag_output, iter_output
from src.infer.qa_type.load_output import router_output, gate_llm_output, gate_rag_output, gate_sota_output


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

    for i, q_id_dict in enumerate(items):
        q_id_dict["id"] = i

    if save:
        output_path = json_dir / "merged.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    return items


if __name__ == "__main__":
    start_time = time.time()
    load_dotenv()
    CFG_emb = get_config("configs/config_emb.yaml")
    CFG_infer = get_config("configs/config_infer.yaml")
    CFG_pth = get_config("configs/config_path.yaml")

    vec_root = Path("db/vector_db")
    ## retrievers    
    emb = Embeddor(CFG_emb.embedor_model_name)
    retriever_list = build_retrievers(vec_root=vec_root, emb=emb)
    multi_retriever = Multi_Retriever(retrievers=retriever_list, k_each=5, top_k=2)
    bm25_list = build_bm25s(vec_root=vec_root, k_each=4)
    lex_retriever = Multi_BM25s(retrievers=bm25_list, top_k=5)

    # load QA
    qa_dir = Path(CFG_pth.qa_dir) # json이 들어있는 디렉터리
    json_merged = merge_json(qa_dir, save=False)
    for i, q_id_dict in enumerate(json_merged):
        q_id_dict["id"] = i

    ## load LM(Language Model)
    qwen = ChatOpenAI(
         model=CFG_infer.model_name,
         temperature=0,
         base_url="http://127.0.0.1:8000/v1",
         api_key="EMPTY",
    )
    gpt = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
    )
    # MiniCheck는 주어진 문서(context)가 특정 문장(claim 또는 answer)을
    # 실제로 근거로 뒷받침하는지를 판단하는 LM 검증 모델
    judge_lm = MiniCheck(
            model_name=CFG_infer.judge_model,
            cache_dir=CFG_infer.judge_cache_dir,
    )

    '''
    <Builder Pattern>
    조립은 외부에서 수행 — prompt/input/output의 다양한 조합을 지원하기 위해
    조합마다 클래스를 따로 만들면 N^3 조합이 생겨 비효율적이므로 빌더 패턴 사용
    '''
    qa_router = (QAtype()
        .set_prompt(logprob_prompt)
        .set_inputs(saq_input)
        .set_outputs(router_output)
        .build())

    qa_llm = (QAtype()
        .set_prompt(saq_llm_prompt)
        .set_inputs(saq_input)
        .set_outputs(gate_llm_output)
        .build())

    qa_rag = (QAtype()
        .set_prompt(saq_rag_prompt)
        .set_inputs(saq_input)
        .set_outputs(gate_rag_output)
        .build())

    qa_sota = (QAtype()
        .set_prompt(saq_llm_prompt)
        .set_inputs(saq_input)
        .set_outputs(gate_sota_output)
        .build())

    '''
    <Strategy Pattern>
    agent는 고정한 채로, 주입하는 qa_type만 바꿔서
    agent가 llm_mcq(전략1) 또는 rag_saq(전략2)로 동작하게 만들 수 있음.
    '''
    router_agent = LLM_agent(llm=qwen, qa_type=qa_router)
    #qa_llm.set_outputs(llm_output) 
    llm_agent = LLM_agent(llm=qwen, qa_type=qa_llm)
    qa_rag.set_outputs(rag_output)
    rag_agent = RAG_agent(llm=qwen, qa_type=qa_rag, retriever=multi_retriever)
    #qa_sota.set_outputs(llm_output) 
    sota_agent = LLM_agent(llm=gpt, qa_type=qa_sota) 
    gateway_agent = Gateway_agent(
                Router_agent=router_agent,
                LLM_agent=llm_agent,
                RAG_agent=rag_agent,
                SOTA_agent = sota_agent,
                judge_lm=judge_lm,
                know_thres=CFG_infer.know_thres,
                relv_thre=CFG_infer.relv_thre,
                faith_thre=CFG_infer.faith_thre   
    )

    #results = llm_agent.answer_all(json_merged)
    results = rag_agent.answer_all(json_merged)
    #results = sota_agent.answer_all(json_merged)
    #results = gateway_agent.answer_all(json_merged)


    output_path = Path(CFG_pth.infered_fpth)
    save_predictions_json(output_path, json_merged, results)

    print(f"[DONE] {output_path} (n={len(results)})")
    print(f"\nTOTAL elapsed: {time.time() - start_time:.2f}s")