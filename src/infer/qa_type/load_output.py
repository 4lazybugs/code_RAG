from typing import Dict, Any
import json

def llm_output(input_dic, gen_ans) -> Dict[str, Any]:
    #breakpoint()
    return {
        "id": input_dic["id"],
        "question": input_dic["question"],
        "generated": gen_ans,
        "answer": input_dic["answer"],
        "ref_doc": input_dic.get("ref_doc", ""),
    }


def rag_output(input_dic, gen_ans, retrieved) -> Dict[str, Any]:
    #breakpoint()
    return {
        "id": input_dic["id"],
        "question": input_dic["question"],
        "generated": gen_ans,
        "answer": input_dic["answer"],
        "ref_doc": input_dic.get("ref_doc", ""),
        "retrieved": [
            {
                "rel_path": doc.metadata.get("rel_path"),
                "rank": doc.metadata.get("__rank__"),
                "content": [line.strip() for line in doc.page_content.split("\n") if line.strip()],
            }
            for doc in retrieved
        ],
    }

def router_output(input_dic, gen_ans) -> Dict[str, Any]:
    #breakpoint()
    return {
        "id": input_dic["id"],
        "question": input_dic["question"],
        "generated": None,
        "answer": input_dic["answer"],
        "agent": "naive_llm",
        "judge_score": [
            {
                "know_prob": gen_ans,
                "relv_score": None,
                "faith_score": None,
            }
        ],
        "ref_doc": input_dic.get("ref_doc", "")
    }

def gate_llm_output(input_dic, gen_ans) -> Dict[str, Any]:
    #breakpoint()
    return {
        "id": input_dic["id"],
        "question": input_dic["question"],
        "generated": gen_ans,
        "answer": input_dic["answer"],
        "agent": "naive_llm",
        "judge_score": [
            {
                "know_prob": input_dic["judge_score"][0]["know_prob"],
                "relv_score": None,
                "faith_score": None,
            }
        ],
        "ref_doc": input_dic.get("ref_doc", "")
    }


def gate_rag_output(input_dic, gen_ans, retrieved) -> Dict[str, Any]:
    #breakpoint()
    return {
        "id": input_dic["id"],
        "question": input_dic["question"],
        "generated": gen_ans,
        "answer": input_dic["answer"],
        "agent": "rag",
        "judge_score": [
            {
                "know_prob": input_dic["judge_score"][0]["know_prob"],
                "relv_score": None,
                "faith_score": None,
            }
        ],
        "ref_doc": input_dic.get("ref_doc", ""),
        "retrieved": [
            {
                "rel_path": doc.metadata.get("rel_path"),
                "rank": doc.metadata.get("__rank__"),
                "content": [line.strip() for line in doc.page_content.split("\n") if line.strip()],
            }
            for doc in retrieved
        ],
    }

def gate_sota_output(input_dic, gen_ans) -> Dict[str, Any]:
    #breakpoint()
    return {
        "id": input_dic["id"],
        "question": input_dic["question"],
        "generated": gen_ans,
        "answer": input_dic["answer"],
        "agent": "sota_llm",
        "judge_score": [
            {
                "know_prob": input_dic["judge_score"][0]["know_prob"],
                "relv_score": input_dic["judge_score"][0]["relv_score"],
                "faith_score": None,
            }
        ],
        "ref_doc": input_dic.get("ref_doc", "")
    }


def iter_output(rag_out: Dict[str, Any], iter_result) -> Dict[str, Any]:
    iter_result = json.loads(iter_result.content)[0]
    return {
        "id": rag_out["id"],
        "iter_needed": iter_result["iter_needed"],
        "question": iter_result["new_query"],
        "generated": rag_out["generated"],
        "answer": rag_out["answer"],
        "ref_doc": rag_out.get("ref_doc", ""),
        "retrieved": rag_out["retrieved"],
    }