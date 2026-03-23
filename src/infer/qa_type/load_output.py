from typing import Dict, Any
import json

def llm_output(input_dic, gen_ans) -> Dict[str, Any]:
    #breakpoint()
    return {
        "id": input_dic["id"],
        "question": input_dic["question"],
        "generated": gen_ans,
        "answer": input_dic["answer"],
    }

def rag_output(input_dic, gen_ans, retrieved) -> Dict[str, Any]:
    #breakpoint()
    return {
        "id": input_dic["id"],
        "question": input_dic["question"],
        "generated": gen_ans,
        "answer": input_dic["answer"],
        "retrieved": [
            {
                "rel_path": doc.metadata.get("rel_path"),
                "rank": doc.metadata.get("__rank__"),
                "content": [line.strip() for line in doc.page_content.split("\n") if line.strip()],
            }
            for doc in retrieved
        ],
    }


def iter_output(rag_out: Dict[str, Any], iter_result) -> Dict[str, Any]:
    iter_result = json.loads(iter_result.content)[0]
    return {
        "id": rag_out["id"],
        "iter_needed": iter_result["iter_needed"],
        "question": iter_result["new_query"],
        "generated": rag_out["generated"],
        "answer": rag_out["answer"],
        "retrieved": rag_out["retrieved"],
    }