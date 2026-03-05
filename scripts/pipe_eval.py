import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, List
import json
import pandas as pd

# evaluator import
from src.eval import Params
from src.eval import EMEvaluator, Rouge1Evaluator, RougeLEvaluator # n-gram 평가지표
from src.eval import BERTEvaluator, SBERTEvaluator, BleurtEvaluator # 의미적 평가지표
from src.eval import RecallEvaluator, MRREvaluator # 검색품질 지표

def write_metric_json(out_dir: Path, mode: str, metric: str, batch_res: List[Dict[str, Any]], scores: List[float]):
    # mode별로 서브디렉토리 생성
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    out_path = mode_dir / f"{metric}.json"
    rows = [{**sample, metric: float(sc)} for sample, sc in zip(batch_res, scores)]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_summary_xlsx(summary_xlsx_path: Path, summary_rows: List[Dict[str, Any]]):
    summary_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    with pd.ExcelWriter(summary_xlsx_path, engine="openpyxl") as w:
        df.pivot(index="mode", columns="metric", values="average") \
        .to_excel(excel_writer=w, sheet_name="average")

        df.pivot(index="mode", columns="metric", values="std") \
        .to_excel(excel_writer=w, sheet_name="std")


if __name__ == "__main__":

    qa_path = Path("db/qa_data/test/retrieval_test.json")
    params_qa = Params(qa_path)
    ref_docs = params_qa.get_param("ref_doc")


    result_path = Path("results/inferenced/test/test_retrieval.json")
    params_res = Params(result_path)
    
    ref = params_res.get_param("answer")
    ans = params_res.get_param("generated")
    retrieved = params_res.get_param("retrieved")
    batch_res = [{"reference": ref, "generated": gen}
        for ref, gen in zip(ref, ans)]
    batch_retr = [{"gen_docs": retr, "ref_docs": doc}
        for retr, doc in zip(retrieved, ref_docs)
    ]

    em_eval = EMEvaluator()
    rouge1_eval = Rouge1Evaluator()
    rougeL_eval = RougeLEvaluator()
    bert_eval = BERTEvaluator()
    sbert_eval = SBERTEvaluator()
    bleurt_eval = BleurtEvaluator()
    mrr_eval = MRREvaluator()
    recall_eval = RecallEvaluator()
    
    result_recall = recall_eval.score_all(batch_retr)
    result_mrr = mrr_eval.score_all(batch_retr)
    result_01 = bert_eval.score_all(batch_res)
    result_02 = sbert_eval.score_all(batch_res)
    result_03 = rouge1_eval.score_all(batch_res)
    result_04 = rougeL_eval.score_all(batch_res)
    result_05 = em_eval.score_all(batch_res)
    result_06 = bleurt_eval.score_all(batch_res)


    print(f"Recall: {result_recall}")
    print(f"MRR: {result_mrr}")
    print(f"BERT: {result_01}")
    print(f"SBERT: {result_02}")
    print(f"ROUGE1: {result_03}")
    print(f"ROUGEL: {result_04}")
    print(f"Exact_Match: {result_05}")
    print(f"BLUERT: {result_06}")

'''
    CFG = get_config()
    
    # mode_list는 딕셔너리 형식: {mode: {data_path, json_dir}}
    mode_configs = CFG.mode_list
    
    base_xlsx_path = Path(CFG.xlsx_path)
    summary_rows: List[Dict[str, Any]] = []

    selected_metrics = [m.strip() for m in CFG.metrics.split(",") if m.strip()]
    for mode, mode_cfg in mode_configs.items():
        data_path = Path(mode_cfg["data_path"])
        json_dir = Path(mode_cfg["json_dir"])
        
        fetcher = Params(data_path)
        
        for metric in selected_metrics:
            EvCls = metric_dict[metric]
            batch = fetcher.fetch(metric)
            scores = EvCls().score_all(batch)

            write_metric_json(json_dir, mode, metric, batch, scores)

            s = pd.Series(scores, dtype="float64")
            summary_rows.append({"mode": mode, "metric": metric, "average": float(s.mean()), "std": float(s.std())})

    write_summary_xlsx(base_xlsx_path, summary_rows)
'''