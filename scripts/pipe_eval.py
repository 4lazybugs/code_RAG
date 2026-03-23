import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -*- coding: utf-8 -*-
from pathlib import Path
import json
import pandas as pd
from typing import Any, Dict, List, Mapping
from dataclasses import dataclass, field

# evaluator import
from src.eval import EMEvaluator, Rouge1Evaluator, RougeLEvaluator # n-gram 평가지표
from src.eval import BERTEvaluator, SBERTEvaluator, BleurtEvaluator # 의미적 평가지표
from src.eval import RecallEvaluator, MRREvaluator # 검색품질 지표

# json이랑 액셀까지 저장
def save_eval_results(
    save_dir: Path,
    results: dict[str, tuple[list[dict], list[float]]],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for metric, (batch, scores) in results.items():
        rows = [{metric: float(sc), **s} for s, sc in zip(batch, scores)]
        with open(save_dir / f"{metric}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        s = pd.Series(scores)
        summary_rows.append({"metric": metric, "average": s.mean(), "std": s.std()})

    df = pd.DataFrame(summary_rows)
    with pd.ExcelWriter(save_dir / "summary.xlsx", engine="openpyxl") as w:
        df.set_index("metric")["average"].to_frame().T.to_excel(w, sheet_name="average")
        df.set_index("metric")["std"].to_frame().T.to_excel(w, sheet_name="std")


class Params:
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.data = self._load_data()

    def _load_data(self) -> list[dict]:
        return json.loads(self.data_path.read_text(encoding="utf-8"))

    def get_param(self, key: str) -> list[Any]:
        return [record.get(key) for record in self.data]


if __name__ == "__main__":

    qa_path = Path("db/qa_data/test_saq/merged.json")
    params_qa = Params(qa_path)
    ref_docs = params_qa.get_param("ref_doc")

    result_path = Path("results/inferenced/qa_in_used.json")
    params_res = Params(result_path)
    
    ref = params_res.get_param("answer")
    ans = params_res.get_param("generated")
    retrieved = params_res.get_param("retrieved")
    batch_res = [{"reference": ref, "generated": gen}
        for ref, gen in zip(ref, ans)]
    batch_retr = [{"ref_docs": doc, "gen_docs": retr}
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

    results={
        "recall": (batch_retr, recall_eval.score_all(batch_retr)),
        "mrr":    (batch_retr, mrr_eval.score_all(batch_retr)),
        "bert":   (batch_res,  bert_eval.score_all(batch_res)),
        "sbert":  (batch_res,  sbert_eval.score_all(batch_res)),
        "rouge1": (batch_res,  rouge1_eval.score_all(batch_res)),
        "rougeL": (batch_res,  rougeL_eval.score_all(batch_res)),
        "em":     (batch_res,  em_eval.score_all(batch_res)),
        "bleurt": (batch_res,  bleurt_eval.score_all(batch_res)),
    }

    save_eval_results(
        save_dir=Path("results/eval_score"),
        results=results,
    )

    # 각 지표별 점수 리스트로 출력
    for metric, (_, scores) in results.items():
        s = pd.Series(scores)
        print(f"\n[{metric}]  avg={s.mean():.4f}  std={s.std():.4f}")
        print(s.round(4).to_string())