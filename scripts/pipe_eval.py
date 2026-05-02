import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.config import get_config, load_yaml

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
        data = json.loads(self.data_path.read_text(encoding="utf-8"))
        return sorted(data, key=lambda x: x.get("id", 0))  # id 순서대로 정렬

    def get_param(self, key: str) -> list[Any]:
        return [record.get(key) for record in self.data]


if __name__ == "__main__":
    start_time = time.time()
    CFG_eval = get_config("configs/config_eval.yaml")
    CFG_fdir = get_config("configs/config_fdir.yaml")

    # evaluator 초기화
    em_eval     = EMEvaluator()
    rouge1_eval = Rouge1Evaluator(CFG_eval.tokenizer_name)
    rougeL_eval = RougeLEvaluator(CFG_eval.tokenizer_name)
    bert_eval   = BERTEvaluator(CFG_eval.bert_model_name, CFG_eval.bert_num_layers)
    sbert_eval  = SBERTEvaluator(CFG_eval.sbert_model_name)
    bleurt_eval = BleurtEvaluator(CFG_eval.bleurt_model_name)
    mrr_eval    = MRREvaluator()
    recall_eval = RecallEvaluator()

    json_files = list(Path(CFG_fdir.infered_dir).rglob("*.json"))
    print(f"Found {len(json_files)} files in {CFG_fdir.infered_dir}")

    for result_path in json_files:
        print(f"\n Processing: {result_path}")
        params_res = Params(result_path)

        ref_docs  = params_res.get_param("ref_doc")
        ref       = params_res.get_param("answer")
        ans       = params_res.get_param("generated")
        ids       = params_res.get_param("id")
        retrieved = params_res.get_param("retrieved")

        batch_res  = [{"id": i, "reference": r, "generated": g}
                      for i, r, g in zip(ids, ref, ans)]
        batch_retr = [{"id": i, "ref_docs": doc, "gen_docs": retr}
                      for i, retr, doc in zip(ids, retrieved, ref_docs)]

        results = {}
        for metric_name, (batch, evaluator) in [
            ("recall", (batch_retr, recall_eval)),
            ("mrr",    (batch_retr, mrr_eval)),
            ("bert",   (batch_res,  bert_eval)),
            ("sbert",  (batch_res,  sbert_eval)),
            ("rouge1", (batch_res,  rouge1_eval)),
            ("rougeL", (batch_res,  rougeL_eval)),
            ("em",     (batch_res,  em_eval)),
        ]:
            results[metric_name] = (batch, evaluator.score_all(batch))

        save_eval_results(
            save_dir=Path(CFG_fdir.eval_dir) / result_path.stem,
            results=results,
        )

        for metric, (_, scores) in results.items():
            s = pd.Series(scores)
            print(f"  [{metric}]  avg={s.mean():.4f}  std={s.std():.4f}")

    print(f"\nTOTAL elapsed: {time.time() - start_time:.2f}s")
    print(f"result in {CFG_fdir.eval_dir}")