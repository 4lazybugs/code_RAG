# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, List
import json
import pandas as pd

from fetch_data import FetchData
from evaluators.base import metric_dict


def write_metric_json(out_dir: Path, mode: str, metric: str, batch_data: List[Dict[str, Any]], scores: List[float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mode}_{metric}.json"
    rows = [{**sample, metric: float(sc)} for sample, sc in zip(batch_data, scores)]
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
    mode = "test"
    data_path = Path("results/inferenced/hotpotqa_test/qag_multihop.json")
    json_dir = Path("results/eval_score/test/json")
    xlsx_path = Path("results/eval_score/test/summary.xlsx")

    fetcher = FetchData(data_path)
    summary_rows: List[Dict[str, Any]] = []

    for metric, EvCls in metric_dict.items():
        batch = fetcher.fetch(metric)
        scores = EvCls().score_all(batch)

        write_metric_json(json_dir, mode, metric, batch, scores)

        s = pd.Series(scores, dtype="float64")
        summary_rows.append({"mode": mode, "metric": metric, "average": float(s.mean()), "std": float(s.std())})

    write_summary_xlsx(xlsx_path, summary_rows)
