# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, List
import json
import pandas as pd
import argparse, os, yaml
from types import SimpleNamespace

from fetch_data import FetchData
from evaluators.base import metric_dict

def load_yaml(path='src/eval/config_eval.yaml'):
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # 환경 변수 치환 처리
    config = {}
    for k, v in raw_config.items():
        if isinstance(v, str):
            config[k] = os.path.expandvars(v)
        else:
            config[k] = v

    return config

def get_config():
    default_cfg = load_yaml()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=default_cfg.get('model_name'))

    args, _ = parser.parse_known_args()

    # ✅ YAML + argparse 병합 → Namespace
    cfg = {**default_cfg, **vars(args)}
    return SimpleNamespace(**cfg)


def write_metric_json(out_dir: Path, mode: str, metric: str, batch_data: List[Dict[str, Any]], scores: List[float]):
    # mode별로 서브디렉토리 생성
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    out_path = mode_dir / f"{metric}.json"
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
    CFG = get_config()
    
    # mode_list는 딕셔너리 형식: {mode: {data_path, json_dir}}
    mode_configs = CFG.mode_list
    
    base_xlsx_path = Path(CFG.xlsx_path)
    summary_rows: List[Dict[str, Any]] = []

    for mode, mode_cfg in mode_configs.items():
        data_path = Path(mode_cfg["data_path"])
        json_dir = Path(mode_cfg["json_dir"])
        
        fetcher = FetchData(data_path)
        
        for metric, EvCls in metric_dict.items():
            batch = fetcher.fetch(metric)
            scores = EvCls().score_all(batch)

            write_metric_json(json_dir, mode, metric, batch, scores)

            s = pd.Series(scores, dtype="float64")
            summary_rows.append({"mode": mode, "metric": metric, "average": float(s.mean()), "std": float(s.std())})

    write_summary_xlsx(base_xlsx_path, summary_rows)
