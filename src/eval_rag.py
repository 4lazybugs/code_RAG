import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import pandas as pd
import torch
import time
import gc
import numpy as np
from konlpy.tag import Okt
from rouge_score.tokenizers import Tokenizer

from models_eval import Rouge1Evaluator, RougeLEvaluator
from models_eval import BertEvaluator, SbertEvaluator
from models_eval import RecallEvaluator, MRREvaluator
from models_eval import GroundEvaluator, CorrectnessEvaluator
from pathlib import Path
from tqdm import tqdm

class KoreanTokenizer(Tokenizer):
    def __init__(self):
        self.okt = Okt()

    def tokenize(self, text):
        return self.okt.morphs(text)

def run_eval_and_save(
    *,
    selected_modes: list[str],
    selected_metrics: list[str],
    dataset_tag: str,  # "manual_book" or "test"
    gt_path: str,      # e.g., "qa_data/GT/manual_book/gt_merged_manual_book.json"
    results_root: str = "results",
    qa_mode: str = "cleaned",
    sample_size=None,
    summary_xlsx_path: str = "results/summary.xlsx",
):
    """
    - GT(QA) / QAG / retrieved json을 로드
    - metric별로 evaluator 실행
    - results/score/{dataset_tag}/{mode}_{metric}.json 저장
    - results/summary.xlsx(average/std) 저장
    """

    # BaseEvaluator 시그니처 맞추기용 더미 (현 구조 유지)
    qa_data_path = {qa_mode: "__unused__"}

    # Evaluator 클래스 매핑
    ev_map = {
        "rouge1": Rouge1Evaluator,
        "rougeL": RougeLEvaluator,
        "bert": BertEvaluator,
        "sbert": SbertEvaluator,
        "recall": RecallEvaluator,
        "mrr": MRREvaluator,
        "ground": GroundEvaluator,
        "correctness": CorrectnessEvaluator,
    }

    summary = []

    gt_pth = Path(gt_path)
    if not gt_pth.exists():
        raise FileNotFoundError(f"GT file not found: {gt_pth}")

    # GT는 mode마다 동일하므로 mode loop 밖에서 한 번만 로드해도 되지만,
    # 원 코드 흐름을 보존하려면 mode loop 안에 둬도 됩니다.
    with open(gt_pth, "r", encoding="utf-8") as f:
        gt_json = json.load(f)

    questions = [r["question"] for r in gt_json]
    gt_ans = [r["answer"] for r in gt_json]
    qa_id = [r["id"] for r in gt_json]
    ref_doc = [r["reference_docs"] for r in gt_json]

    results_root = Path(results_root)

    for mode in selected_modes:
        print(f"\n[MODE START] {mode}", flush=True)

        # ---------- generated(QAG) 로드 ----------
        gen_pth = results_root / "qag" / dataset_tag / f"qag_{mode}.json"
        if not gen_pth.exists():
            raise FileNotFoundError(f"QAG file not found: {gen_pth}")
        with open(gen_pth, "r", encoding="utf-8") as f:
            gen_json = json.load(f)
        gen_ans = [r["generated"] for r in gen_json]

        # ---------- retrieved 로드 ----------
        ret_pth = results_root / "retrieved" / dataset_tag / f"retrieved_{mode}.json"
        if not ret_pth.exists():
            raise FileNotFoundError(f"Retrieved file not found: {ret_pth}")
        with open(ret_pth, "r", encoding="utf-8") as f:
            retr_json = json.load(f)
        gen_docs = [r["retrieved"] for r in retr_json]

        # ---------- metric별 평가 ----------
        for metric in selected_metrics:
            if metric not in ev_map:
                raise ValueError(f"Unknown metric: {metric}")

            print(f"  [METRIC] {metric}", flush=True)
            EvCls = ev_map[metric]

            ev = EvCls(expert=None, qa_data_path=qa_data_path[qa_mode], sample_size=sample_size)

            rows = []
            for i in tqdm(range(len(qa_id)), desc=f"{mode}-{metric}", leave=False):
                score = ev.compute_scores([gt_ans[i]], [gen_ans[i]], gen_docs[i], ref_doc[i])[0]

                rows.append({
                    "id": qa_id[i],
                    "question": questions[i],
                    "reference": gt_ans[i],
                    "generated": gen_ans[i],
                    "gen_docs_num": len(gen_docs[i]),
                    metric: float(score),
                })

            out_path = results_root / "score" / dataset_tag / f"{mode}_{metric}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

            values = [r[metric] for r in rows]
            summary.append({
                "mode": mode,
                "metric": metric,
                "average": float(np.mean(values)),
                "std": float(np.std(values)),
            })

            torch.cuda.empty_cache()
            gc.collect()
            print(f"  [DONE] {mode}-{metric} saved ({len(rows)})", flush=True)

    # ---------- summary.xlsx 저장 ----------
    print("  [SUMMARY] Saving summary...", flush=True)
    df = pd.DataFrame(summary)
    df_avg = df.pivot(index="mode", columns="metric", values="average")
    df_std = df.pivot(index="mode", columns="metric", values="std")

    summary_xlsx_path = Path(summary_xlsx_path)
    summary_xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(summary_xlsx_path) as writer:
        df_avg.to_excel(writer, sheet_name="average")
        df_std.to_excel(writer, sheet_name="std")

    print(f"✅ 모든 평가 완료: {summary_xlsx_path} (average/std 시트 포함)")


if __name__ == '__main__':
    start_time = time.time()

    selected_modes = ['partial_10', 'raw_llm']
    selected_metrics = ['rouge1', 'rougeL', 'bert', 'sbert', 'recall', 'mrr', 'ground', 'correctness']

    run_eval_and_save(
        selected_modes=selected_modes,
        selected_metrics=selected_metrics,
        dataset_tag="manual_book",
        gt_path="qa_data/GT/manual_book/gt_merged_manual_book.json",
        results_root="results",
        summary_xlsx_path="results/manual_book/summary.xlsx",
    )

    run_eval_and_save(
        selected_modes=selected_modes,
        selected_metrics=selected_metrics,
        dataset_tag="farm_consulting",
        gt_path="qa_data/GT/farm_consulting/gt_merged_farm_consulting.json",
        results_root="results",
        summary_xlsx_path="results/farm_consulting/summary.xlsx",
    )

    '''
    run_eval_and_save(
        selected_modes=selected_modes,
        selected_metrics=selected_metrics,
        dataset_tag="test",
        gt_path="qa_data/test/gt_merged_test.json",
        results_root="results",
        summary_xlsx_path="results/test/summary.xlsx",
    )
    '''

    elapsed = time.time() - start_time
    print(f"⏱ 전체 평가 완료: {elapsed/60:.2f}분")
