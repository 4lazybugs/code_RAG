from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import gc
import time
from tqdm import tqdm
from functools import partial

from models_eval import Rouge1Evaluator, RougeLEvaluator
from models_eval import BertEvaluator, SbertEvaluator
from models_eval import RecallEvaluator, MRREvaluator
from models_eval import GroundEvaluator, CorrectnessEvaluator
from models_eval import ExactMatchEvaluator


######## Config ##########
@dataclass(frozen=True)
class EvalConfig:
    qa_gt_path: Path
    retr_path: Path
    qag_path: Path
    md_results_path: Path


################## Repositories ####################################################
# Ground Truth Repository
class GTRepository:
    def __init__(self, gt_path: Path):
        self.gt_path = gt_path
        self.qid_list = []
        self.questions_list = []
        self.answers_list = []
        self.options_list = []
        self.ref_docs_list = []

    def load(self):
        if not self.gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {self.gt_path}")
        with open(self.gt_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)

        self.qid_list = [r.get("id") for r in gt_json]
        self.questions_list = [r.get("question", "") for r in gt_json]
        self.answers_list = [r.get("answer", "") for r in gt_json]
        self.options_list = [r.get("options", []) for r in gt_json]
        self.ref_docs_list = [r.get("reference_docs", []) for r in gt_json]
        return self


# QAG Repository
class QAGRepository:
    def __init__(self, qag_path: Path):
        self.qag_path = qag_path
        self.qid_list = []
        self.questions_list = []
        self.generated_list = []

    def load(self):
        if not self.qag_path.exists():
            raise FileNotFoundError(f"QAG file not found: {self.qag_path}")
        with open(self.qag_path, "r", encoding="utf-8") as f:
            qag_json = json.load(f)

        self.qid_list = [r.get("id") for r in qag_json]
        self.questions_list = [r.get("question", "") for r in qag_json]
        self.generated_list = [r.get("generated", "") for r in qag_json]
        return self


# Retrieved Repository
class RetrievedRepository:
    def __init__(self, retr_path: Path):
        self.retr_path = retr_path
        self.qid_list = []
        self.retrieved_list = []   # list[list[dict]]

    def load(self):
        if not self.retr_path.exists():
            raise FileNotFoundError(f"Retrieved file not found: {self.retr_path}")
        with open(self.retr_path, "r", encoding="utf-8") as f:
            retr_json = json.load(f)

        self.qid_list = []
        self.retrieved_list = []

        for row in retr_json:
            self.qid_list.append(row.get("id"))
            retrieved = row.get("retrieved", []) or []

            # ✅ 핵심: 샘플별 retrieved 전체를 리스트에 append
            self.retrieved_list.append(retrieved)

        return self


################## Evaluator Factory ####################################################
class EvaluatorFactory:
    ev_map = {
        "rouge1": Rouge1Evaluator,
        "rougeL": RougeLEvaluator,
        "bert": BertEvaluator,
        "sbert": SbertEvaluator,
        "recall": RecallEvaluator,
        "mrr": MRREvaluator,
        "ground": GroundEvaluator,
        "correctness": CorrectnessEvaluator,
        "em": ExactMatchEvaluator,
    }

    @classmethod
    def build(cls, metrics):
        factories = {}
        for m in metrics:
            if m not in cls.ev_map:
                raise ValueError(f"Unknown metric: {m}")
            factories[m] = partial(cls.ev_map[m])  # evaluator 생성 지연
        return factories

################## Score saving: markdown ####################################################
def write_metric_json(*, md_results_path: Path, mode: str, metric: str, rows: list[dict]):
    score_dir = md_results_path
    score_dir.mkdir(parents=True, exist_ok=True)

    out_path = score_dir / f"{mode}_{metric}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


################## EvalRunner ####################################################
class EvalRunner:
    def __init__(
        self,
        cfg: EvalConfig,
        *,
        gt_repo: GTRepository | None = None,
        qag_repo: QAGRepository | None = None,
        ret_repo: RetrievedRepository | None = None,
    ):
        self.cfg = cfg
        self.gt_repo = gt_repo or GTRepository(cfg.qa_gt_path)
        self.qag_repo = qag_repo or QAGRepository(cfg.qag_path)
        self.ret_repo = ret_repo or RetrievedRepository(cfg.retr_path)

    def _build_evaluators(self, metrics: list[str]): # evaluator 생성, runner 내부에서 사용
        factories = EvaluatorFactory.build(metrics)
        return {m: fac() for m, fac in factories.items()}

    def _iter_samples(self):
        """
        샘플 접근을 한 곳으로 몰아서 메인 loop 가독성 향상
        """
        gt, qag, ret = self.gt_repo, self.qag_repo, self.ret_repo
        n = len(gt.qid_list)

        for i in range(n):
            yield {
                "id": gt.qid_list[i],
                "question": gt.questions_list[i],
                "answer": gt.answers_list[i],
                "ref_docs": gt.ref_docs_list[i],
                "generated": qag.generated_list[i],
                "retrieved": ret.retrieved_list[i],  
            }

    def run(self, *, mode: str, metrics: list[str]) -> list[dict]: 
        print(f"Running {mode} evaluation...")
        # load once
        self.gt_repo.load()
        self.qag_repo.load()
        self.ret_repo.load()

        evaluators = self._build_evaluators(metrics) # line:148 함수를 호출하여 evaluator 생성
        rows_by_metric = {m: [] for m in metrics}

        total = len(self.gt_repo.qid_list)

        for sample in tqdm(self._iter_samples(), total=total, desc=mode, leave=True):
            base = {
                "id": sample["id"],
                "question": sample["question"],
                "reference": sample["answer"],
                "generated": sample["generated"],
                "gen_docs_num": len(sample["retrieved"]),
            }

            for m, ev in evaluators.items():
                score = ev.compute_scores(
                    [sample["answer"]],
                    [sample["generated"]],
                    sample["retrieved"],      # retrieved 전체 전달
                    sample["ref_docs"],
                )[0]
                row = dict(base)
                row[m] = float(score)
                rows_by_metric[m].append(row)

        # outputs
        md_results_path = self.cfg.md_results_path
        summary_rows = []
        for m in metrics:
            write_metric_json(md_results_path=md_results_path, mode=mode, metric=m, rows=rows_by_metric[m])

            values = [r[m] for r in rows_by_metric[m]]
            summary_rows.append({
                "mode": mode,
                "metric": m,
                "average": float(np.mean(values)),
                "std": float(np.std(values)),
            })

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Evaluation completed for {mode}")
        return summary_rows

################## save concatenated results: excel ####################################################
def write_summary_xlsx(summary_xlsx_path: Path, summary_rows: list[dict]):
    score_dir = summary_xlsx_path.parent
    score_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(summary_rows)
    df_avg = df.pivot(index="mode", columns="metric", values="average")
    df_std = df.pivot(index="mode", columns="metric", values="std")

    with pd.ExcelWriter(summary_xlsx_path) as writer:
        df_avg.to_excel(writer, sheet_name="average")
        df_std.to_excel(writer, sheet_name="std")


################## Main ####################################################
if __name__ == "__main__":
    start = time.time()

    metrics = ["mrr", "recall", "ground", "correctness", "em","rouge1", "rougeL", "bert", "sbert"]
    
    cfg_hotpot = EvalConfig(
        qa_gt_path=Path("qa_data/GT/hotpotqa_test/gt_merged_hotpotqa_test.json"),
        retr_path=Path("results/hotpotqa_test/retrieved_multihop.json"),
        qag_path=Path("results/inferenced/hotpotqa_test/qag_multihop.json"),
        md_results_path=Path("results/eval_score/hotpotqa_test/"),
    )

    cfg_mcq_rag = EvalConfig(
        qa_gt_path=Path("qa_data/GT/manual_book/mcq/gt_merged_manual_book.json"),
        retr_path=Path("results/retrieved/manual_book/mcq/retrieved_partial_10.json"),
        qag_path=Path("results/inferenced/manual_book/mcq/qag_partial_10.json"),
        md_results_path=Path("results/eval_score/mcq/"),
    )

    cfg_mcq_llm = EvalConfig(
        qa_gt_path=Path("qa_data/GT/manual_book/mcq/gt_merged_manual_book.json"),
        retr_path=Path("results/retrieved/manual_book/mcq/retrieved_raw_llm.json"),
        qag_path=Path("results/inferenced/manual_book/mcq/qag_raw_llm.json"),
        md_results_path=Path("results/eval_score/mcq/"),
    )

    cfg_test_rag = EvalConfig(
        qa_gt_path=Path("qa_data/test/gt_merged_test.json"),
        retr_path=Path("results/retrieved/test/retrieved_partial_10.json"),
        qag_path=Path("results/inferenced/test/qag_partial_10.json"),
        md_results_path=Path("results/eval_score/test_rag.xlsx"),
    )

    cfg_test_llm = EvalConfig(
        qa_gt_path=Path("qa_data/test/gt_merged_test.json"),
        retr_path=Path("results/retrieved/test/retrieved_raw_llm.json"),
        qag_path=Path("results/inferenced/test/qag_raw_llm.json"),
        md_results_path=Path("results/eval_score/test_llm.xlsx"),
    )

    runner_test_rag = EvalRunner(cfg_test_rag)
    runner_test_llm = EvalRunner(cfg_test_llm)
    runner_mcq_rag = EvalRunner(cfg_mcq_rag)
    runner_mcq_llm = EvalRunner(cfg_mcq_llm)

    all_summary_rows: list[dict] = []

    results_test_rag = runner_test_rag.run(mode="RAG", metrics=metrics)
    all_summary_rows.extend(results_test_rag)
    results_test_llm = runner_test_llm.run(mode="naive_LLM", metrics=metrics)
    all_summary_rows.extend(results_test_llm)
    results_mcq_rag = runner_mcq_rag.run(mode="RAG", metrics=metrics)
    all_summary_rows.extend(results_mcq_rag)
    results_mcq_llm = runner_mcq_llm.run(mode="naive_LLM", metrics=metrics)
    all_summary_rows.extend(results_mcq_llm)

    write_summary_xlsx(
        summary_xlsx_path=Path("results/eval_score/mcq/summary_test.xlsx"),
        summary_rows=all_summary_rows
    )
    print(f"⏱ 전체 평가 완료: {(time.time()-start)/60:.2f}분")
