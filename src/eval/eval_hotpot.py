import json, time, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch, gc
from tqdm import tqdm
from functools import partial

# ✅ 당신 프로젝트 evaluator import 경로에 맞추세요
from models_eval import Rouge1Evaluator, RougeLEvaluator
from models_eval import BertEvaluator, SbertEvaluator
from models_eval import RecallEvaluator, MRREvaluator
from models_eval import GroundEvaluator, CorrectnessEvaluator
from models_eval import ExactMatchEvaluator
from models_eval.base import BaseEvaluator  # ✅ Supporting evaluator 만들려면 필요


# =============================================================================
# 0) Supporting Fact Metric (NEW)
# =============================================================================

def _normalize(text) -> str:
    if text is None:
        return ""
    if hasattr(text, "content"):
        text = text.content
    if isinstance(text, dict):
        for k in ("content", "text", "answer", "output"):
            if k in text:
                text = text[k]
                break
    return str(text).strip()

class SupportingGenContainEvaluator(BaseEvaluator):
    """
    supporting_fact_gen이 supporting_fact_ref 또는 supporting_fact_comp에 포함되는지 체크.
    - 포함되면 1.0, 아니면 0.0
    - info dict는 QAG json의 row["info"]에서 꺼내서 runner가 전달해줘야 함.
    """
    metric_key = "supporting_gen_contain"

    def __init__(self, remove_spaces: bool = True, case_insensitive: bool = True):
        super().__init__()
        self.remove_spaces = remove_spaces
        self.case_insensitive = case_insensitive

    def _prep(self, s: str) -> str:
        s = _normalize(s)
        if self.case_insensitive:
            s = s.lower()
        if self.remove_spaces:
            s = re.sub(r"\s+", "", s)
        return s

    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs: list) -> list:
        """
        여기서는 generated = [info_dict] 형태로 들어오는 것을 기대.
        (references/ref_docs/gen_docs는 사용 안 함)
        """
        scores = []
        for g in generated:
            if not isinstance(g, dict):
                scores.append(0.0)
                continue

            sf_gen = self._prep(g.get("supporting_fact_gen", ""))
            sf_ref = self._prep(g.get("supporting_fact_ref", ""))
            sf_comp = self._prep(g.get("supporting_fact_comp", ""))

            # gen이 비어있으면 실패
            if not sf_gen:
                scores.append(0.0)
                continue

            ok = (sf_gen in sf_ref) or (sf_gen in sf_comp)
            scores.append(1.0 if ok else 0.0)

        return scores


# =============================================================================
# 1) Config
# =============================================================================

@dataclass(frozen=True)
class HotpotEvalCfg:
    gt_path: Path          # Hotpot GT json
    qag_path: Path         # inferenced qag json
    retr_path: Path        # retrieved json
    out_dir: Path          # per-metric json 저장 폴더
    summary_xlsx: Optional[Path] = None  # 요약 엑셀(optional)


# =============================================================================
# 2) Repositories (입력 JSON 읽기만 담당)
# =============================================================================

class HotpotGTRepo:
    """
    기대 GT format:
    [
      { "id":..., "question":..., "answer":...,
        "supporting_fact_ref":..., "supporting_fact_comp":..., "link_word":... (optional),
        "reference_docs":... (optional) }
    ]
    """
    def __init__(self, path: Path):
        self.path = path
        self.items: list[dict] = []

    def load(self) -> "HotpotGTRepo":
        if not self.path.exists():
            raise FileNotFoundError(f"GT file not found: {self.path}")
        self.items = json.loads(self.path.read_text(encoding="utf-8"))
        return self


class QAGRepo:
    """
    기대 QAG format (당신 infer 코드):
    [
      { "id":..., "question":..., "reference":..., "generated":..., "info": {...} }
    ]
    """
    def __init__(self, path: Path):
        self.path = path
        self.items: list[dict] = []

    def load(self) -> "QAGRepo":
        if not self.path.exists():
            raise FileNotFoundError(f"QAG file not found: {self.path}")
        self.items = json.loads(self.path.read_text(encoding="utf-8"))
        return self


class RetrievedRepo:
    """
    기대 retrieved format:
    [
      { "id":..., "retrieved": [ {rank, filename, content(list[str])}, ... ] }
    ]
    """
    def __init__(self, path: Path):
        self.path = path
        self.items: list[dict] = []

    def load(self) -> "RetrievedRepo":
        if not self.path.exists():
            raise FileNotFoundError(f"Retrieved file not found: {self.path}")
        self.items = json.loads(self.path.read_text(encoding="utf-8"))
        return self


# =============================================================================
# 3) Evaluator Factory
# =============================================================================

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

        # ✅ NEW supporting metric
        "supporting_gen_contain": SupportingGenContainEvaluator,
    }

    @classmethod
    def build(cls, metrics: list[str]) -> dict[str, Any]:
        out = {}
        for m in metrics:
            if m not in cls.ev_map:
                raise ValueError(f"Unknown metric: {m}")
            out[m] = partial(cls.ev_map[m])()
        return out


# =============================================================================
# 4) Writer
# =============================================================================

def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

def write_summary_xlsx(path: Path, summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    df_avg = df.pivot(index="mode", columns="metric", values="average")
    df_std = df.pivot(index="mode", columns="metric", values="std")
    with pd.ExcelWriter(path) as writer:
        df_avg.to_excel(writer, sheet_name="average")
        df_std.to_excel(writer, sheet_name="std")


# =============================================================================
# 5) Hotpot Evaluator Runner (핵심)
# =============================================================================

class HotpotEvalRunner:
    def __init__(
        self,
        cfg: HotpotEvalCfg,
        *,
        gt_repo: Optional[HotpotGTRepo] = None,
        qag_repo: Optional[QAGRepo] = None,
        ret_repo: Optional[RetrievedRepo] = None,
    ):
        self.cfg = cfg
        self.gt_repo = gt_repo or HotpotGTRepo(cfg.gt_path)
        self.qag_repo = qag_repo or QAGRepo(cfg.qag_path)
        self.ret_repo = ret_repo or RetrievedRepo(cfg.retr_path)

    def _index_by_id(self, items: list[dict]) -> dict[Any, dict]:
        out = {}
        for it in items:
            out[it.get("id")] = it
        return out

    def run(self, *, mode: str, metrics: list[str]) -> list[dict]:
        # 1) load
        self.gt_repo.load()
        self.qag_repo.load()
        self.ret_repo.load()

        qag_map = self._index_by_id(self.qag_repo.items)
        ret_map = self._index_by_id(self.ret_repo.items)

        # 2) evaluators
        evaluators = EvaluatorFactory.build(metrics)
        rows_by_metric = {m: [] for m in metrics}

        # 3) iterate by GT order
        total = len(self.gt_repo.items)
        for gt in tqdm(self.gt_repo.items, total=total, desc=f"Eval({mode})", leave=True):
            qid = gt.get("id")
            qag = qag_map.get(qid, {})
            ret = ret_map.get(qid, {})

            question = gt.get("question", "")
            answer = gt.get("answer", "")
            generated = qag.get("generated", "")

            # GT supporting facts
            gt_sf_ref = gt.get("supporting_fact_ref", "")
            gt_sf_comp = gt.get("supporting_fact_comp", "")
            gt_link = gt.get("link_word", "")

            # infer info
            info = qag.get("info") or {}
            gen_sf_ref = info.get("supporting_fact_ref", "")
            gen_sf_comp = info.get("supporting_fact_comp", "")
            gen_sf_gen = info.get("supporting_fact_gen", "")
            gen_link = info.get("link_word", "")

            retrieved = ret.get("retrieved", []) or []
            ref_docs = gt.get("reference_docs", [])  # 없으면 []

            base = {
                "id": qid,
                "question": question,
                "reference": answer,
                "generated": generated,
                "gen_docs_num": len(retrieved),

                "gt_link_word": gt_link,
                "gt_supporting_fact_ref": gt_sf_ref,
                "gt_supporting_fact_comp": gt_sf_comp,

                "gen_link_word": gen_link,
                "gen_supporting_fact_ref": gen_sf_ref,
                "gen_supporting_fact_comp": gen_sf_comp,
                "gen_supporting_fact_gen": gen_sf_gen,
            }

            # metric 계산
            for m, ev in evaluators.items():
                # ✅ supporting metric은 generated에 "info dict"를 넣어줌
                if getattr(ev, "metric_key", "") == "supporting_gen_contain":
                    ref_in = [""]          # not used
                    gen_in = [info]        # dict 전달
                else:
                    ref_in = [answer]
                    gen_in = [generated]

                score = ev.compute_scores(
                    ref_in,
                    gen_in,
                    retrieved,
                    ref_docs,
                )[0]

                row = dict(base)
                row[m] = float(score)
                rows_by_metric[m].append(row)

        # 4) save per-metric + summary
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        for m in metrics:
            out_path = self.cfg.out_dir / f"{mode}_{m}.json"
            write_json(out_path, rows_by_metric[m])

            vals = [r[m] for r in rows_by_metric[m]]
            summary_rows.append({
                "mode": mode,
                "metric": m,
                "average": float(np.mean(vals)) if vals else 0.0,
                "std": float(np.std(vals)) if vals else 0.0,
            })

        # 5) cleanup
        torch.cuda.empty_cache()
        gc.collect()

        # optional xlsx
        if self.cfg.summary_xlsx is not None:
            write_summary_xlsx(self.cfg.summary_xlsx, summary_rows)

        return summary_rows


# =============================================================================
# 6) MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    # ✅ 기존 metric + supporting metric 추가
    metrics = ["mrr", "recall", "ground", "correctness", "em",
               "rouge1", "rougeL", "bert", "sbert",
               "supporting_gen_contain"]

    cfg = HotpotEvalCfg(
        gt_path=Path("qa_data/GT/hotpotqa_test/gt_merged_hotpotqa_test.json"),
        qag_path=Path("results/inferenced/hotpotqa_test/qag_multihop.json"),
        retr_path=Path("results/retrieved/hotpotqa_test/retrieved_multihop.json"),
        out_dir=Path("results/eval_score/hotpotqa_test/"),
        summary_xlsx=Path("results/eval_score/hotpotqa_test/summary.xlsx"),
    )

    runner = HotpotEvalRunner(cfg)
    summary = runner.run(mode="multihop", metrics=metrics)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"⏱ 전체 평가 완료: {(time.time()-start)/60:.2f}분")
