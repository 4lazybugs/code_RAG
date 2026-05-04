from __future__ import annotations

import json
from pathlib import Path
from dotenv import load_dotenv
from itertools import groupby

from src.config import get_config
from src.postprocess.nli_filter import NLIFilter, nli_QA, nli_CD

def qa_key(item: dict) -> tuple:
    return (
        item.get("chunk_name"),
        item.get("question"),
        item.get("answer"),
    )

def run_nli_filter(qa2d_results: list[dict], nli: NLIFilter, threshold: float, strategy):
    print(f"[INFO] NLI 필터링 중... (threshold={threshold})")
    all_results, entailed, neutral, contradiction = nli.filter_batch(qa2d_results)

    if strategy == nli_CD:
        filtered = [r for r in all_results if r["score_entailment"] >= threshold]
    else:  # nli_QA
        filtered = [r for r in all_results if r["score_neutral"] >= threshold]

    print(f"[INFO] 필터링 결과: {len(filtered)}/{len(all_results)} "
          f"({len(filtered)/max(len(all_results), 1)*100:.1f}%)")
    return all_results, entailed, neutral, contradiction, filtered

def save_qa(qa_list: list[dict], chunk_root: Path, out_root: Path, suffix: str = "") -> None:
    chunk_files = list(chunk_root.rglob("*_chunk*.json"))

    keyfunc = lambda x: x["chunk_name"]
    for source_file, group in groupby(sorted(qa_list, key=keyfunc), key=keyfunc):
        items = list(group)

        matched = next((f for f in chunk_files if f.name.startswith(Path(source_file).stem.rsplit("_", 1)[0])), None)
        rel_dir  = matched.parent.relative_to(chunk_root) if matched else Path(".")
        save_dir = out_root / suffix / rel_dir if suffix else out_root / rel_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(source_file).stem
        for qa_idx, item in enumerate(items):
            filename = f"{stem}_qa{qa_idx+1:03d}.json" # source_file마다 1부터 시작하는 번호
            with open(save_dir / filename, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)

def load_qa_from_dir(qa_root: Path) -> list[dict]:
    results = []
    for f in sorted(qa_root.rglob("*.json")):
        item = json.loads(f.read_text(encoding="utf-8"))
        results.append(item)
    print(f"[INFO] {len(results)}개 QA 로드 완료")
    return results

############### load params #######################
CFG = get_config("configs/config_eval.yaml")

QA_ROOT  = Path(CFG.qa_in_dir)
OUT_ROOT = Path(CFG.qa_filt_dir)

STRATEGIES = [
    {"strategy": nli_CD, "threshold": CFG.nli_cd_threshold},
    {"strategy": nli_QA, "threshold": CFG.nli_qa_threshold},
]

NLI_MODEL = CFG.nli_model
MAX_QA = CFG.MAX_QA
###################################################

if __name__ == "__main__":
    load_dotenv()

    chunk_root = QA_ROOT
    qa_list    = load_qa_from_dir(QA_ROOT)
    qa_list = qa_list[:MAX_QA]

    selected_by_strategy = {}

    for cfg in STRATEGIES:
        strategy  = cfg["strategy"]
        threshold = cfg["threshold"]

        print(f"\n[INFO] === {strategy.__name__} 전략 시작 (threshold={threshold}) ===")
        nli = NLIFilter(model_name=NLI_MODEL, strategy=strategy, device="cuda")

        all_results, entailed, neutral, contradiction, filtered = run_nli_filter(
            qa_list, nli, threshold, strategy
        )

        out_dir  = OUT_ROOT / strategy.__name__
        rejected = [r for r in all_results if r not in filtered]

        # threshold 기반 selected/rejected
        save_qa(filtered,  chunk_root, out_dir, suffix="selected")
        save_qa(rejected,  chunk_root, out_dir, suffix="rejected")

        # 추가: 전략별 selected 저장
        selected_by_strategy[strategy.__name__] = filtered

        print(f"[INFO] entailed {len(entailed)} / neutral {len(neutral)} / contradiction {len(contradiction)}")
        print(f"[INFO] selected {len(filtered)} / rejected {len(rejected)}")

    # 추가: 두 selected 폴더에 공통으로 존재하는 QA만 final_selected 저장
    def qa_key(item: dict) -> tuple:
        return (
            item.get("chunk_name"),
            item.get("question"),
            item.get("answer"),
        )

    cd_selected = selected_by_strategy["nli_CD"]
    qa_selected = selected_by_strategy["nli_QA"]

    qa_keys = {qa_key(item) for item in qa_selected}

    final_selected = [
        item for item in cd_selected
        if qa_key(item) in qa_keys
    ]

    final_out_dir = OUT_ROOT / "final_selected"
    save_qa(final_selected, chunk_root, final_out_dir)

    total = len(qa_list)
    final_ratio = len(final_selected) / max(total, 1) * 100

    print(f"[INFO] final selected {len(final_selected)}/{total} ({final_ratio:.1f}%)")
    print(f"\n[INFO] 완료 → {OUT_ROOT}")