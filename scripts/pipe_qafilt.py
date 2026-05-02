from __future__ import annotations

import json
from pathlib import Path
from dotenv import load_dotenv
from itertools import groupby

from src.config import get_config
from src.postprocess.nli_filter import NLIFilter

def run_nli_filter(qa2d_results: list[dict], nli: NLIFilter, threshold: float) -> tuple[list[dict], list[dict]]:
    """3단계: NLI 기반 entailment 판단 → (전체, 필터링된 것) 반환"""
    print(f"[INFO] NLI 필터링 중... (threshold={threshold})")
    all_results, _ = nli.filter_batch(qa2d_results)
    filtered = [
        r for r in all_results
        if r["label"] == "entailment" and r["score"] >= threshold
    ]
    print(f"[INFO] 필터링 결과: {len(filtered)}/{len(all_results)} "
          f"({len(filtered)/max(len(all_results), 1)*100:.1f}%)")
    return all_results, filtered

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
CFG = get_config("configs/config_gen.yaml")

QA_ROOT          = Path(CFG.qa_in_dir)
OUT_ROOT         = Path(CFG.qa_filt_dir)
NLI_MODEL        = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
ENTAIL_THRESHOLD = 0.95
###################################################

if __name__ == "__main__":
    load_dotenv()

    nli = NLIFilter(model_name=NLI_MODEL, device="cuda")

    qa_list = load_qa_from_dir(QA_ROOT)

    # chunk_root는 all/ 의 상위인 qa_out_dir 기준
    chunk_root = QA_ROOT

    all_results, filtered = run_nli_filter(qa_list, nli, ENTAIL_THRESHOLD)

    # 통과된 것 → selected
    save_qa(filtered, chunk_root, OUT_ROOT, suffix="selected")
    print(f"[INFO] 통과 QA {len(filtered)}개 저장 완료 → selected")

    # 실패한 것 → rejected
    rejected = [r for r in all_results if r not in filtered]
    save_qa(rejected, chunk_root, OUT_ROOT, suffix="rejected")
    print(f"[INFO] 실패 QA {len(rejected)}개 저장 완료 → rejected")

    print(f"\n[INFO] 완료 → {OUT_ROOT}")