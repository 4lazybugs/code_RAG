import json, time, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

from retriever import MultiCosineRetriever, load_retrievers
from models_RAG import HotpotExpert  # ✅ Hotpot 전용

# ✅ naive expert import (네 프로젝트 실제 경로에 맞게 수정)
# 예시1) from experts.raw_llm import RawLlmExpert
# 예시2) from models_RAG.raw_llm_expert import RawLlmExpert
from models_RAG import RawLlmExpert


# =============================================================================
# 0) Utils
# =============================================================================

_RE_TR = re.compile(r"<tr.*?>.*?</tr>", flags=re.DOTALL | re.IGNORECASE)

def make_pretty(text: str, max_block_chars: int = 1200) -> list[str]:
    """짧은 버전: <tr> 있으면 tr 단위, 없으면 빈줄 문단 단위. 블록 길이만 제한."""
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return []
    trs = _RE_TR.findall(s)
    blocks = trs if trs else re.split(r"\n\s*\n+", s)
    out = []
    for b in blocks:
        b = "\n".join(ln.strip() for ln in b.splitlines() if ln.strip()).strip()
        if not b:
            continue
        if len(b) > max_block_chars:
            b = b[:max_block_chars] + "\n...(truncated)"
        out.append(b)
    return out

def to_plain_text(x: Any) -> str:
    """LLM/체인 응답을 문자열로 안전 변환."""
    if x is None:
        return ""
    if hasattr(x, "content"):
        return "" if x.content is None else str(x.content)
    if isinstance(x, dict):
        return str(x.get("content") or x.get("text") or x.get("answer") or x.get("output") or "")
    return str(x)


# =============================================================================
# 1) Observer (EventBus) - 최소형
# =============================================================================

class EventBus:
    def __init__(self):
        self._subs: dict[str, list[Callable[..., None]]] = {}

    def on(self, event: str, fn: Callable[..., None]) -> None:
        self._subs.setdefault(event, []).append(fn)

    def emit(self, event: str, **payload) -> None:
        for fn in self._subs.get(event, []):
            fn(**payload)

def log_msg(**payload):
    msg = payload.get("msg")
    if msg:
        print(msg)


# =============================================================================
# 2) Config + Hotpot GT loader
# =============================================================================

@dataclass
class Cfg:
    gt_path: str
    retriever_mode: str = "cleaned_multi"
    ndocs_init: int = 10
    k_each: int = 10
    top_k: int = 7
    sample_size: Optional[int] = None
    subdir: str = "hotpotqa_test"  # ✅ hotpot 전용 기본값


def load_hotpot_gt(path: str, sample_size: Optional[int] = None) -> dict[str, list]:
    """
    Hotpot GT 형식:
    [
      { "id":..., "question":..., "answer":..., ... }
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if sample_size is not None:
        items = items[: int(sample_size)]

    def _get_answer(it: dict) -> str:
        if "answer" in it:
            return it.get("answer", "") or ""
        return it.get("reference", "") or ""

    return {
        "qa_ids": [it.get("id") for it in items],
        "questions": [it.get("question", "") for it in items],
        "references": [_get_answer(it) for it in items],
        "options": [[] for _ in items],  # ✅ hotpot 기본: options 없음
    }


# =============================================================================
# 3) Hotpot Runner (multihop + naive)
# =============================================================================

def run_hotpot(cfg: Cfg, results_dir: Path, bus: Optional[EventBus] = None) -> None:
    bus = bus or EventBus()

    # (1) retriever 구축 (multihop 전용)
    bus.emit("stage", msg="[BUILD] loading retrievers...")
    leaf = load_retrievers(ndocs=cfg.ndocs_init)
    retr = MultiCosineRetriever(retrievers=leaf, k_each=cfg.k_each, top_k=cfg.top_k)
    retr_map = {cfg.retriever_mode: retr}

    # (2) GT 로드 (multihop/naive 공통)
    gt = load_hotpot_gt(cfg.gt_path, cfg.sample_size)
    qa_ids, questions, references, options = gt["qa_ids"], gt["questions"], gt["references"], gt["options"]
    bus.emit("stage", msg=f"[DATA] loaded Hotpot GT: n={len(qa_ids)}")

    # (3) Experts
    # 3-1) multihop expert
    mh_expert = HotpotExpert(retr_map, cfg.retriever_mode)

    # 3-2) naive expert (retrieval 없이)
    naive_expert = RawLlmExpert(retriever_map={}, retriever_mode="naive")
    naive_expert.setup(retriever_mode="naive")

    # (4) output dir
    qag_root = results_dir / "inferenced" / cfg.subdir
    ret_root = results_dir / "retrieved" / cfg.subdir
    qag_root.mkdir(parents=True, exist_ok=True)
    ret_root.mkdir(parents=True, exist_ok=True)

    # ✅ 평가 코드와 파일명/경로 호환
    qag_out_multihop = qag_root / "qag_multihop.json"
    ret_out_multihop = ret_root / "retrieved_multihop.json"
    qag_out_naive = qag_root / "qag_naive.json"

    bus.emit("stage", msg="[RUN] multihop + naive start")

    qag_mh_records = []
    ret_mh_records = []
    qag_naive_records = []

    for qid, q, opts, ref in tqdm(
        list(zip(qa_ids, questions, options, references)),
        desc="Generating(multihop+naive)"
    ):
        bus.emit("q", msg=f"[QID={qid}] ...")

        # ------------------------------------------------------------
        # (A) multihop 추론
        # ------------------------------------------------------------
        res = mh_expert.handle(q, options=opts)

        if isinstance(res, tuple):
            ans_mh, info_mh = res
        else:
            ans_mh, info_mh = res, None

        ans_mh = to_plain_text(ans_mh)

        qag_mh_records.append({
            "id": qid,
            "question": q,
            "reference": ref,
            "generated": ans_mh,
            "info": info_mh or {},
        })

        # retrieved(multihop만)
        docs = getattr(mh_expert, "retrieved_snippets", None) or getattr(mh_expert, "last_retrieved_docs", [])
        retrieved_list = []
        for rank, d in enumerate(docs or [], start=1):
            md = getattr(d, "metadata", {}) or {}
            retrieved_list.append({
                "rank": md.get("__rank__", rank),   # rank 없으면 enumerate로 보정
                "filename": md.get("filename"),
                "content": make_pretty(getattr(d, "page_content", "")),
            })
        ret_mh_records.append({"id": qid, "retrieved": retrieved_list})

        # ------------------------------------------------------------
        # (B) naive 추론 (GT 동일, question만 사용)
        # ------------------------------------------------------------
        ans_nv = naive_expert.handle(q, options=None)
        ans_nv = to_plain_text(ans_nv)

        qag_naive_records.append({
            "id": qid,
            "question": q,
            "reference": ref,
            "generated": ans_nv,
            "info": {},  # naive는 supporting/link_word 없음
        })

    # save
    with open(qag_out_multihop, "w", encoding="utf-8") as f:
        json.dump(qag_mh_records, f, ensure_ascii=False, indent=2)
    with open(ret_out_multihop, "w", encoding="utf-8") as f:
        json.dump(ret_mh_records, f, ensure_ascii=False, indent=2)
    with open(qag_out_naive, "w", encoding="utf-8") as f:
        json.dump(qag_naive_records, f, ensure_ascii=False, indent=2)

    bus.emit(
        "stage",
        msg=(
            "[RUN] saved ->\n"
            f"- {qag_out_multihop}\n"
            f"- {ret_out_multihop}\n"
            f"- {qag_out_naive}"
        )
    )


# =============================================================================
# 4) MAIN
# =============================================================================

if __name__ == "__main__":
    bus = EventBus()
    bus.on("stage", log_msg)
    bus.on("q", log_msg)   # 질문별 로그 끄려면 삭제

    results_dir = (Path(__file__).resolve().parent / ".." / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ✅ gt_path는 multihop/naive 공통으로 동일 파일 사용
    cfg = Cfg(
        gt_path="qa_data/GT/hotpotqa_test/gt_merged_hotpotqa_test.json",
        retriever_mode="cleaned_multi",
        ndocs_init=10,
        k_each=10,
        top_k=7,
        sample_size=None,
        subdir="hotpotqa_test",
    )

    start = time.time()
    run_hotpot(cfg, results_dir=results_dir, bus=bus)
    print(f"⏱ 전체 추론 완료: {(time.time()-start)/60:.2f}분")
