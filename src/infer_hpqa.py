import json, time, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

from retriever import MultiCosineRetriever, load_retrievers
from models_RAG import HotpotExpert  # ✅ Hotpot 전용


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

    # ✅ answer 키가 항상 있다고 가정하되, 혹시 몰라 fallback도 둠
    def _get_answer(it: dict) -> str:
        if "answer" in it:
            return it.get("answer", "") or ""
        # 과거 포맷 호환
        return it.get("reference", "") or ""

    return {
        "qa_ids": [it.get("id") for it in items],
        "questions": [it.get("question", "") for it in items],
        "references": [_get_answer(it) for it in items],  # ✅ 평가 코드에서 reference로 씀
        "options": [[] for _ in items],                   # ✅ hotpot 기본: options 없음
    }


# =============================================================================
# 3) Hotpot-only Runner
# =============================================================================

def run_hotpot(cfg: Cfg, results_dir: Path, bus: Optional[EventBus] = None) -> None:
    bus = bus or EventBus()

    # (1) retriever 구축
    bus.emit("stage", msg="[BUILD] loading retrievers...")
    leaf = load_retrievers(ndocs=cfg.ndocs_init)
    retr = MultiCosineRetriever(retrievers=leaf, k_each=cfg.k_each, top_k=cfg.top_k)
    retr_map = {cfg.retriever_mode: retr}

    # (2) GT 로드
    gt = load_hotpot_gt(cfg.gt_path, cfg.sample_size)
    qa_ids, questions, references, options = gt["qa_ids"], gt["questions"], gt["references"], gt["options"]
    bus.emit("stage", msg=f"[DATA] loaded Hotpot GT: n={len(qa_ids)}")

    # (3) HotpotExpert (mode 고정)
    expert = HotpotExpert(retr_map, cfg.retriever_mode)

    # (4) output dir
    qag_root = results_dir / "inferenced" / cfg.subdir
    ret_root = results_dir / "retrieved" / cfg.subdir
    qag_root.mkdir(parents=True, exist_ok=True)
    ret_root.mkdir(parents=True, exist_ok=True)

    # ✅ 평가 코드와 파일명 호환: qag_multihop.json / retrieved_multihop.json
    qag_out = qag_root / "qag_multihop.json"
    ret_out = ret_root / "retrieved_multihop.json"

    bus.emit("stage", msg="[RUN] multihop start")

    qag_records = []
    ret_records = []

    for qid, q, opts, ref in tqdm(
        list(zip(qa_ids, questions, options, references)),
        desc="Generating(multihop)"
    ):
        bus.emit("q", msg=f"[QID={qid}] ...")

        res = expert.handle(q, options=opts)

        # generated + info
        if isinstance(res, tuple):
            ans, info = res
        else:
            ans, info = res, None

        if hasattr(ans, "content"):
            ans = ans.content
        elif isinstance(ans, dict):
            ans = ans.get("content") or ans.get("text") or ans.get("answer") or str(ans)
        ans = "" if ans is None else str(ans)

        # ✅ 평가 코드(QAGRepo)와 호환: id/question/reference/generated/info
        qag_records.append({
            "id": qid,
            "question": q,
            "reference": ref,   # GT answer를 그대로 넣어 둠(평가에서 사용하진 않아도 디버깅에 좋음)
            "generated": ans,
            "info": info or {},
        })

        # retrieved
        docs = getattr(expert, "retrieved_snippets", None) or getattr(expert, "last_retrieved_docs", [])
        retrieved_list = []
        for d in (docs or []):
            retrieved_list.append({
                "rank": d.metadata.get("__rank__"),
                "filename": d.metadata.get("filename"),
                "content": make_pretty(d.page_content),
            })
        ret_records.append({"id": qid, "retrieved": retrieved_list})

    # save
    with open(qag_out, "w", encoding="utf-8") as f:
        json.dump(qag_records, f, ensure_ascii=False, indent=2)
    with open(ret_out, "w", encoding="utf-8") as f:
        json.dump(ret_records, f, ensure_ascii=False, indent=2)

    bus.emit("stage", msg=f"[RUN] saved -> {qag_out} / {ret_out}")


# =============================================================================
# 4) MAIN
# =============================================================================

if __name__ == "__main__":
    bus = EventBus()
    bus.on("stage", log_msg)
    bus.on("q", log_msg)   # 질문별 로그 끄려면 삭제

    results_dir = (Path(__file__).resolve().parent / ".." / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = Cfg(
        gt_path="qa_data/GT/hotpotqa_test/gt_merged_hotpotqa_test.json",  # ✅ 확장자 포함
        retriever_mode="cleaned_multi",
        ndocs_init=10,
        k_each=10,
        top_k=7,
        sample_size=None,
        subdir="hotpotqa_test",  # ✅ 평가 cfg_hotpot의 경로와 맞추는 걸 추천
    )

    start = time.time()
    run_hotpot(cfg, results_dir=results_dir, bus=bus)
    print(f"⏱ 전체 추론 완료: {(time.time()-start)/60:.2f}분")
