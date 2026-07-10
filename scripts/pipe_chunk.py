import json, re
from pathlib import Path
from dotenv import load_dotenv
from itertools import groupby

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_config
from src.prompts.chunking_prompt import (
    md2text_prompt, boundary_prompt, decision_prompt, meta_prompt, agentic_prompt
)
from src.preprocess.chunking import (
    parse_json,
    decision, decision_batch, lumber_chunking,
    agentic_chunking, agentic_chunking_batch,
    recursive_chunking, fixed_size_chunking, semantic_chunking, split_sentences
)


def collect_md_files(md_dir: Path) -> list[Path]:
    def natural_key(path: Path):
        parts = re.split(r'(\d+)', path.stem)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    files = sorted(md_dir.rglob("*.md"), key=natural_key)
    print(f"[INFO] 탐색 경로: {md_dir}")
    for f in files:
        print(f"  발견: {f.relative_to(md_dir)}")
    return files


def collect_all_md_files(md_dirs: list[Path]) -> list[Path]:
    """MD_DIRS 전체(파일/폴더 혼합)를 순회하며 md 파일 목록을 모은다."""
    all_files = []
    for d in md_dirs:
        d = Path(d)
        if d.is_file() and d.suffix == ".md":
            all_files.append(d)
        elif d.is_dir():
            all_files.extend(collect_md_files(d))
        else:
            print(f"[WARN] 건너뜀 (파일/폴더 아님): {d}")
    return all_files


# 같은 디렉토리에 대해 Decision 필터링을 여러 번 반복하지 않도록 캐싱한다.
# (ours / lumber_no_contextual 등 use_decision_filter=True인 여러 버전이
#  동일한 MD_DIRS를 공유하므로, 필터링 결과를 재사용해 LLM 호출을 줄인다)
_FILTER_CACHE: dict[Path, list[tuple[Path, str]]] = {}


def get_filtered_pages(cache_key: Path, page_items: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    if cache_key in _FILTER_CACHE:
        cached = _FILTER_CACHE[cache_key]
        print(f"[INFO] 필터링 캐시 재사용: {cache_key} ({len(cached)}개 페이지)")
        return cached

    raw_texts = [t for _, t in page_items]
    useful_flags = decision_batch(raw_texts, decision_chain)
    useful_items = [(f, t) for (f, t), ok in zip(page_items, useful_flags) if ok]
    _FILTER_CACHE[cache_key] = useful_items
    return useful_items


def full_pipeline(
    md_dir: Path | list[Path],
    output_dir: Path,
    accumulate_pages: int = 3,
    use_decision_filter: bool = True,
    use_contextual_summary: bool = True,
    use_agentic_chunking: bool = True,
) -> list[dict]:

    if isinstance(md_dir, list):
        all_chunks = []
        for d in md_dir:
            d = Path(d)
            print(f"\n{'='*60}")
            if d.is_file() and d.suffix == ".md":
                print(f"[INFO] 파일 처리 시작: {d}")
            elif d.is_dir():
                print(f"[INFO] 디렉토리 처리 시작: {d}")
            else:
                print(f"[WARN] 건너뜀 (파일/폴더 아님): {d}")
                continue
            print(f"{'='*60}")
            all_chunks.extend(full_pipeline(
                d, output_dir, accumulate_pages,
                use_decision_filter, use_contextual_summary, use_agentic_chunking,
            ))
        print(f"\n[INFO] 전체 완료 — 총 {len(all_chunks)}개 청크")
        return all_chunks

    md_dir = Path(md_dir)
    output_dir = Path(output_dir)

    if md_dir.is_file() and md_dir.suffix == ".md":
        md_files = [md_dir]
        md_root  = md_dir.parent
        print(f"[INFO] 단일 파일 모드: {md_dir}")
    else:
        md_files = collect_md_files(md_dir)
        md_root  = md_dir
        print(f"\n[INFO] 총 {len(md_files)}개 md 파일 발견")

    # ── 1단계: Decision 필터링(조건부) + 줄글 변환 ──────────────
    print("\n[INFO] 1단계: Decision 필터링 + 줄글 변환 중...")
    page_items = []
    for i, md_file in enumerate(md_files):
        rel = md_file.relative_to(md_root)
        print(f"  처리중: {rel} ({i+1}/{len(md_files)})")

        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            print(f"  → [SKIP] 빈 페이지")
            continue

        raw_text = content
        page_items.append((md_file, raw_text))

    if use_decision_filter:
        useful_items = get_filtered_pages(md_dir, page_items)
        print(f"\n[INFO] Decision 통과: {len(useful_items)}/{len(page_items)}")
    else:
        useful_items = page_items
        print(f"\n[INFO] Decision 필터링 skip — 전체 {len(useful_items)}페이지 사용")

    prose_pages = []
    if useful_items:
        prose_inputs = [{"page_text": raw_text} for _, raw_text in useful_items]
        prose_responses = prose_chain.batch(prose_inputs)

        for (md_file, _), response in zip(useful_items, prose_responses):
            prose = response.content.strip()
            if prose:
                prose_pages.append((prose, md_file))
                print(f"  → [PASS] 줄글 변환 완료: {md_file.name}")

    print(f"\n[INFO] {len(prose_pages)}개 페이지 통과")

    # ── 2단계 & 3단계: 폴더별로 분리해서 처리 ───────────────────
    print("\n[INFO] 2단계 & 3단계: 폴더별 Lumber + Agentic Chunking 중...")

    def get_base(md_file: Path) -> str:
        rel   = md_file.relative_to(md_root)
        parts = rel.parts
        return parts[0] if len(parts) > 1 else md_root.name

    prose_pages_sorted = sorted(prose_pages, key=lambda x: get_base(x[1]))
    all_chunks = []

    for base_name, group in groupby(prose_pages_sorted, key=lambda x: get_base(x[1])):
        group_pages = list(group)
        group_texts = [p for p, _ in group_pages]
        group_files = [f for _, f in group_pages]

        print(f"\n  [{base_name}] {len(group_texts)}개 페이지 → Lumber Chunking")
        lumber_chunks = lumber_chunking(group_texts, boundary_chain, accumulate_pages)
        print(f"  [{base_name}] {len(lumber_chunks)}개 큰 청크 생성")

        lumber_texts = [chunk for chunk, _, _ in lumber_chunks]

        if use_agentic_chunking:
            batch_results = agentic_chunking_batch(lumber_texts, meta_chain, agentic_chain)
        else:
            # Agentic chunking skip — LumberChunker가 만든 큰 청크를 그대로 raw_chunk로 사용
            batch_results = [("", [lumber_text]) for lumber_text in lumber_texts]

        chunk_idx = 1

        for i, ((lumber_chunk, start_idx, end_idx), (md_summary, chunk_texts)) in enumerate(zip(lumber_chunks, batch_results)):
            rep_file   = group_files[start_idx]
            end_file   = group_files[end_idx]
            rel_subdir = rep_file.parent.relative_to(md_root)

            source_files = list(dict.fromkeys(
                f.name for f in group_files[start_idx:end_idx+1]
            ))

            print(f"\n    큰 청크 {i+1}/{len(lumber_chunks)} 처리중... source_files: {source_files}")
            if use_agentic_chunking:
                print(f"    → {len(chunk_texts)}개 agentic chunk 생성")
            else:
                print(f"    → agentic chunking skip, LumberChunker 청크 그대로 사용")

            sub_output_dir = output_dir / rel_subdir
            sub_output_dir.mkdir(parents=True, exist_ok=True)

            for chunk_text in chunk_texts:
                record = {
                    "id":           chunk_idx,
                    "source_files": source_files,
                    "md_summary":   md_summary if use_contextual_summary else None,
                    "raw_chunk":    chunk_text,
                }
                all_chunks.append(record)

                if rep_file.name == end_file.name:
                    filename = f"{rep_file.stem}_chunk{chunk_idx:03d}.json"
                else:
                    end_num    = re.search(r'(\d+)$', end_file.stem)
                    end_suffix = end_num.group(1) if end_num else end_file.stem
                    filename   = f"{rep_file.stem}_to_{end_suffix}_chunk{chunk_idx:03d}.json"

                with open(sub_output_dir / filename, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)

                chunk_idx += 1

    print(f"\n[INFO] {len(all_chunks)}개 청크 저장 완료 → {output_dir}")
    return all_chunks


def run_baseline(
    md_files: list[Path],
    output_dir: Path,
    chunk_fn,
    tag: str,
    filtered_items: list[tuple[Path, str]] | None = None,
    **kwargs,
) -> list[dict]:
    """Recursive / Fixed-size / Semantic 등 non-LLM 청킹 baseline.

    ours와 동일하게 Decision 필터링은 거치되, 자연어 변환(prose_chain)은 생략하고
    OCR 원본 markdown을 그대로 청킹한다.

    filtered_items를 미리 계산해서 넘기면 Decision 필터링(LLM 호출)을
    다시 수행하지 않고 그대로 재사용한다. recursive/fixed/semantic처럼
    동일한 md_files에 대해 여러 baseline을 돌릴 때 중복 호출을 막기 위함.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filtered_items is None:
        items = [(f, f.read_text(encoding="utf-8")) for f in md_files]
        items = [(f, t) for f, t in items if t.strip()]
        texts = [t for _, t in items]
        flags = decision_batch(texts, decision_chain)
        useful_items = [(f, t) for (f, t), ok in zip(items, flags) if ok]
        print(f"[INFO] [{tag}] Decision 통과: {len(useful_items)}/{len(items)}")
    else:
        useful_items = filtered_items
        print(f"[INFO] [{tag}] 필터링 캐시 재사용: {len(useful_items)}개 페이지")

    all_chunks = []
    idx = 1
    for f, text in useful_items:
        chunks = chunk_fn(text, **kwargs)
        for chunk in chunks:
            record = {
                "id": idx,
                "source_files": [f.name],
                "md_summary": None,
                "raw_chunk": chunk,
            }
            all_chunks.append(record)
            with open(output_dir / f"{tag}_{idx:03d}.json", "w", encoding="utf-8") as out:
                json.dump(record, out, ensure_ascii=False, indent=2)
            idx += 1

    print(f"[INFO] [{tag}] {len(all_chunks)}개 청크 저장 완료 → {output_dir}")
    return all_chunks


#########  파라미터  #######################
CFG = get_config("configs/config_gen.yaml")

MD_DIRS   = [Path(d) for d in CFG.md_in_dirs]
CHUNK_DIR = Path(CFG.chunk_out_dir)

load_dotenv()

llm            = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
embed_model    = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},   # GPU 있으면 "cuda"
    encode_kwargs={"normalize_embeddings": True},
)
prose_chain    = md2text_prompt  | llm
boundary_chain = boundary_prompt | llm
meta_chain     = meta_prompt     | llm
agentic_chain  = agentic_prompt  | llm
decision_chain = decision_prompt | llm
###############################################


if __name__ == "__main__":
    load_dotenv()

    # ── Table 5: Filter ablation ──
    # full_pipeline(
    #     md_dir=MD_DIRS, output_dir=CHUNK_DIR / "ours",
    #     accumulate_pages=1,
    #     use_decision_filter=True, use_contextual_summary=True,
    # )

    md_files = collect_all_md_files(MD_DIRS)

    # recursive/fixed/semantic이 동일한 md_files를 쓰므로
    # Decision 필터링(LLM 호출)을 여기서 한 번만 수행하고 재사용한다.
    _items = [(f, f.read_text(encoding="utf-8")) for f in md_files]
    _items = [(f, t) for f, t in _items if t.strip()]
    _texts = [t for _, t in _items]
    _flags = decision_batch(_texts, decision_chain)
    baseline_filtered_items = [(f, t) for (f, t), ok in zip(_items, _flags) if ok]
    print(f"[INFO] Baseline 공통 필터링: {len(baseline_filtered_items)}/{len(_items)}")

    run_baseline(
        md_files, CHUNK_DIR / "recursive", recursive_chunking, "recursive",
        filtered_items=baseline_filtered_items,
        chunk_size=512, overlap=50,
    )
    run_baseline(
        md_files, CHUNK_DIR / "fixed", fixed_size_chunking, "fixed",
        filtered_items=baseline_filtered_items,
        chunk_size=512, overlap=50,
    )
    run_baseline(
        md_files, CHUNK_DIR / "semantic",
        lambda t: semantic_chunking(split_sentences(t), embed_model), "semantic",
        filtered_items=baseline_filtered_items,
    )

    full_pipeline(
        md_dir=MD_DIRS, output_dir=CHUNK_DIR / "ours_no_filter",
        accumulate_pages=1,
        use_decision_filter=False, use_contextual_summary=True,
    )

    # ── Table 6: Chunking method comparison ──
    # LumberChunker(경계 판단)만 사용, Agentic 재분할과 Contextual summary 둘 다 제거
    full_pipeline(
        md_dir=MD_DIRS, output_dir=CHUNK_DIR / "lumber_no_contextual",
        accumulate_pages=1,
        use_decision_filter=True, use_contextual_summary=False, use_agentic_chunking=False,
    )