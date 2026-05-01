import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from src.config import get_config
from src.preprocess.md_filter import filt_and_save
from src.prompts.chunking_prompt import (
    md2text_prompt, boundary_prompt, agentic_prompt, decision_prompt, meta_prompt
)
from src.preprocess.chunking import (
    html2text, parse_json,
    decision, lumber_chunking, agentic_chunking
)


def collect_md_files(md_dir: Path) -> list[Path]:
    """재귀적으로 모든 하위 폴더의 .md 파일 수집 (정렬 포함)"""
    files = sorted(md_dir.rglob("*.md"))
    print(f"[INFO] 탐색 경로: {md_dir}")
    for f in files:
        print(f"  발견: {f.relative_to(md_dir)}")
    return files

def full_pipeline(
    md_dir: Path | list[Path],
    output_dir: Path,
    accumulate_pages: int = 3,
) -> list[dict]:
    # ── 리스트면 각 디렉토리마다 재귀 호출 ──────────────────────
    if isinstance(md_dir, list):
        all_chunks = []
        for d in md_dir:
            print(f"\n{'='*60}")
            print(f"[INFO] 디렉토리 처리 시작: {d}")
            print(f"{'='*60}")
            all_chunks.extend(full_pipeline(d, output_dir, accumulate_pages))
        print(f"\n[INFO] 전체 완료 — 총 {len(all_chunks)}개 청크")
        return all_chunks

    md_dir = Path(md_dir)
    output_dir = Path(output_dir)

    md_files = collect_md_files(md_dir)
    print(f"\n[INFO] 총 {len(md_files)}개 md 파일 발견")

    llm = ChatOpenAI(model="gpt-4o-mini")
    prose_chain    = md2text_prompt  | llm
    boundary_chain = boundary_prompt | llm
    meta_chain     = meta_prompt     | llm
    agentic_chain  = agentic_prompt  | llm
    decision_chain = decision_prompt | llm

    # ── 1단계: Decision + 줄글 변환 ──────────────────────────────
    print("\n[INFO] 1단계: Decision 필터링 + 줄글 변환 중...")
    prose_pages = []
    for i, md_file in enumerate(md_files):
        rel = md_file.relative_to(md_dir)
        print(f"  처리중: {rel} ({i+1}/{len(md_files)})")

        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            print(f"  → [SKIP] 빈 페이지")
            continue

        raw_text = html2text(content)
        if not decision(raw_text, decision_chain):
            print(f"  → [SKIP] 쓸모없는 페이지")
            continue

        response = prose_chain.invoke({"page_text": raw_text})
        prose = response.content.strip()
        if prose:
            prose_pages.append((prose, md_file))
            print(f"  → [PASS] 줄글 변환 완료")

    print(f"\n[INFO] {len(prose_pages)}개 페이지 통과")

    prose_texts = [p for p, _ in prose_pages]
    prose_files = [f for _, f in prose_pages]

    # ── 2단계: Lumber Chunking ────────────────────────────────────
    print("\n[INFO] 2단계: Lumber Chunking 중...")
    lumber_chunks = lumber_chunking(prose_texts, boundary_chain, accumulate_pages)
    print(f"[INFO] {len(lumber_chunks)}개 큰 청크 생성")

    # ── 3단계: Agentic Chunking ───────────────────────────────────
    print("\n[INFO] 3단계: Agentic Chunking 중...")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_idx  = 0

    for i, lumber_chunk in enumerate(lumber_chunks):
        page_start = i * accumulate_pages
        rep_file   = prose_files[page_start] if page_start < len(prose_files) else prose_files[-1]
        rel_subdir = rep_file.parent.relative_to(md_dir)

        print(f"\n  큰 청크 {i+1}/{len(lumber_chunks)} 처리중... (출처: {rel_subdir})")

        agentic_chunks, md_summary = agentic_chunking(lumber_chunk, meta_chain, agentic_chain)
        print(f"  → {len(agentic_chunks)}개 agentic chunk 생성")

        sub_output_dir = output_dir / rel_subdir
        sub_output_dir.mkdir(parents=True, exist_ok=True)

        for chunk in agentic_chunks:
            record = {
                "id":            chunk_idx,
                "source_file":   rep_file.name,
                "md_summary":    md_summary,
                "raw_chunk":     chunk["raw_chunk"],
                "agentic_chunk": chunk["agentic_chunk"],
            }
            all_chunks.append(record)

            filename = f"{md_dir.name}_chunk{chunk_idx:03d}.json"
            with open(sub_output_dir / filename, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            chunk_idx += 1

    print(f"\n[INFO] {len(all_chunks)}개 청크 저장 완료 → {output_dir}")
    return all_chunks

#########  파라미터 로드 #######################
CFG = get_config("configs/config_preproc.yaml")

MD_DIRS    = [Path(d) for d in CFG.md_in_dirs]
CHUNK_DIR  = Path(CFG.chunk_out_dir)

load_dotenv()

llm            = ChatOpenAI(model="gpt-4o-mini")
meta_chain     = meta_prompt     | llm
chunking_chain = agentic_prompt  | llm
decision_chain = decision_prompt | llm
###############################################


if __name__ == "__main__":
    load_dotenv()

    full_pipeline(
        md_dir=MD_DIRS, 
        output_dir=CHUNK_DIR,
        accumulate_pages=1,
    )