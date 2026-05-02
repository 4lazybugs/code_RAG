import json, re
from pathlib import Path
from dotenv import load_dotenv
from itertools import groupby

from langchain_openai import ChatOpenAI

from src.config import get_config
from src.prompts.chunking_prompt import (
    md2text_prompt, boundary_prompt, agentic_prompt, decision_prompt, meta_prompt
)
from src.preprocess.chunking import (
    html2text, parse_json,
    decision, lumber_chunking, agentic_chunking
)


def collect_md_files(md_dir: Path) -> list[Path]:
    """재귀적으로 모든 하위 폴더의 .md 파일 수집 (숫자 순 정렬)"""
    def natural_key(path: Path):
        parts = re.split(r'(\d+)', path.stem)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    files = sorted(md_dir.rglob("*.md"), key=natural_key)
    print(f"[INFO] 탐색 경로: {md_dir}")
    for f in files:
        print(f"  발견: {f.relative_to(md_dir)}")
    return files


def full_pipeline(
    md_dir: Path | list[Path],
    output_dir: Path,
    accumulate_pages: int = 3,
) -> list[dict]:

    # ── 리스트면 각 항목마다 재귀 호출 (파일/폴더 혼합 OK) ──────
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
            all_chunks.extend(full_pipeline(d, output_dir, accumulate_pages))
        print(f"\n[INFO] 전체 완료 — 총 {len(all_chunks)}개 청크")
        return all_chunks

    md_dir = Path(md_dir)
    output_dir = Path(output_dir)

    # ── 단일 .md 파일이면 해당 파일만 처리 ──────────────────────
    if md_dir.is_file() and md_dir.suffix == ".md":
        md_files = [md_dir]
        md_root  = md_dir.parent
        print(f"[INFO] 단일 파일 모드: {md_dir}")
    else:
        md_files = collect_md_files(md_dir)
        md_root  = md_dir
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
        rel = md_file.relative_to(md_root)
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

        chunk_idx = 1  # 폴더마다 초기화

        for i, lumber_chunk in enumerate(lumber_chunks):
            page_start = i * accumulate_pages
            rep_file   = group_files[page_start] if page_start < len(group_files) else group_files[-1]
            rel_subdir = rep_file.parent.relative_to(md_root)

            print(f"\n    큰 청크 {i+1}/{len(lumber_chunks)} 처리중... (출처: {rel_subdir})")

            agentic_chunks, md_summary = agentic_chunking(lumber_chunk, meta_chain, agentic_chain)
            print(f"    → {len(agentic_chunks)}개 agentic chunk 생성")

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

                filename = f"{base_name}_chunk{chunk_idx:03d}.json"
                with open(sub_output_dir / filename, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)

                chunk_idx += 1

    print(f"\n[INFO] {len(all_chunks)}개 청크 저장 완료 → {output_dir}")
    return all_chunks


#########  파라미터 로드 #######################
CFG = get_config("configs/config_preproc.yaml")

MD_DIRS   = [Path(d) for d in CFG.md_in_dirs]
CHUNK_DIR = Path(CFG.chunk_out_dir)

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