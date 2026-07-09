import json, re
from pathlib import Path
from dotenv import load_dotenv
from itertools import groupby

from langchain_openai import ChatOpenAI

from src.config import get_config
from src.prompts.chunking_prompt import (
    md2text_prompt, boundary_prompt, decision_prompt, meta_prompt, agentic_prompt
)
from src.preprocess.chunking import (
    parse_json,
    decision, decision_batch, lumber_chunking,
    agentic_chunking, agentic_chunking_batch,
    fixed_size_chunking, semantic_chunking, split_sentences
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


def full_pipeline(
    md_dir: Path | list[Path],
    output_dir: Path,
    accumulate_pages: int = 3,
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
            all_chunks.extend(full_pipeline(d, output_dir, accumulate_pages))
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

    # ── 1단계: Decision + 줄글 변환 ──────────────────────────────
    print("\n[INFO] 1단계: Decision 필터링 + 줄글 변환 중...")
    page_items = []
    for i, md_file in enumerate(md_files):
        rel = md_file.relative_to(md_root)
        print(f"  처리중: {rel} ({i+1}/{len(md_files)})")

        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            print(f"  → [SKIP] 빈 페이지")
            continue

        # raw_text = html2text(content)
        raw_text = content
        page_items.append((md_file, raw_text))

    raw_texts = [raw_text for _, raw_text in page_items]
    useful_flags = decision_batch(raw_texts, decision_chain)

    useful_items = [
        (md_file, raw_text)
        for (md_file, raw_text), is_useful in zip(page_items, useful_flags)
        if is_useful
    ]

    print(f"\n[INFO] Decision 통과: {len(useful_items)}/{len(page_items)}")

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
        batch_results = agentic_chunking_batch(lumber_texts, meta_chain, agentic_chain)
        chunk_idx = 1

        for i, ((lumber_chunk, start_idx, end_idx), (md_summary, chunk_texts)) in enumerate(zip(lumber_chunks, batch_results)):
            rep_file   = group_files[start_idx]
            end_file   = group_files[end_idx]
            rel_subdir = rep_file.parent.relative_to(md_root)

            source_files = list(dict.fromkeys(
                f.name for f in group_files[start_idx:end_idx+1]
            ))

            print(f"\n    큰 청크 {i+1}/{len(lumber_chunks)} 처리중... source_files: {source_files}")
            print(f"    → {len(chunk_texts)}개 agentic chunk 생성")

            sub_output_dir = output_dir / rel_subdir
            sub_output_dir.mkdir(parents=True, exist_ok=True)

            for chunk_text in chunk_texts:
                record = {
                    "id":           chunk_idx,
                    "source_files": source_files,
                    "md_summary":   md_summary,
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


#########  파라미터  #######################
CFG = get_config("configs/config_gen.yaml")

MD_DIRS   = [Path(d) for d in CFG.md_in_dirs]
CHUNK_DIR = Path(CFG.chunk_out_dir)

load_dotenv()

llm            = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
prose_chain    = md2text_prompt  | llm
boundary_chain = boundary_prompt | llm
meta_chain     = meta_prompt     | llm
agentic_chain  = agentic_prompt  | llm
decision_chain = decision_prompt | llm
###############################################

if __name__ == "__main__":
    load_dotenv()

    full_pipeline(
        md_dir=MD_DIRS,
        output_dir=CHUNK_DIR,
        accumulate_pages=1,
    )