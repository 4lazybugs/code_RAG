from pathlib import Path
import time
from typing import Optional
from typing import List, Tuple

# PaddleOCR import
import fitz  # pymupdf
import numpy as np
from PIL import Image
from paddleocr import PaddleOCRVL, PPStructureV3

from pathlib import Path
from src.preprocess.md_utils import collect_md_files, get_filtered_pages

# Docling imports: https://docling-project.github.io/docling/
#from docling.document_converter import DocumentConverter

def pdf_to_img(pdf_path: Path, page_index: int, dpi: int = 300) -> Image.Image:
    """PDF 한 페이지를 RGB PIL 이미지로 렌더링"""
    doc = fitz.open(str(pdf_path))
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=mat,
        colorspace=fitz.csRGB,
        alpha=False,
    )

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def pdfs_to_imgs(
    pdf_path: Path,
    dpi: int = 300,
) -> List[Tuple[int, str, Image.Image]]:
    """
    원본 PDF의 각 페이지를 렌더링한 뒤,
    분할 없이 그대로 PIL 이미지로 반환.

    반환:
      [(page_no, part_name, pil_img), ...]
    """
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    doc.close()

    outputs = []

    for page_idx in range(num_pages):
        img = pdf_to_img(pdf_path, page_idx, dpi=dpi)
        part_name = "full"
        outputs.append((page_idx + 1, part_name, img))

    return outputs



def extract_paddle(
    pipeline,
    image: Image.Image,
    image_name: str,
    md_dir: Path,
    json_dir: Path,
):
    md_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    img_np = np.array(image.convert("RGB"))

    print(f"[DEBUG] starting predict for image: {image_name}", flush=True)

    for res in pipeline.predict(img_np):
        md_path = md_dir / f"{image_name}.md"
        json_path = json_dir / f"{image_name}.json"

        res.save_to_markdown(save_path=md_path)
        res.save_to_json(save_path=json_path)

        print(f"saved → {md_path}", flush=True)
        print(f"saved → {json_path}", flush=True)

        del res


def extract_docling(pdf_path: Path, save_dir: Path, converter) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)  # save_dir를 폴더로 쓰는 구조

    doc = converter.convert(pdf_path).document

    for page_idx, _ in enumerate(sorted(doc.pages), start=1):  

        # Docling은 보통 page 단독 export가 page_item이 아니라 doc에서 page_no로 함
        page_md = doc.export_to_markdown(page_no=page_idx)

        saved_md_path = save_dir / f"{pdf_path.stem}_{page_idx}.md" 
        saved_md_path.parent.mkdir(parents=True, exist_ok=True)
        saved_md_path.write_text(page_md, encoding="utf-8")

        print(f"Saved page {page_idx} → {saved_md_path}")



def md2text(
    md_dir: Path | list[Path],
    output_dir: Path,
    prose_chain,
    decision_chain,
    use_decision_filter: bool = True,
) -> list[dict]:
    """Decision 필터링 + 줄글(prose) 변환만 수행해서,
    입력과 동일한 하위 디렉토리 구조를 유지한 채 output_dir 아래에 저장한다.
    (Page Merging / Agentic Chunking 없음)
    """

    if isinstance(md_dir, list):
        all_results = []
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
            all_results.extend(md2text(
                d, output_dir, prose_chain, decision_chain, use_decision_filter,
            ))
        print(f"\n[INFO] 전체 완료 — 총 {len(all_results)}개 파일")
        return all_results

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

    # ── Decision 필터링(조건부) + 줄글 변환 ─────────────────────
    print("\n[INFO] Decision 필터링 + 줄글 변환 중...")
    page_items = []
    for i, md_file in enumerate(md_files):
        rel = md_file.relative_to(md_root)
        print(f"  처리중: {rel} ({i+1}/{len(md_files)})")

        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            print(f"  → [SKIP] 빈 페이지")
            continue

        page_items.append((md_file, content))

    if use_decision_filter:
        useful_items = get_filtered_pages(md_dir, page_items, decision_chain)
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

    # ── 입력과 동일한 하위 구조로 저장 ─────────────────────────
    results = []
    for prose, md_file in prose_pages:
        rel_subdir = md_file.parent.relative_to(md_root)
        sub_output_dir = output_dir / rel_subdir
        sub_output_dir.mkdir(parents=True, exist_ok=True)

        out_path = sub_output_dir / f"{md_file.stem}_prose.md"
        out_path.write_text(prose, encoding="utf-8")

        record = {"source_file": md_file.name, "prose": prose}
        results.append(record)
        print(f"  → [SAVE] {out_path}")

    print(f"\n[INFO] {len(results)}개 파일 저장 완료 → {output_dir}")
    return results