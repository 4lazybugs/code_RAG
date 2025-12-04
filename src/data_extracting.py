from pathlib import Path
import json
import os
import re
from dotenv import load_dotenv

# docling
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    smolvlm_picture_description
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# PyMuPDF
import fitz  # pip install pymupdf

load_dotenv()

######################## <Docling 파이프라인 설정> ########################
pipeline_options = PdfPipelineOptions(
    generate_page_images=True,
    generate_picture_images=True,
    images_scale=1.0,
    do_ocr=True,
    do_picture_description=True,
    ocr_options=RapidOcrOptions(
        lang=["korean"],          # 문자열 → 리스트
        use_cls=True,             # use_angle_cls ❌ → use_cls ✅
        force_full_page_ocr=False # 필요 시 True로 (스캔본 강제 OCR)
    ),
    picture_description_options=smolvlm_picture_description
)

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

######################## <데이터 추출 + Fallback 로직> #######################
pdf_dirs = [
    #Path("db/raw_db/farm_consulting"),
    Path("db/raw_db/manual_book")
]

output_root = Path("db/raw_db_extracted/markdown")  # Markdown 저장 루트 디렉토리
output_root.mkdir(parents=True, exist_ok=True)

pdf_files = []

# 두 디렉토리의 PDF를 재귀 탐색
for pdf_dir in pdf_dirs:
    pdf_files.extend(pdf_dir.rglob("*.pdf"))

for pdf_path in pdf_files:
    print(f"[INFO] 변환중 → {pdf_path}")

    # 저장 경로: PDF 이름을 동일하게 사용
    md_name = pdf_path.stem + ".md"
    save_path = output_root / md_name

    # 1) 먼저 docling 시도
    try:
        result = converter.convert(pdf_path)
        doc = result.document

        markdown_text = doc.export_to_markdown()
        save_path.write_text(markdown_text, encoding="utf-8")

        print(f"[DONE][docling] Markdown 저장 → {save_path}")
        continue  # docling 성공했으면 다음 파일로

    except Exception as e:
        print(f"[WARN] docling 변환 실패 → {pdf_path}")
        print(f"       이유: {e}")
        print(f"[INFO] PyMuPDF로 fallback 시도...")

    # 2) docling 실패 시 PyMuPDF로 fallback
    try:
        with fitz.open(pdf_path) as doc:
            page_texts = []
            for i, page in enumerate(doc):
                # 필요하면 "dict"나 "blocks"로 더 정교하게 처리 가능
                text = page.get_text("text")  # 순수 텍스트
                # 간단한 페이지 구분 + 제목을 Markdown 형태로
                page_texts.append(f"# Page {i+1}\n\n{text.strip()}")

        markdown_text = "\n\n---\n\n".join(page_texts)
        save_path.write_text(markdown_text, encoding="utf-8")

        print(f"[DONE][PyMuPDF] Markdown 저장 → {save_path}")

    except Exception as e2:
        print(f"[ERROR] PyMuPDF 변환도 실패 → {pdf_path}")
        print(f"        이유: {e2}")
 