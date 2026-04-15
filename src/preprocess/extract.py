from pathlib import Path
import time
from typing import Optional
from typing import List, Tuple

# PaddleOCR import
import fitz  # pymupdf
import numpy as np
from PIL import Image
from paddleocr import PaddleOCRVL, PPStructureV3

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