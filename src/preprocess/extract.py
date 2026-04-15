from pathlib import Path
import time

# PaddleOCR import
import fitz  # pymupdf
from paddleocr import PaddleOCRVL

# Docling imports: https://docling-project.github.io/docling/
from docling.document_converter import DocumentConverter

def extract_paddle(pipeline, pdf_path: Path, save_dir: Path):
    print(f"PDF ({pdf_path})를 찾았습니다.", flush=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    print(f"[DEBUG] page_count = {page_count} ({pdf_path})", flush=True)

    pdf_dir = save_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # 페이지별 PDF 미리 저장
    for page_idx in range(page_count):
        one_page_pdf = pdf_dir / f"{pdf_path.stem}_{page_idx + 1}.pdf"
        one = fitz.open()
        one.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        one.save(str(one_page_pdf))
        one.close()
    doc.close()

    # predict는 원본 통째로
    print(f"[DEBUG] starting predict for {pdf_path}", flush=True)  # ✅ 추가
    for page_no, res in enumerate(pipeline.predict(str(pdf_path)), start=1):
        print(f"[DEBUG] page {page_no} complete!", flush=True)
        stem = f"{pdf_path.stem}_{page_no}"
        md_path   = save_dir / f"{stem}.md"
        json_path = save_dir / f"{stem}.json"
        res.save_to_markdown(save_path=md_path)
        res.save_to_json(save_path=json_path)
        print(f"[{page_no}] saved → {md_path}", flush=True)
        print(f"[{page_no}] saved → {json_path}", flush=True)
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


# 모듈 테스트 코드
if __name__ == "__main__":
    start = time.time()

    # Docling 초기화
    converter = DocumentConverter()
    extract_docling(
        converter=converter,
        pdf_path=Path("db/raw_db/test_db/farm_consulting_test.pdf"), # pdf_path는 dir가 아니라 .pdf를 요구
        save_dir=Path("db/raw_db_extracted/farm_consulting/"), # save_dir는 .pdf가 아니라 폴더를 요구
    )
    
    # OCR pipeline 초기화
    pipeline = PaddleOCRVL()
    extract_paddle(
        pipeline=pipeline,
        pdf_path=Path("db/raw_db/test_db/Rice (Machine Transplanting) Farming Schedule.pdf"), # pdf_path는 dir가 아니라 .pdf를 요구
        save_dir=Path("db/raw_db_extracted/farm_consulting/"), # save_dir는 .pdf가 아니라 폴더를 요구
    )

    end = time.time()
    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n총 걸린 시간: {int(minutes)}분 {seconds:.2f}초")
