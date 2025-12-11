from pathlib import Path
from paddleocr import PaddleOCRVL

# OCR pipeline 초기화 (모델 로딩)
pipeline = PaddleOCRVL()

# 입력 폴더 (재귀 탐색)
input_dir = Path("db/raw_db/")
pdf_files = list(input_dir.rglob("*.pdf"))
print(f"총 {len(pdf_files)}개의 PDF를 찾았습니다.")

# 출력 폴더
save_dir = Path("db/raw_db_extracted/")
save_dir.mkdir(parents=True, exist_ok=True)

for pdf_path in pdf_files:
    print(f"\nProcessing: {pdf_path}")

    # PDF 한 개 OCR 수행
    output = pipeline.predict(str(pdf_path))

    # 각 페이지 결과 저장
    for page_idx, res in enumerate(output):
        # Markdown 파일 이름 생성: test2_0.md처럼
        save_path = save_dir / pdf_path.stem

        # 저장
        res.save_to_markdown(save_path=save_path)
        print(f"Saved page {page_idx} → {save_path}")
