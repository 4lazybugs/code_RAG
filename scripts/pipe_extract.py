from pathlib import Path
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"           # HuggingFace tokenizer 경고 방지
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"  # 오프라인 환경 필수

# paddle ocr
from src.preprocess.extract import extract_paddle
from paddleocr import PaddleOCRVL, PPStructureV3

# docling
from src.preprocess.extract import DocumentConverter, extract_docling

if __name__ == "__main__":
    start = time.time()
    
    # Docling, PaddlePaddleOCR 초기화
    #converter = DocumentConverter()

    # PaddleOCRVL, PPStructureV3 아무거나 써도 되는데 PaddleOCRVL은 VLM 기반이라 품질 더 좋은 대신 좀 느림
    pipeline_ocr = PaddleOCRVL(
        use_chart_recognition=False,         # 그래프/차트를 표로 인식하는 옵션
        format_block_content=True,          # 기본 False → 블록 내용 포맷 정리
        layout_threshold=0.2,               # 기본 0.3 → 낮추면 표 경계 더 잘 잡음
        device="gpu:0",  
    )

    pipeline_pp = PPStructureV3(
        use_chart_recognition=False,         # 그래프/차트를 표로 인식하는 옵션
        format_block_content=True,          # 기본 False → 블록 내용 포맷 정리
        layout_threshold=0.2,               # 기본 0.3 → 낮추면 표 경계 더 잘 잡음
        device="gpu:0",  
        lang="korean",  # PPStructureV3 
    )

    # root 입력 폴더
    input_root = Path("db/raw_db/db_in_use")
    #input_root = Path("db/raw_db/test_db/consulting")

    # 모든 pdf 재귀 탐색
    for pdf_path in input_root.rglob("*.pdf"):

        # 저장 폴더
        save_dir = Path("db/raw_db_extracted") / pdf_path.stem

        # Docling
        #extract_docling(converter=converter, pdf_path=pdf_path, save_dir=save_dir)
        # Paddle OCR
        extract_paddle(pipeline=pipeline_ocr, pdf_path=pdf_path, save_dir=save_dir)
    
    end = time.time()

    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)

    print(f"\n총 걸린 시간: {int(minutes)}분 {seconds:.2f}초")