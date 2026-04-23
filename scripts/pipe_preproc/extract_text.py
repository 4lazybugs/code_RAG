from pathlib import Path
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from src.preprocess.extract import extract_paddle, pdfs_to_imgs
from paddleocr import PaddleOCRVL


if __name__ == "__main__":
    start = time.time()
    
    #converter = DocumentConverter() # docling

    # pipeline_pp = PPStructureV3(
    #     use_chart_recognition=False,
    #     format_block_content=True,
    #     layout_threshold=0.2,
    #     device="gpu:0",
    #     lang="korean",
    # )

    pipeline_ocrvl = PaddleOCRVL(
        use_chart_recognition=False,
        format_block_content=True,
        #layout_threshold=0.6,  # 높을수록 보수적으로 표 detecting 판단
        device="gpu:0",
    )

    input_root = Path("db/raw_db/in_use/")

    for pdf_path in input_root.rglob("*.pdf"):
        save_root = Path("db/raw_db_extracted_resolution") / pdf_path.stem

        pdfs_dir = save_root / "pdfs"
        mds_dir = save_root / "mds"
        jsons_dir = save_root / "jsons"

        pdfs_dir.mkdir(parents=True, exist_ok=True)
        mds_dir.mkdir(parents=True, exist_ok=True)
        jsons_dir.mkdir(parents=True, exist_ok=True)

        page_items = pdfs_to_imgs(
            pdf_path=pdf_path,
            dpi=300,
        )

        for page_no, part_name, page_img in page_items:
            stem_name = f"{pdf_path.stem}_{page_no}"

            image_path = pdfs_dir / f"{stem_name}.png"
            page_img.save(image_path)

            print(
                f"[OCR] {pdf_path.name} | page={page_no} | image={image_path}",
                flush=True
            )

            #extract_docling(converter=converter, pdf_path=pdf_path, save_dir=save_dir)
            extract_paddle(
                pipeline=pipeline_ocrvl,
                image=page_img,
                image_name=stem_name,
                md_dir=mds_dir,
                json_dir=jsons_dir,
            )

    end = time.time()

    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)

    print(f"\n총 걸린 시간: {int(minutes)}분 {seconds:.2f}초")