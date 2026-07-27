from pathlib import Path
from src.config import get_config

import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from src.preprocess.parsing import extract_paddle, pdfs_to_imgs
from paddleocr import PaddleOCRVL


def make_extraction_dirs(save_root: Path) -> tuple[Path, Path, Path]:
    dirs = [save_root / name for name in ("pdfs", "mds", "jsons")]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return tuple(dirs)


def process_page(pipeline, pdf_path: Path, page_no: int, page_img, pdfs_dir: Path, mds_dir: Path, jsons_dir: Path):
    stem_name = f"{pdf_path.stem}_{page_no}"
    image_path = pdfs_dir / f"{stem_name}.png"
    page_img.save(image_path)

    print(f"[OCR] {pdf_path.name} | page={page_no} | image={image_path}", flush=True)

    extract_paddle(
        pipeline=pipeline,
        image=page_img,
        image_name=stem_name,
        md_dir=mds_dir,
        json_dir=jsons_dir,
    )


def upscale_img(pipeline, pdf_path: Path, output_root: Path):
    pdfs_dir, mds_dir, jsons_dir = make_extraction_dirs(output_root / pdf_path.stem)

    for page_no, part_name, page_img in pdfs_to_imgs(pdf_path=pdf_path, dpi=300):
        process_page(pipeline, pdf_path, page_no, page_img, pdfs_dir, mds_dir, jsons_dir)


############ 파라미터 로드 ###############
CFG = get_config("configs/config_preproc.yaml")

DB_DIRS   = [Path(d) for d in CFG.db_dirs]  # list[Path]
MD_DIR = Path(CFG.md_out_dir)
###########################################

if __name__ == "__main__":
    start = time.time()

    pipeline_ocrvl = PaddleOCRVL(
        use_chart_recognition=False,
        format_block_content=True,
        device="gpu:0",
    )

    output_root = MD_DIR

    # DB_DIRS는 list[Path]이므로 각 디렉토리를 순회
    for input_dir in DB_DIRS:
        print(f"\n[INFO] 탐색 중: {input_dir}")
        pdf_files = sorted(input_dir.rglob("*.pdf"))
        print(f"[INFO] {len(pdf_files)}개 PDF 발견")

        for pdf_path in pdf_files:
            print(f"\n[INFO] 처리 중: {pdf_path.relative_to(input_dir)}")
            upscale_img(pipeline_ocrvl, pdf_path, output_root)

    minutes, seconds = divmod(time.time() - start, 60)
    print(f"\n총 걸린 시간: {int(minutes)}분 {seconds:.2f}초")