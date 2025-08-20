# db_init.py (프로젝트 최상위에서 실행)
import os
import sys
import shutil
import src.store_db as store_db           # 라이브러리 모듈
from src.utils import get_config  

# --- src 모듈 import 경로 추가 ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

        # 공용 설정 로더

def decide_embed_model(cfg) -> str:
    # 1) 환경변수 우선
    env_model = os.getenv("EMBED_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    # 2) CFG의 embedor_model_name
    cfg_model = getattr(cfg, "embedor_model_name", None)
    if isinstance(cfg_model, str) and cfg_model.strip():
        return cfg_model.strip()
    # 3) 마지막 폴백(원하는 기본 모델로 바꿔도 됨)
    return "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # get_config()는 자체 argparse를 사용하므로, 이 스크립트에서는 별도 인자 안 씀
    CFG = get_config()

    # 임베딩 모델 결정 후 store_db 전역에 주입
    embed_model = decide_embed_model(CFG)
    store_db.EMBED_MODEL = embed_model
    print(f"[INIT] Embedding model = {embed_model}")

    # 전체 벡터DB 루트 정리
    if os.path.isdir(store_db.VEC_ROOT):
        print(f"[INIT] 기존 벡터 DB 전체 삭제: {store_db.VEC_ROOT}")
        shutil.rmtree(store_db.VEC_ROOT)

    # QnA, Crop 벡터 DB 구축
    store_db.build_qna_vector_db()
    store_db.build_crop_vector_db()

    # Soil PDF (USDA SSM) 벡터 DB 구축
    store_db.build_pdf_vector_db(
        VEC_DIR=store_db.SOIL_VEC_DIR,
        PDF_PATH=store_db.PDF_PATH_01,   # raw_db/USDA_soil_survey_manual.pdf
        collection_name="soil",
        init=True
    )

    # Bugs PDF 필요 시 주석 해제
    # store_db.build_pdf_vector_db(
    #     VEC_DIR=store_db.BUGS_VEC_DIR,
    #     PDF_PATH=store_db.PDF_PATH_04,
    #     collection_name="bugs",
    #     init=True
    # )

    # Farming 통합 PDF 벡터 DB 구축
    store_db.build_farming_vector_db()

    # SQL DB 구축
    store_db.build_qna_sql_db()
    store_db.build_crop_sql_db()

    print("[INIT] 모든 DB 초기화/구축 완료")


if __name__ == "__main__":
    main()
