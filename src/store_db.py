# store_db.py
import os
import shutil
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sqlalchemy import create_engine
import argparse
# PDF 처리를 위한 추가 라이브러리
import PyPDF2
from pathlib import Path
import fitz  # PyMuPDF

# --- RAW DB 경로 선언 ---
CROP_CSV_PATH      = "raw_db/crop_recommendation.csv"
QNA_PARQ_PATH      = "raw_db/agriculture_QnA.parquet"
PDF_PATH_01   = "raw_db/USDA_soil_survey_manual.pdf"
PDF_PATH_02   = "raw_db/food_and_agriculture.pdf"
PDF_PATH_03   = "raw_db/WRB_soil.pdf"

# ---VEC DB 경로 선언 ---
VEC_ROOT        = "./db/vector_db"
# 벡터 스토어 디렉토리명에 .db 확장자 추가
SOIL_VEC_DIR    = os.path.join(VEC_ROOT, "soil.db")
QNA_VEC_DIR     = os.path.join(VEC_ROOT, "agriculture_QnA.db")
CROP_VEC_DIR    = os.path.join(VEC_ROOT, "crop_recommendation.db")

SQL_ROOT        = "./db/sql_db"
QNA_SQL_FPATH   = os.path.join(SQL_ROOT, "agriculture_QnA.db")
CROP_SQL_FPATH  = os.path.join(SQL_ROOT, "crop_recommendation.db")

EMBED_MODEL     = "mxbai-embed-large"

##########################################
######## csv -> vector db #################
##########################################
def build_qna_vector_db():
    """raw_db/agriculture_QnA.parquet 로부터 QnA 벡터 DB를 새로 만듭니다."""
    if os.path.isdir(QNA_VEC_DIR):
        print(f"[INIT] 기존 QnA 벡터 폴더 삭제: {QNA_VEC_DIR}")
        shutil.rmtree(QNA_VEC_DIR)
    os.makedirs(QNA_VEC_DIR, exist_ok=True)

    print("[INIT] Q&A 벡터 DB 구축 중...")
    df_qna = pd.read_parquet(QNA_PARQ_PATH, engine="pyarrow")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    docs, ids = [], []
    for i, row in df_qna.iterrows():
        q = row['question'].strip()
        a = row['answers'].strip()
        docs.append(Document(page_content=f"Q: {q}\nA: {a}", metadata={"id": str(i)}))
        ids.append(str(i))

    batch_size, total = 5000, len(docs)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"[INIT] QnA 문서 업로드 중: {start}~{end} / {total}")
        store = Chroma(
            collection_name="agriculture_QnA",
            persist_directory=QNA_VEC_DIR,
            embedding_function=embeddings
        )
        store.add_documents(documents=docs[start:end], ids=ids[start:end])
        del store

    print(f"[INIT] QnA 벡터 DB 저장 완료: {total} documents")


def build_crop_vector_db():
    """raw_db/crop_recommendation.csv 로부터 Crop 벡터 DB를 새로 만듭니다."""
    if os.path.isdir(CROP_VEC_DIR):
        print(f"[INIT] 기존 Crop 벡터 폴더 삭제: {CROP_VEC_DIR}")
        shutil.rmtree(CROP_VEC_DIR)
    os.makedirs(CROP_VEC_DIR, exist_ok=True)

    print("[INIT] Crop 추천 벡터 DB 구축 중...")
    raw_df = pd.read_csv(CROP_CSV_PATH, header=None)
    raw_df.columns = raw_df.iloc[0]
    df = raw_df[1:].reset_index(drop=True)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    docs, ids = [], []
    for i, row in df.iterrows():
        content = " ".join(f"{k}: {v}" for k, v in row.items())
        docs.append(Document(page_content=content, metadata={"id": str(i)}))
        ids.append(str(i))

    batch_size, total = 5000, len(docs)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"[INIT] Crop 문서 업로드 중: {start}~{end} / {total}")
        store = Chroma(
            collection_name="crop_recommendation",
            persist_directory=CROP_VEC_DIR,
            embedding_function=embeddings
        )
        store.add_documents(documents=docs[start:end], ids=ids[start:end])
        del store

    print(f"[INIT] Crop 벡터 DB 저장 완료: {total} documents")

##########################################
######## pdf -> vector db ################
##########################################
# --- PDF 에서 글자 추출해서 text document 로 변환하는 함수 ----------
def parse_pdf_to_docs(pdf_path: str) -> list[Document]:
    """PDF 파일을 페이지 단위로 읽어 Document 리스트로 반환."""
    docs = []
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": Path(pdf_path).name, "page": i + 1}
                ))
    return docs


def build_pdf_vector_db(VEC_DIR, PDF_PATH, collection_name, init=False):
    if init == True:
        """raw_db/Soil_Questions.pdf 로부터 Soil Exam PDF 벡터 DB를 새로 만듭니다."""
        if os.path.isdir(VEC_DIR):
            print(f"[INIT] 기존 Soil PDF 벡터 폴더 삭제: {VEC_DIR}")
            shutil.rmtree(VEC_DIR)
        os.makedirs(VEC_DIR, exist_ok=True)

    print(f"[INIT] PDF → Document 추출 중: {PDF_PATH}")
    pdf_docs = parse_pdf_to_docs(PDF_PATH)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    store = Chroma(
        collection_name=collection_name,
        persist_directory=VEC_DIR,
        embedding_function=embeddings
    )
    ids = [f"page_{d.metadata['page']}" for d in pdf_docs]
    store.add_documents(documents=pdf_docs, ids=ids)
    print(f"[INIT] Soil PDF 벡터 DB 저장 완료: {len(pdf_docs)} documents")



##########################################
######## csv -> SQL db #################
##########################################
def build_qna_sql_db():
    """raw_db/agriculture_QnA.parquet 로부터 QnA용 SQLite DB를 새로 만듭니다."""
    os.makedirs(SQL_ROOT, exist_ok=True)
    if os.path.exists(QNA_SQL_FPATH):
        print(f"[INIT] 기존 QnA SQL DB 삭제: {QNA_SQL_FPATH}")
        os.remove(QNA_SQL_FPATH)

    print("[INIT] QnA SQL DB 구축 중...")
    df = pd.read_parquet(QNA_PARQ_PATH, engine="pyarrow")
    engine = create_engine(f"sqlite:///{QNA_SQL_FPATH}")
    df.to_sql("agriculture_QnA", engine, if_exists="replace", index=False)
    print(f"[INIT] QnA SQL DB 저장 완료: {df.shape[0]} rows")


def build_crop_sql_db():
    """raw_db/crop_recommendation.csv 로부터 Crop용 SQLite DB를 새로 만듭니다."""
    os.makedirs(SQL_ROOT, exist_ok=True)
    if os.path.exists(CROP_SQL_FPATH):
        print(f"[INIT] 기존 Crop SQL DB 삭제: {CROP_SQL_FPATH}")
        os.remove(CROP_SQL_FPATH)

    print("[INIT] Crop SQL DB 구축 중...")
    df_sql = pd.read_csv(CROP_CSV_PATH)
    engine = create_engine(f"sqlite:///{CROP_SQL_FPATH}")
    df_sql.to_sql("crop_recommendation", engine, if_exists="replace", index=False)
    print(f"[INIT] Crop SQL DB 저장 완료: {df_sql.shape[0]} rows")


##########################################
######## load and store db ###############
##########################################
def load_all_docs():
    df_qna = pd.read_parquet(QNA_PARQ_PATH, engine="pyarrow")
    
    ALL_DOCS = [
        Document(page_content=f"Q: {row['question'].strip()}\nA: {row['answers'].strip()}")
        for _, row in df_qna.iterrows()
    ]

    return ALL_DOCS

def load_stores(ndocs : int):
    """모든 벡터/SQL 스토어가 없으면 빌드하고, 있으면 로드한 뒤 retriever를 반환합니다."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # QnA 벡터 retriever
    if not os.path.isdir(QNA_VEC_DIR):
        build_qna_vector_db()
    qna_store = Chroma(
        collection_name="agriculture_QnA",
        persist_directory=QNA_VEC_DIR,
        embedding_function=embeddings
    )
    retriever_qna = qna_store.as_retriever(search_kwargs={"k": ndocs})

    # Crop 벡터 retriever
    if not os.path.isdir(CROP_VEC_DIR):
        build_crop_vector_db()
    crop_store = Chroma(
        collection_name="crop_recommendation",
        persist_directory=CROP_VEC_DIR,
        embedding_function=embeddings
    )
    retriever_crop = crop_store.as_retriever(search_kwargs={"k": ndocs})

    # Soil PDF 벡터 retriever
    soil_store = Chroma(
        collection_name="soil",
        persist_directory=SOIL_VEC_DIR,
        embedding_function=embeddings
    )
    retriever_soil = soil_store.as_retriever(search_kwargs={"k": ndocs})

    # SQL DB
    if not os.path.exists(QNA_SQL_FPATH):
        build_qna_sql_db()
    if not os.path.exists(CROP_SQL_FPATH):
        build_crop_sql_db()

    return retriever_qna, retriever_crop, retriever_soil, QNA_SQL_FPATH, CROP_SQL_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector/SQL DB 초기화")
    parser.add_argument("--init", action="store_true",
                        help="기존 DB를 삭제하고 재구축합니다.")
    args = parser.parse_args()

    if args.init:
        build_pdf_vector_db(VEC_DIR = SOIL_VEC_DIR, PDF_PATH = PDF_PATH_03, collection_name="soil", init=False)
    else:
        print("사용법:")
        print("  python store_db.py --init   # DB를 초기화하고 재구축")
