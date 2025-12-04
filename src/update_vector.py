# src/vector_utils.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Literal

import fitz  # PyMuPDF
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from utils import get_config


# ================== 공통 설정 ==================

CFG = get_config()
EMBED_MODEL = getattr(CFG, "embedor_model_name", "upskyy/bge-m3-korean")

BASE_DIR = Path(__file__).resolve().parent.parent   # project 루트
VEC_ROOT = BASE_DIR / "db" / "vector_db"


class SentenceTransformerEmbeddings(Embeddings):
    """SentenceTransformer 기반 임베딩 래퍼 (모델 이름은 config에서)"""
    def __init__(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = EMBED_MODEL
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0].tolist()


# ================== PDF → Documents ==================
def parse_pdf_to_docs(pdf_path: Path, category: Optional[str] = None) -> List[Document]:
    """단일 PDF를 페이지 단위 Document 리스트로 파싱"""
    docs: List[Document] = []
    with fitz.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            if not text:
                continue
            meta = {
                "source": pdf_path.name,
                "page": i + 1,
            }
            if category is not None:
                meta["category"] = category
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def build_pdf_vector_db(
    target: Path,
    collection_name: str,
    vec_dir: Optional[Path] = None,
    clear_existing: bool = False,
    pattern: str = "*.pdf",
    category_mode: Literal["none", "dir", "file"] = "none",
):
    """
    PDF 파일(또는 폴더)을 받아서 하나의 컬렉션으로 벡터DB 구축.

    - target:
        - PDF 파일 경로  → 그 파일만 인덱싱
        - 디렉토리 경로 → 하위의 pattern에 매칭되는 PDF 전부 인덱싱
    - collection_name: Chroma 컬렉션 이름
    - vec_dir: 벡터DB가 저장될 디렉토리 (None이면 VEC_ROOT/collection_name 사용)
    - clear_existing: True이면 기존 vec_dir 삭제 후 새로 생성
    - category_mode:
        - "none" : category 메타데이터 안 넣음
        - "dir"  : 상위 폴더 이름을 category로 사용
        - "file" : 파일 이름(확장자 제외)을 category로 사용
    """
    target = target.resolve()

    if vec_dir is None:
        vec_dir = VEC_ROOT / f"{collection_name}.db"
    vec_dir = vec_dir.resolve()

    if clear_existing and vec_dir.exists():
        print(f"[INIT] 기존 벡터 폴더 삭제: {vec_dir}")
        shutil.rmtree(vec_dir)
    vec_dir.mkdir(parents=True, exist_ok=True)

    emb = SentenceTransformerEmbeddings()
    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(vec_dir),
        embedding_function=emb,
    )

    # ---- 대상 PDF 리스트 수집 ----
    pdf_files: List[Path] = []
    if target.is_file() and target.suffix.lower() == ".pdf":
        pdf_files = [target]
    elif target.is_dir():
        pdf_files = [p for p in target.rglob(pattern) if p.is_file()]
    else:
        raise FileNotFoundError(f"PDF target not found: {target}")

    print(f"[INIT] PDF 파일 수: {len(pdf_files)} (collection={collection_name})")

    total_docs = 0
    for pdf_path in pdf_files:
        if category_mode == "dir":
            category = pdf_path.parent.name
        elif category_mode == "file":
            category = pdf_path.stem
        else:
            category = None

        pdf_docs = parse_pdf_to_docs(pdf_path, category=category)
        if not pdf_docs:
            continue

        ids = [
            f"{pdf_path.stem}_p{d.metadata['page']}"
            for d in pdf_docs
        ]
        store.add_documents(pdf_docs, ids=ids)
        total_docs += len(pdf_docs)
        print(f"[OK] {pdf_path.relative_to(BASE_DIR)} -> {len(pdf_docs)} pages")

    print(f"[DONE] PDF 벡터 DB 구축 완료: files={len(pdf_files)}, docs={total_docs}")
    return store


# ================== MD → Documents ==================

def build_md_vector_db(
    md_root: Path,
    collection_name: str,
    vec_dir: Optional[Path] = None,
    clear_existing: bool = False,
    group_by_subdir: bool = False,
):
    """
    md_root 이하 모든 .md 파일을 하나의 컬렉션으로 벡터DB 구축.

    - md_root: .md 파일들이 있는 디렉토리 (rglob 사용)
    - collection_name: Chroma 컬렉션 이름
    - vec_dir: 벡터DB 디렉토리 (None이면 VEC_ROOT/collection_name 사용)
    - clear_existing: True이면 기존 vec_dir 삭제
    - group_by_subdir: True이면 metadata["group"]에 1레벨 하위 폴더 이름 저장
    """
    md_root = md_root.resolve()

    if not md_root.exists():
        raise FileNotFoundError(f"markdown root not found: {md_root}")

    if vec_dir is None:
        vec_dir = VEC_ROOT / f"{collection_name}"
    vec_dir = vec_dir.resolve()

    if clear_existing and vec_dir.exists():
        print(f"[INIT] 기존 MD 벡터 폴더 삭제: {vec_dir}")
        shutil.rmtree(vec_dir)
    vec_dir.mkdir(parents=True, exist_ok=True)

    emb = SentenceTransformerEmbeddings()
    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(vec_dir),
        embedding_function=emb,
    )

    files = [p for p in md_root.rglob("*.md") if p.is_file()]
    print(f"[INIT] md_root={md_root}, .md files={len(files)} (collection={collection_name})")

    if not files:
        print(f"[WARN] No .md files found under {md_root}")
        return store

    for p in files:
        text = p.read_text(encoding="utf-8")
        src = str(p.resolve())
        rel_path = str(p.relative_to(md_root))

        meta = {
            "filename": p.name,
            "source": src,
            "rel_path": rel_path,
        }
        if group_by_subdir:
            # md_root 바로 아래 1레벨 폴더 이름을 group으로
            try:
                sub = p.relative_to(md_root).parts[0]
            except IndexError:
                sub = ""
            meta["group"] = sub

        doc = Document(page_content=text, metadata=meta)

        # 재색인 시 중복 제거
        store.delete(where={"source": src})
        store.add_documents([doc], ids=[p.stem])

        print(f"[OK] {p.relative_to(BASE_DIR)}")

    print(f"[DONE] MD 벡터 DB 구축 완료: {len(files)} files -> {vec_dir}")
    return store


if __name__ == "__main__":
    # 예시 실행 (원할 때만 사용)
    # 1) 단일 PDF 파일 인덱싱
    # build_pdf_vector_db(
    #     target=BASE_DIR / "raw_db" / "USDA_soil_survey_manual.pdf",
    #     collection_name="soil",
    #     clear_existing=True,
    # )

    # 2) PDF 디렉토리 전체 인덱싱 (카테고리를 상위 폴더명으로)
    # build_pdf_vector_db(
    #     target=BASE_DIR / "raw_db" / "Farming Schedules and Strategic Crop Guidebooks",
    #     collection_name="farming",
    #     clear_existing=True,
    #     category_mode="dir",
    # )

    # 3) docs_semantic_md 전체 인덱싱
    build_md_vector_db(
        md_root=BASE_DIR / "db" / "docs_semantic_md",
        collection_name="docs_semantic_md",
        clear_existing=True,
        group_by_subdir=True,
    )

