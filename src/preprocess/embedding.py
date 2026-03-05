# src/preproc/vectorize.py
from __future__ import annotations

import re
import yaml, argparse, os
from types import SimpleNamespace
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.api import CreateCollectionConfiguration
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from collections import defaultdict


############## 임베딩 래퍼 및 벡터 DB 구축 ###################################
class Embeddor(Embeddings):
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


def build_chroma(
    texts: list[str],
    metadatas: list[dict],
    *,
    pdf_name: str,
    vector_dir: Path,
    emb: Embeddings,
):
    # 1) 컬렉션(=폴더별) 열기/생성
    '''' cosine 유사도 사용 명시 (기본값은 distance) '''
    store = Chroma(
        collection_name=pdf_name,
        persist_directory=str(vector_dir),
        embedding_function=emb,
        collection_configuration=CreateCollectionConfiguration(
            hnsw={"space": "cosine"}),
    )

    # 2) 문서 id: 폴더별 collection이므로 rel_path만으로 충분
    ids = [m["rel_path"] for m in metadatas]

    # 3) 임베딩 계산 (한 번에)
    embeddings = emb.embed_documents(texts)

    # 4) ✅ upsert: 있으면 업데이트, 없으면 추가
    store._collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return store

def md2vecdb(md_root: Path, pdf_name: str, vector_dir: Path, emb: Embeddings):
    files = sorted([p for p in md_root.rglob("*.md") if p.is_file()])
    if not files:
        print(f"[SKIP] no md: {md_root}")
        return None

    texts, metas = [], []
    for p in files:
        texts.append(p.read_text(encoding="utf-8"))
        metas.append({
            "filename": p.name,
            "source": str(p.resolve()),
            "rel_path": str(p.relative_to(md_root)),
            "type": "md",
        })

    return build_chroma(
        texts=texts,
        metadatas=metas,
        pdf_name=pdf_name,
        vector_dir=vector_dir,
        emb=emb,
    )
#######################################################################