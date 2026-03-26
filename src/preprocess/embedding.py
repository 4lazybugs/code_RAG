# src/preproc/vectorize.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Callable

from chromadb.api import CreateCollectionConfiguration
from kiwipiepy import Kiwi
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


_kiwi = Kiwi()

def _tokenize(text: str) -> list[str]:
    return [t.form for t in _kiwi.tokenize(text)]


############## 임베딩 래퍼 ##################################################
class Embeddor(Embeddings):
    """SentenceTransformer 기반 임베딩 래퍼"""
    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        #self.model = SentenceTransformer(model_name, device="gpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0].tolist()


############## 빌드 함수 #####################################################
def build_chromadb(texts: list[str], vector_dir: Path, pdf_name: str,
                   metadatas: list[dict], emb: Embeddings) -> None:
    store = Chroma(
        collection_name=pdf_name,
        persist_directory=str(vector_dir),
        embedding_function=emb,
        collection_configuration=CreateCollectionConfiguration(hnsw={"space": "cosine"}),
    )
    store._collection.upsert(
        ids=[m["rel_path"] for m in metadatas],
        documents=texts,
        metadatas=metadatas,
        embeddings=emb.embed_documents(texts),
    )


def build_lexdb(texts: list[str], vector_dir: Path, pdf_name: str,
                *args) -> None:
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    (vector_dir / f"{pdf_name}_bm25.pkl").write_bytes(pickle.dumps(bm25))


############## md → vector DB ################################################
def md2db(
    md_root: Path, pdf_name: str, vector_dir: Path,
    emb: Embeddings | None = None,
    fns: list[Callable] = [build_lexdb, build_chromadb],
) -> None:
    files = sorted(p for p in md_root.rglob("*.md") if p.is_file())
    if not files:
        print(f"[SKIP] no md: {md_root}")
        return

    texts, metas = [], []
    for p in files:
        texts.append(p.read_text(encoding="utf-8"))
        metas.append({
            "filename": p.name,
            "source":   str(p.resolve()),
            "rel_path": str(p.relative_to(md_root)),
            "type":     "md",
        })

    for fn in fns:
        fn(texts, vector_dir, pdf_name, metas, emb)