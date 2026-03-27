# src/retrieval/lex_retriever.py
from __future__ import annotations

import chromadb
import pickle
from pathlib import Path
from typing import Dict, Any, List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from src.preprocess.embedding import _tokenize


def load_lexdb(vector_dir: Path, pdf_name: str) -> BM25Okapi:
    return pickle.loads((vector_dir / f"{pdf_name}_bm25.pkl").read_bytes())


def build_bm25s(*, vec_root: Path, k_each: int = 4) -> Dict[str, BM25Retriever]:
    client = chromadb.PersistentClient(path=str(vec_root))
    retriever_map: Dict[str, BM25Retriever] = {}

    for c in client.list_collections():
        pkl_path = vec_root / f"{c.name}_bm25.pkl"
        if not pkl_path.exists():
            print(f"[SKIP] no bm25 pkl: {c.name}")
            continue

        store = Chroma(collection_name=c.name, persist_directory=str(vec_root))
        raw   = store._collection.get(include=["documents", "metadatas"])
        docs  = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]
        retriever_map[c.name] = BM25Retriever(
            vectorizer=load_lexdb(vec_root, c.name),
            docs=docs,
            k=k_each,
            preprocess_func=_tokenize,
        )

    print(f"[LOAD] loaded {len(retriever_map)} bm25 retrievers from: {vec_root}")
    return retriever_map


class Multi_BM25s(BaseRetriever):
    retrievers: Dict[str, Any]
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        candidates: List[Document] = []

        for name, retriever in self.retrievers.items():
            for doc in retriever.invoke(query):
                md = dict(doc.metadata)
                md["__source_store__"] = name
                doc.metadata = md
                candidates.append(doc)

        top_docs = candidates[: self.top_k]
        for rank, d in enumerate(top_docs, start=1):
            d.metadata["__rank__"] = rank

        return top_docs