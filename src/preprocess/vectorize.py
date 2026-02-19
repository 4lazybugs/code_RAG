# src/preproc/vectorize.py
from __future__ import annotations

import re
import shutil
import yaml, argparse, os
from types import SimpleNamespace
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.api import CreateCollectionConfiguration
from sentence_transformers import SentenceTransformer
from typing import List, Optional

########## config utils #######################################
def load_yaml(path):
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # 환경 변수 치환 처리
    config = {}
    for k, v in raw_config.items():
        if isinstance(v, str):
            config[k] = os.path.expandvars(v)
        else:
            config[k] = v

    return config

def get_config(path='configs/config_vec.yaml'):
    default_cfg = load_yaml(path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=default_cfg.get('model_name'))
    parser.add_argument("--metrics", type=str, default=",".join(default_cfg.get("metrics", [])))

    args, _ = parser.parse_known_args()

    # ✅ YAML + argparse 병합 → Namespace
    cfg = {**default_cfg, **vars(args)}
    return SimpleNamespace(**cfg)
#######################################################################

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
    collection_name: str,
    persist_dir: Path,
    emb: Embeddings,
):
    # 1) 컬렉션(=폴더별) 열기/생성
    '''' cosine 유사도 사용 명시 (기본값은 distance) '''
    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
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

def md2vecdb(md_root: Path, collection_name: str, persist_dir: Path, emb: Embeddings):
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
        collection_name=collection_name,
        persist_dir=persist_dir,
        emb=emb,
    )
#######################################################################

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    VEC_DIR = BASE_DIR / "db" / "vector_db"
    EXTRACT_DIR = BASE_DIR / "db" / "raw_db_extracted"

    CFG = get_config(path='configs/config_vec.yaml')
    EMBED_MODEL = CFG.embedor_model_name

    # ✅ (1) 전체 초기화가 필요하면 여기서만 삭제 (딱 1번)
    # shutil.rmtree(VEC_DIR, ignore_errors=True)
    VEC_DIR.mkdir(parents=True, exist_ok=True)

    emb = Embeddor(EMBED_MODEL)

    md_files = list(EXTRACT_DIR.rglob("*.md"))
    print("[DBG] md_files:", len(md_files), "EXTRACT_DIR:", EXTRACT_DIR)

    level_folders = sorted({p.parent for p in md_files})
    print("[DBG] level_folders:", len(level_folders))

    first = True
    for folder in level_folders:
        print("[DBG] processing folder:", folder)
        rel = folder.relative_to(EXTRACT_DIR).as_posix()
        safe_rel = re.sub(r"[^a-zA-Z0-9_-]+", "_", rel).strip("_")
        collection = f"cleaned_{safe_rel}"[:63].strip("_-.")
        md2vecdb(folder, collection, VEC_DIR, emb)