# src/vector_utils.py
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional

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
        doc_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", rel_path).strip("_")
        store.add_documents([doc], ids=[doc_id])

        print(f"[OK] {p.relative_to(BASE_DIR)}")

    print(f"[DONE] MD 벡터 DB 구축 완료: {len(files)} files -> {vec_dir}")
    return store


if __name__ == "__main__":
    cleaned_root = BASE_DIR / "db" / "cleaned_md"
    vec_root = VEC_ROOT / "cleaned_md"

    # cleaned_md 이하에서 .md를 포함한 모든 하위 디렉터리를 색인 대상으로 수집
    md_files = list(cleaned_root.rglob("*.md"))
    level_folders = sorted({p.parent for p in md_files})

    for folder in level_folders:
        # folder.relative_to(cleaned_root) = "<L1>/<L2>"
        rel = folder.relative_to(cleaned_root).as_posix()  # 예: "A/subA"

        # 컬렉션 이름 안전화 (Chroma 제약 대응)
        safe_rel = re.sub(r"[^a-zA-Z0-9_-]+", "_", rel).strip("_")
        collection = f"cleaned_{safe_rel}"[:63].strip("_-.")

        # persist도 L1/L2 구조 유지 (충돌 방지 + 디버깅 용이)
        vec_dir = (vec_root / folder.relative_to(cleaned_root)).resolve()
        vec_dir.parent.mkdir(parents=True, exist_ok=True)

        build_md_vector_db(
            md_root=folder,              # L2 폴더 기준
            collection_name=collection,  # L1/L2 기준으로 고유하게
            vec_dir=vec_dir,             # db/vector_db/cleaned_md/<L1>/<L2>
            clear_existing=True,
            group_by_subdir=False,
        )
