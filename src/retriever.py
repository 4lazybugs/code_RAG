# retriever.py
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import math
import re

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

from utils import get_config

# =========================
# 공통 설정 / 임베딩 래퍼
# =========================
CFG = get_config()
SEMANTIC_EMBED_MODEL = getattr(CFG, "embedor_model_name", "upskyy/bge-m3-korean")


class SentenceTransformerEmbeddings(Embeddings):
    """SentenceTransformer 임베딩 래퍼"""
    def __init__(self, model_name: str | None = None):
        if model_name is None:
            model_name = SEMANTIC_EMBED_MODEL
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


# 프로젝트 루트 기준 경로
BASE_DIR = Path(__file__).resolve().parent.parent


# =========================
# 컬렉션 이름 생성 (범용)
# =========================
def make_collection_name(folder_key: str, prefix: str) -> str:
    """
    prefix + folder_key를 Chroma collection 규칙에 맞게 안전 변환
    예:
      prefix="semantic", folder_key="straw"   -> "semantic_straw"
      prefix="cleaned",  folder_key="A/subA"  -> "cleaned_A_subA"
    """
    raw = f"{prefix}_{folder_key}"

    # 알파벳/숫자/언더스코어/하이픈 외는 '_'로 치환
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_")

    if len(name) < 3:
        name = "col_" + (name or "default")
    if len(name) > 63:
        name = name[:63]

    if not re.match(r"^[a-zA-Z0-9]", name):
        name = "c_" + name
    if not re.match(r".*[a-zA-Z0-9]$", name):
        name = name + "_c"

    return name


# =========================
# 범용 벡터DB 로더
# =========================
def load_folder_vector_retrievers(
    *,
    data_root: Path,
    vec_root: Path,
    prefix: str,
    level: int = 1,
    ndocs: int = 5,
) -> Dict[str, Any]:
    """
    data_root 하위의 level-depth 폴더들을 순회하며
    vec_root/<rel_path> 에서 Chroma를 로드해 retriever dict로 반환.

    - 반환 key: rel_path (예: "straw" 또는 "A/subA")
    - collection: make_collection_name(rel_path, prefix)
    - persist_directory: vec_root/rel_path

    level 의미:
      level=1 => data_root/* (하위 1레벨 폴더가 DB 단위)
      level=2 => data_root/*/* (하위 2레벨 폴더가 DB 단위)
    """
    data_root = data_root.resolve()
    vec_root = vec_root.resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not vec_root.exists():
        raise FileNotFoundError(f"vec_root not found: {vec_root}")

    emb = SentenceTransformerEmbeddings()
    retrievers: Dict[str, Any] = {}

    # level-depth 폴더들 수집
    # level=1 -> "*/"
    # level=2 -> "*/*/"
    pattern = ("*/" * level).rstrip("/")
    candidates = [p for p in data_root.glob(pattern) if p.is_dir()]

    if not candidates:
        print(f"[WARN] no folders under {data_root} at level={level}")
        return retrievers

    for folder in candidates:
        rel = folder.relative_to(data_root).as_posix()  # 예: "straw" or "A/subA"
        vec_dir = vec_root / rel

        if not vec_dir.exists():
            print(f"[WARN] vec DB not found: {vec_dir}")
            continue

        collection_name = make_collection_name(rel, prefix=prefix)

        store = Chroma(
            collection_name=collection_name,
            persist_directory=str(vec_dir),
            embedding_function=emb,
        )
        retrievers[rel] = store.as_retriever(search_kwargs={"k": ndocs})

    print(f"[LOAD] loaded {len(retrievers)} retrievers (prefix={prefix}, level={level})")
    return retrievers


# ---------------------------------------------
# 여러 retriever를 돌려서 cosine score(유사도) 기준으로
# 전역 top-k 문서를 돌려주는 멀티 리트리버
# ---------------------------------------------
class MultiCosineRetriever(BaseRetriever):
    """폴더별 retriever들을 합쳐서 전역 top-k를 뽑는 커스텀 리트리버"""

    retrievers: Dict[str, Any]
    k_each: int = 3   # 각 retriever에서 몇 개 가져올지
    top_k: int = 5    # 전체에서 최종 몇 개만 쓸지

    def _search_one(
        self,
        name: str,
        retriever: Any,
        query: str,
    ) -> List[Tuple[Document, Optional[float]]]:
        vs = getattr(retriever, "vectorstore", None)

        if vs is not None and hasattr(vs, "similarity_search_with_relevance_scores"):
            try:
                results = vs.similarity_search_with_relevance_scores(
                    query, k=self.k_each
                )
                return results  # List[Tuple[Document, float]]
            except Exception as e:
                print(f"[WARN] similarity_search_with_relevance_scores 실패 ({name}): {e}")

        try:
            docs = (
                retriever.invoke(query)
                if hasattr(retriever, "invoke")
                else retriever.get_relevant_documents(query)
            )
        except Exception as e:
            print(f"[WARN] retriever 호출 실패 ({name}): {e}")
            return []

        docs = docs[: self.k_each]
        return [(d, None) for d in docs]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        candidates: List[Tuple[Document, Optional[float]]] = []

        for name, r in self.retrievers.items():
            results = self._search_one(name, r, query)
            for doc, score in results:
                md = dict(doc.metadata)
                md["__source_store__"] = name
                if score is not None and not (isinstance(score, float) and math.isnan(score)):
                    md["__score__"] = float(score)
                doc.metadata = md
                candidates.append((doc, score))

        def key_fn(item: Tuple[Document, Optional[float]]):
            _doc, s = item
            return s if (s is not None and not math.isnan(s)) else -1e9

        candidates.sort(key=key_fn, reverse=True)
        top_docs = [d for d, _ in candidates[: self.top_k]]

        # ✅ rank 저장 (권장)
        for rank, d in enumerate(top_docs, start=1):
            md = dict(d.metadata)
            md["__rank__"] = rank
            d.metadata = md

        return top_docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=None)


def load_cleaned_md_level2_retrievers(ndocs: int = 5) -> Dict[str, Any]:
    """
    cleaned_md 하위 2레벨(L1/L2) 폴더별 DB 로드
    - data_root: db/cleaned_md/<L1>/<L2>
    - vec_root : db/vector_db/cleaned_md/<L1>/<L2>
    - collection: cleaned_<L1>_<L2> (safe)
    """
    return load_folder_vector_retrievers(
        data_root=BASE_DIR / "db" / "cleaned_md",
        vec_root=BASE_DIR / "db" / "vector_db" / "cleaned_md",
        prefix="cleaned",
        level=2,
        ndocs=ndocs,
    )