# retriever.py 예시

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
    """docs_semantic_md용 SentenceTransformer 임베딩 래퍼"""
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
MD_ROOT  = BASE_DIR / "db" / "docs_semantic_md"
VEC_ROOT = BASE_DIR / "db" / "vector_db" / "docs_semantic_md"


def make_collection_name(folder_name: str) -> str:
    """
    Chroma collection 이름 규칙을 만족하도록 폴더 이름을 안전하게 변환
    (store_db_semantic에서 쓰던 로직이랑 동일하게 맞춰줘야,
     이미 만들어둔 vec DB를 그대로 불러올 수 있음)
    """
    raw = f"semantic_{folder_name}"
    # 알파벳/숫자/언더스코어/하이픈 외는 '_'로 치환
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw)
    name = name.strip("_")

    if len(name) < 3:
        name = "col_" + (name or "default")
    if len(name) > 63:
        name = name[:63]

    if not re.match(r"^[a-zA-Z0-9]", name):
        name = "c_" + name
    if not re.match(r".*[a-zA-Z0-9]$", name):
        name = name + "_c"

    return name


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
        return top_docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=None)


# ---------------------------------------------
# docs_semantic_md 하위 폴더별 retriever 로더
# ---------------------------------------------
def load_semantic_retrievers(ndocs: int = 5) -> Dict[str, Any]:
    """
    db/docs_semantic_md/<folder> 구조를 가정하고,
    각 폴더별로 Chroma 벡터DB를 로드해서 retriever로 반환.

    - vec DB 경로: db/vector_db/docs_semantic_md/<folder_name>
    - collection_name: make_collection_name(folder_name) 로 생성
    - 이미 store_db_semantic(또는 별도 스크립트)로 인덱싱해 둔 상태라고 가정.
      (없으면 경고만 찍고 스킵)
    """
    if not MD_ROOT.exists():
        raise FileNotFoundError(f"MD root not found: {MD_ROOT}")

    VEC_ROOT.mkdir(parents=True, exist_ok=True)

    emb = SentenceTransformerEmbeddings()
    retrievers: Dict[str, Any] = {}

    subdirs = [d for d in MD_ROOT.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"[WARN] docs_semantic_md 하위 폴더가 없습니다: {MD_ROOT}")

    for folder in subdirs:
        vec_dir = VEC_ROOT / folder.name
        collection_name = make_collection_name(folder.name)

        if not vec_dir.exists():
            print(f"[WARN] vec DB not found for '{folder.name}' → {vec_dir}")
            # 이미 인덱싱 해 둔 환경이 아니라면, 여기서 자동으로 md 스캔해서
            # build 해주는 로직을 추가해도 됨.
            continue

        store = Chroma(
            collection_name=collection_name,
            persist_directory=str(vec_dir),
            embedding_function=emb,
        )

        retrievers[folder.name] = store.as_retriever(
            search_kwargs={"k": ndocs}
        )

    print(f"[LOAD] semantic retrievers loaded: {list(retrievers.keys())}")
    return retrievers
