# retriever.py (minimal)
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import chromadb

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

from src.config import get_config, load_yaml
from src.preprocess.embedding import Embeddor


def build_retrievers(*, vec_root: Path, emb) -> Dict[str, Any]:
    # 메타데이터 목록 읽어옴
    client = chromadb.PersistentClient(path=str(vec_root))
    cols = client.list_collections()  

    retriever_list: Dict[str, Any] = {}

    for c in cols:
        name = c.name
        # 저장된 chroma db들 로드
        store = Chroma(
            collection_name=name,
            persist_directory=str(vec_root),
            embedding_function=emb,
        )
        # 메타데이터에서 읽어온 DB 이름을 기준으로 각 Retriever를 매핑
        retriever_list[name] = store.as_retriever() # default k=4

    print(f"[LOAD] loaded {len(retriever_list)} retrievers from DB: {vec_root}")
    return retriever_list


class Multi_Retriever(BaseRetriever):
    retrievers: Dict[str, Any]
    k_each: int = 3
    top_k: int = 5

    def _search_one(self, name: str, retriever: Any, query: str) -> List[Tuple[Document, Optional[float]]]:
        vector_db = getattr(retriever, "vectorstore", None)
        '''
        ## similarity_search_with_relevance_scores() ## <--- _select_relevance_score_fn()
        /home/jwkim/miniconda3/envs/rag_env/lib/python3.10/site-packages/langchain_core/vectorstores/base.py :ctrl+o
        
            ## _select_relevance_score_fn() ## <-- _cosine_relevance_score_fn() <-- similarity_search_with_scores()
            /home/jwkim/miniconda3/envs/rag_env/lib/python3.10/site-packages/langchain_chroma/vectorstores.py :ctrl+o
        
                ## _cosine_relevance_score_fn() ##
                /home/jwkim/miniconda3/envs/rag_env/lib/python3.10/site-packages/langchain_core/vectorstores/base.py :ctrl+o
                ## similarity_search_with_scores() ##
                /home/jwkim/miniconda3/envs/rag_env/lib/python3.10/site-packages/langchain_chroma/vectorstores.py :ctrl+o
        '''
        return vector_db.similarity_search_with_relevance_scores(query, k=self.k_each)

    '''
    ### BaseRetriever는 invoke()를 이미 구현해 놓았으므로, _get_relevant_documents()만 구현하면 됨 ####
    def invoke(self, query):
        return self._get_relevant_documents(query)
    '''

    def _get_relevant_documents(self, query: str) -> List[Document]:
        candidates: List[Tuple[Document, Optional[float]]] = []

        for name, retriever in self.retrievers.items():
            for doc, score in self._search_one(name, retriever, query):
                #breakpoint()
                md = dict(doc.metadata)
                md["__source_store__"] = name # 원래 폴더명(pdf 파일명)
                md["__score__"] = float(score)
                doc.metadata = md
                candidates.append((doc, score))

        def get_score(item: Tuple[Document, Optional[float]]):
            score = item[1]
            return score

        candidates.sort(key=get_score, reverse=True)  # 내림차순으로 정렬: score 높은 것이 1순위
        top_docs = [d for d, _ in candidates[: self.top_k]]

        for rank, d in enumerate(top_docs, start=1):
            md = dict(d.metadata)
            md["__rank__"] = rank
            d.metadata = md

        return top_docs


# -------------------------
# main usage
# -------------------------
if __name__ == "__main__":
    CFG = get_config("configs/config_emb.yaml")
    emb = Embeddor(CFG.embedor_model_name)

    pth = Path("/home/jwkim/[code]생성형과제_server/db/vector_db")
    retriever_list = build_retrievers(vec_root=pth, emb=emb)
    multi_retriever = Multi_Retriever(retrievers=retriever_list, k_each=3, top_k=5)
    q = "벼 이앙 재배 일정 알려줘"
    top_docs = multi_retriever.invoke(q)    

    for d in top_docs:
        print(
            d.metadata.get("__rank__"),
            d.metadata.get("rel_path"),   # ← 원래 md 파일명
            d.metadata.get("__score__")
        )


