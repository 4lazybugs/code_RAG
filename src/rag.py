import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from models_RAG import ( 
    BaseExpert, AllExpert, PartialExpert, SqlExpert,
    RawLlmExpert, AdaptiveExpert, SelfAskExpert
)
from typing import Dict, Any
from retriever import MultiCosineRetriever
from utils import load_yaml, get_config

from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer
from typing import Dict, Any, List, Optional, Tuple
from retriever import load_semantic_retrievers
from langchain_core.documents import Document

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"

# --------------------------------------------------

def create_expert_instances(retriever_map: Dict[str, Any], retriever_mode: str, 
                           qna_sql_path: str, crop_sql_path: str) -> Dict[str, BaseExpert]:
    """Factory function to create expert instances"""
    experts = {}
    experts["all"] = AllExpert(retriever_map, retriever_mode)
    experts["partial"] = PartialExpert(retriever_map, retriever_mode)
    experts["sql"] = SqlExpert(retriever_map, retriever_mode, qna_sql_path, crop_sql_path)
    #experts["adaptive"] = AdaptiveExpert(retriever_map, retriever_mode)
    experts["raw_llm"] = RawLlmExpert(retriever_map, retriever_mode)
    experts["self_ask"] = SelfAskExpert(retriever_map, retriever_mode)

    # (옵션) self_ask_5 같은 변형 처리하려면 여기 확장
    return experts


def get_help_text() -> str:
    return """
[모드별 사용 가이드]
mode       | 설명
-----------------------------------------------
all        | 전체 DB를 대상으로 RAG (정확도 우선)
partial    | 상위 K개 snippet만 RAG (속도/부분답변)
sql        | text→SQL→실행→해석 (테이블 기반 추천)
raw_llm    | LLM 자유 응답 (retrieval 없이)
self_ask   | Self-Ask 방식: 충분도 검사→추가질문→답변 집계
(종료: q)
""".strip()

# adaptive   | Adaptive gating + RAG (Self-RAG 방식)

if __name__ == "__main__":
    CFG = get_config()

    import store_db
    store_db.EMBED_MODEL = "upskyy/bge-m3-korean"

    try:
        ndocs = 5

        # 기존 csv/pdf/straw 등
        stores = load_stores(ndocs=ndocs)

        # 🔹 docs_semantic_md 하위 폴더별 retriever 로드
        semantic_folder_retrievers = load_semantic_retrievers(ndocs=ndocs)

        # 🔹 폴더별 retriever들을 하나로 묶는 멀티 리트리버 생성
        semantic_multi = MultiCosineRetriever(
            retrievers=semantic_folder_retrievers,
            k_each=10,     # 폴더당 top-10 문서
            top_k=5        # 전체 최종 5개
        )

        # 전체 retriever 맵 구성
        retriever_map = {
            "semantic_multi": semantic_multi,
            "straw": stores.get("straw"),
            "qna": stores.get("qna"),
            "crop": stores.get("crop"),
            "soil": stores.get("soil"),
            "bugs": stores.get("bugs"),
            "farm": stores.get("farm"),
        }

        # 기본 retriever 설정
        retriever_mode = "semantic_multi"
        print(f"[DEBUG] retriever_mode: {retriever_mode}")

        # SQL 경로
        qna_sql_path  = stores['qna_sql']
        crop_sql_path = stores['crop_sql']

        expert_instances = create_expert_instances(
            retriever_map,
            retriever_mode,
            qna_sql_path,
            crop_sql_path
        )

        help_text = get_help_text()

        while True:
            print(help_text)
            mode = input("Mode → all/partial/sql/raw_llm/self_ask (q to quit): ").strip().lower()
            if mode == "q":
                break
            if mode not in expert_instances:
                print("Invalid mode. 다시 선택하세요.")
                continue

            question = input("Question → ").strip()
            if question.lower() == "q":
                break

            expert = expert_instances[mode]

            try:
                res = expert.handle(question)
                answer = res[0] if isinstance(res, tuple) else res
                print(f"\n[Answer]\n{answer}\n")
            except Exception as e:
                print(f"\n[오류 발생]: {str(e)}\n")

    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n[시스템 오류]: {str(e)}")
