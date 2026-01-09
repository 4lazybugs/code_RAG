import os, argparse, yaml
from langchain_core.messages import BaseMessage
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"

from typing import Dict, Any
from inference.models_RAG import BaseExpert, PartialExpert, RawLlmExpert
from retriever import MultiCosineRetriever, load_retrievers
from load_params import load_yaml, get_config

##################################
##### load config.yaml ###########
##################################
def load_yaml(path='src/config.yaml'):
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

def get_config():
    default_cfg = load_yaml()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=default_cfg.get('model_name'))
    parser.add_argument("--sbert_model_name", type=str, default=default_cfg.get('sbert_model_name'))
    parser.add_argument("--bert_model_name", type=str, default=default_cfg.get('bert_model_name'))
    parser.add_argument("--embedor_model_name", type=str, default=default_cfg.get('embedor_model_name'))
    
    
    parser.add_argument("--input_file", type=str, default=default_cfg.get('input_file'))
    parser.add_argument("--output_file", type=str, default=default_cfg.get('output_file'))
    parser.add_argument("--task", type=str, default=default_cfg.get('task'))
    parser.add_argument("--ndocs", type=int, default=default_cfg.get('ndocs'))
    parser.add_argument("--max_new_tokens", type=int, default=default_cfg.get('max_new_tokens'))
    parser.add_argument("--threshold", type=float, default=default_cfg.get('threshold'))
    parser.add_argument("--w_rel", type=float, default=default_cfg.get('w_rel'))
    parser.add_argument("--w_sup", type=float, default=default_cfg.get('w_sup'))
    parser.add_argument("--w_use", type=float, default=default_cfg.get('w_use'))


    args = parser.parse_args()
    return args

# --------------------------------------------------
def normalize_answer(res) -> str:
    # (1) tuple이면 첫 번째를 답으로 간주
    if isinstance(res, tuple):
        res = res[0]

    # (2) LangChain 메시지(AIMessage 등)면 content만 꺼냄
    if isinstance(res, BaseMessage):
        return (res.content or "").strip()

    # (3) dict 형태면 흔한 키들 우선 처리
    if isinstance(res, dict):
        for key in ("result", "answer", "output_text", "content"):
            if key in res:
                return str(res[key]).strip()
        return str(res).strip()

    # (4) 그 외는 그냥 문자열 처리
    return str(res).strip()


def create_expert_instances(retriever_map: Dict[str, Any], retriever_mode: str) -> Dict[str, BaseExpert]:
    """Factory function to create expert instances"""
    experts = {}
    experts["partial"] = PartialExpert(retriever_map, retriever_mode)
    experts["raw_llm"] = RawLlmExpert(retriever_map, retriever_mode)
    #experts["self_ask"] = SelfAskExpert(retriever_map, retriever_mode)
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


if __name__ == "__main__":
    CFG = get_config()
    CFG.EMBED_MODEL = "embedor_model_name"

    try:
        ndocs = 5

        # ✅ cleaned_md/<L1>/<L2> 폴더별 retriever 로드
        cleaned_folder_retrievers = load_retrievers(ndocs=ndocs)

        # ✅ 폴더별 retriever들을 하나로 묶는 멀티 리트리버 생성
        cleaned_multi = MultiCosineRetriever(
            retrievers=cleaned_folder_retrievers,
            k_each=10,     # 폴더당 top-10 문서
            top_k=5        # 전체 최종 5개
        )

        # ✅ retriever_map 구성 (최소)
        retriever_map = {
            "cleaned_multi": cleaned_multi,
        }

        # ✅ 기본 retriever 설정
        retriever_mode = "cleaned_multi"
        print(f"[DEBUG] retriever_mode: {retriever_mode}")

        expert_instances = create_expert_instances(
            retriever_map,
            retriever_mode,
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
                answer = normalize_answer(res)
                print(f"\n[Answer]\n{answer}\n")
            except Exception as e:
                print(f"\n[오류 발생]: {str(e)}\n")

    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n[시스템 오류]: {str(e)}")
