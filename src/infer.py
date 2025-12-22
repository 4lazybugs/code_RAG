import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from rag import PartialExpert, RawLlmExpert, SelfAskExpert
import json
import time
from models_eval import Rouge1Evaluator, RougeLEvaluator, BleuEvaluator # n-gram 기반 평가지표
from models_eval import BertEvaluator, SbertEvaluator, MoverEvaluator # 임베딩 기반 평가지표
from retriever import MultiCosineRetriever  # (기존 semantic 폴더별)
from retriever import load_cleaned_md_level2_retrievers  # ✅ cleaned_md/<L1>/<L2> 폴더별 (당신이 만든 것)

from pathlib import Path
# Expert 클래스들
from rag import PartialExpert, RawLlmExpert, SelfAskExpert
from models_eval.base import BaseEvaluator

class QagOnlyEvaluator(BaseEvaluator):
    metric_key = "qag_only"

    def compute_scores(self, references: list, generated: list) -> list:
        # QAG만 만들 거라 점수 계산은 사용하지 않음
        return []

if __name__ == '__main__':
    start_time = time.time()
    results_dir = (Path(__file__).resolve().parent / ".." / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) 파라미터 입력
    qa_mode = 'cleaned'
    retriever_mode = 'cleaned_multi'

    selected_modes = ['partial_10', 'raw_llm']
    selected_metrics = ['rouge1','rougeL','bert','sbert','bleu']
    #selected_metrics = ['sbert']
    sample_size = None

    qa_data_path = {'cleaned': 'qa_data/GT/test/'}

    # --------------------------------------------------
    # ✅ [추가] GT 폴더(재귀) -> 단일 JSON으로 병합 (Evaluator 수정 없이)
    # -------------------------------------------------
    def _load_json_any(fp: Path):
        data = json.load(open(fp, "r", encoding="utf-8"))
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        if isinstance(data, list):
            return data
        return [data]

    gt_root = Path(qa_data_path[qa_mode])
    merged_items = []
    for fp in gt_root.rglob("*.json"):          # 필요하면 "*.jsonl"도 추가해서 처리
        merged_items.extend(_load_json_any(fp))

    # ✅ id(숫자) 기준으로 정렬
    merged_items.sort(key=lambda x: x["id"])

    merged_path = gt_root.parent / "_merged_gt.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_items, f, ensure_ascii=False, indent=2)

    qa_data_path[qa_mode] = str(merged_path)    # ✅ 이후 Evaluator는 “단일 파일”로 인식
    # --------------------------------------------------

    # 2) 초기 retriever 세팅
    ndocs_init = 10
    cleaned_folder_retrievers = load_cleaned_md_level2_retrievers(ndocs=ndocs_init)
    cleaned_multi = MultiCosineRetriever(retrievers=cleaned_folder_retrievers, k_each=10, top_k=7)

    retr_map = {"cleaned_multi": cleaned_multi}
    summary = []

    for mode in selected_modes:
        if mode.startswith("partial_"):
            expert = PartialExpert(retr_map, retriever_mode)
        elif mode.startswith("selfask_"):
            k = int(mode.split("_")[1])
            expert = SelfAskExpert(retr_map, retriever_mode, max_iter=k)
        elif mode == "raw_llm":
            expert = RawLlmExpert(retr_map, retriever_mode)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        qag_ev = QagOnlyEvaluator(expert, qa_data_path=qa_data_path[qa_mode], sample_size=sample_size)
        questions, references, generated, qa_id = qag_ev.get_data(mode)

        out_path = results_dir / "qag" / f"{mode}.json"   # metric 제거
        qag_ev.save_json(mode, qa_id, questions, references, generated, {}, str(out_path))

    elapsed = time.time() - start_time
    print(f"⏱ 전체 추론 완료: {elapsed/60:.2f}분")
