import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from rag import PartialExpert, RawLlmExpert, SelfAskExpert
from rag import MultiCosineRetriever                # ✅ 추가
from retriever import load_cleaned_md_level2_retrievers
import json
import pandas as pd
import torch
import time
import gc
import openai
import numpy as np
from utils import get_config
from konlpy.tag import Okt
from rouge_score.tokenizers import Tokenizer
from models_eval import (
    Rouge1Evaluator, RougeLEvaluator, BertEvaluator,
    SbertEvaluator, MoverEvaluator, BleurtEvaluator, BleuEvaluator
)
from retriever import (
    MultiCosineRetriever,               # (기존 semantic 폴더별)
    load_cleaned_md_level2_retrievers        # ✅ cleaned_md/<L1>/<L2> 폴더별 (당신이 만든 것)
)
from pathlib import Path
# Expert 클래스들
from rag import PartialExpert, RawLlmExpert, SelfAskExpert

CFG = get_config()
# store_db 전역 임베딩 모델 주입
openai.api_key = os.getenv("OPENAI_API_KEY")


class KoreanTokenizer(Tokenizer):
    def __init__(self):
        self.okt = Okt()

    def tokenize(self, text):
        return self.okt.morphs(text)


if __name__ == '__main__':
    start_time = time.time()

    # 1) 파라미터 입력
    qa_mode = 'cleaned'
    retriever_mode = 'cleaned_multi'

    selected_modes = ['partial_10', 'raw_llm']
    selected_metrics = ['rouge1','rougeL','bert','sbert','bleu']
    sample_size = None

    qa_data_path = {'cleaned': 'qa_data/GT/'}

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

    merged_path = gt_root.parent / "_merged_gt.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_items, f, ensure_ascii=False, indent=2)

    qa_data_path[qa_mode] = str(merged_path)    # ✅ 이후 Evaluator는 “단일 파일”로 인식
    # --------------------------------------------------

    # 2) 초기 retriever 세팅
    ndocs_init = 10
    cleaned_folder_retrievers = load_cleaned_md_level2_retrievers(ndocs=ndocs_init)
    cleaned_multi = MultiCosineRetriever(retrievers=cleaned_folder_retrievers, k_each=10, top_k=3)

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

        base_ev = Rouge1Evaluator(expert, qa_data_path=qa_data_path[qa_mode], sample_size=sample_size)
        questions, references, generated = base_ev.get_data(mode)

        for metric in selected_metrics:
            EvCls = {
                'rouge1': Rouge1Evaluator,
                'rougeL': RougeLEvaluator,
                'bert':   BertEvaluator,
                'sbert':  SbertEvaluator,
                'mover':  MoverEvaluator,
                'bleu':   BleuEvaluator
            }[metric]

            ev = EvCls(expert, qa_data_path[qa_mode], sample_size)
            ev._cache[mode] = (questions, references, generated)
            ev._infos[mode] = base_ev._infos.get(mode, [None]*len(questions))

            out_path = f"../results/{mode}_{metric}.json"
            ev.eval(mode, out_path)

            recs = json.load(open(out_path, 'r', encoding='utf-8'))
            values = [r[metric] for r in recs]
            summary.append({
                'mode': mode,
                'metric': metric,
                'average': float(np.mean(values)),
                'std': float(np.std(values))
            })

        del expert
        torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(summary)
    df_avg = df.pivot(index='mode', columns='metric', values='average')
    df_std = df.pivot(index='mode', columns='metric', values='std')

    os.makedirs('../results', exist_ok=True)
    with pd.ExcelWriter('../results/summary.xlsx') as writer:
        df_avg.to_excel(writer, sheet_name='average')
        df_std.to_excel(writer, sheet_name='std')

    print("✅ 모든 평가 완료: ../results/summary.xlsx (average/std 시트 포함)")
    elapsed = time.time() - start_time
    print(f"⏱ 전체 평가 완료: {elapsed/60:.2f}분")
