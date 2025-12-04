import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from rag import AllExpert, PartialExpert, SqlExpert, RawLlmExpert, SelfAskExpert
from rag import MultiCosineRetriever                # ✅ 추가
from retriever import load_semantic_retrievers  # ✅ 추가
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
# Expert 클래스들
from rag import AllExpert, PartialExpert, SqlExpert, RawLlmExpert, AdaptiveExpert, SelfAskExpert

CFG = get_config()
# store_db 전역 임베딩 모델 주입
openai.api_key = os.getenv("OPENAI_API_KEY")


class KoreanTokenizer(Tokenizer):
    def __init__(self):
        self.okt = Okt()

    def tokenize(self, text):
        return self.okt.morphs(text)


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    start_time = time.time()

    # 1) 파라미터 입력
    qa_mode = 'semantic'          # 데이터셋 선택: 'qna' / 'crop' / 'soil' / 'bugs' / 'farm'
    retriever_mode = 'semantic_multi'   # 리트리버 선택: 'qna' / 'crop' / 'soil' / 'bugs' / 'farm'
    
    selected_modes = [
        #'partial_1',
        #'partial_3',
        #'partial_5',
        'partial_10',
        #'selfask_3',
        #'selfask_5',
        'raw_llm'
    ]
    selected_metrics = ['rouge1','rougeL','bert','sbert','bleu']
    sample_size      = None  # None이면 전체 데이터셋 사용
    qa_data_path = {
        'qna':  'qa_data/qa_agriculture.json',
        'crop': 'qa_data/qa_crop.json',
        'soil': 'qa_data/qa_soil_llama400b.json',
        #'soil': 'qa_data/qa_soil_gemini_100.json',
        'bugs': 'qa_data/qa_bug_gpt5_advanced_kor.json',
        'farm': 'qa_data/qa_farm.json',
        'semantic': 'qa_data/gpt_qa.json'
    }

    # 2) 초기 retriever 세팅 (dict 반환 사용)
    ndocs_init = 10
    semantic_folder_retrievers = load_semantic_retrievers(ndocs=ndocs_init)

    semantic_multi = MultiCosineRetriever(
        retrievers=semantic_folder_retrievers,
        k_each=10,   # ✅ multi retriever 개당 찾는 수
        top_k=3      # ✅ 전체에서 최종 몇 개만 쓸지
    )

    # 🔹 전체 retriever 맵 구성
    retr_map = {
        "semantic_multi": semantic_multi,
        #"qna": retrievers.get("qna"),
        #"crop": retrievers.get("crop"),
        #"soil": retrievers.get("soil"),
        #"bugs": retrievers.get("bugs"),
        #"farm": retrievers.get("farm"),
        # "semantic": retrievers.get("semantic"),  # 필요하면 유지
    }

    # 3) (선택) AdaptiveExpert는 필요시만 생성
    # adaptive = AdaptiveExpert(retr_map, retriever_mode)

    summary = []

    for mode in selected_modes:
        # 1) 모드별 Expert 설정
        if mode.startswith('partial_'):

            if retriever_mode == "semantic_multi":
                # ✅ 숫자 k는 그냥 이름일 뿐, 실제 retriever는
                #    위에서 만든 semantic_multi( k_each=10, top_k=3 ) 그대로 사용
                expert = PartialExpert(retr_map, retriever_mode)
                # PartialExpert.setup()에서 이미 retriever_map["semantic_multi"]를 가져오므로
                # expert.retriever 따로 덮어쓸 필요 없음

            else:
                # 기존 qna/crop/... 모드일 때는 숫자 k를 계속 사용
                k = int(mode.split('_')[1])
                stores_k = semantic_folder_retrievers(ndocs=k)
                retr = stores_k[retriever_mode]
                expert = PartialExpert(retr_map, retriever_mode)
                expert.retriever = retr


        elif mode.startswith('selfask_'):
            k = int(mode.split('_')[1])
            expert = SelfAskExpert(retr_map, retriever_mode, max_iter=k)

        elif mode == 'all':
            expert = AllExpert(retr_map, retriever_mode)
        elif mode == 'raw_llm':
            expert = RawLlmExpert(retr_map, retriever_mode)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2) 한 번만 답 생성 (base 캐시)
        base_ev = Rouge1Evaluator(
            expert,
            qa_data_path=qa_data_path[qa_mode],
            sample_size=sample_size
        )
        questions, references, generated = base_ev.get_data(mode)

        # 3) metric별 평가
        for metric in selected_metrics:
            EvCls = {
                'rouge1': Rouge1Evaluator,
                'rougeL': RougeLEvaluator,
                'bert':   BertEvaluator,
                'sbert':  SbertEvaluator,
                'mover':  MoverEvaluator,
                'bleu':   BleuEvaluator
            }[metric]

            # ⚠️ 평가 데이터셋은 qa_mode 기준으로 유지
            ev = EvCls(expert, qa_data_path[qa_mode], sample_size)
            ev._cache[mode] = (questions, references, generated)
            ev._infos[mode] = base_ev._infos.get(mode, [None]*len(questions))

            out_path = f"results/{mode}_{metric}.json"
            ev.eval(mode, out_path)

            recs = json.load(open(out_path, 'r', encoding='utf-8'))
            values = [r[metric] for r in recs]
            avg = float(np.mean(values))
            std = float(np.std(values))
            summary.append({
                'mode': mode,
                'metric': metric,
                'average': avg,
                'std': std
            })

        # 4) 정리
        if not mode.startswith('adaptive_'):
            del expert
            torch.cuda.empty_cache()
            gc.collect()

    # 5) 결과 저장 (average / std 시트)
    df = pd.DataFrame(summary)
    df_avg = df.pivot(index='mode', columns='metric', values='average')
    df_std = df.pivot(index='mode', columns='metric', values='std')

    os.makedirs('results', exist_ok=True)
    with pd.ExcelWriter('results/summary.xlsx') as writer:
        df_avg.to_excel(writer, sheet_name='average')
        df_std.to_excel(writer, sheet_name='std')
    print("✅ 모든 평가 완료: results/summary.xlsx (average/std 시트 포함)")

    elapsed = time.time() - start_time
    print(f"⏱ 전체 평가 완료: {elapsed/60:.2f}분")
