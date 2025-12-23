import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import pandas as pd
import torch
import time
import gc
import numpy as np
from konlpy.tag import Okt
from rouge_score.tokenizers import Tokenizer

from models_eval import Rouge1Evaluator, RougeLEvaluator, BleuEvaluator
from models_eval import BertEvaluator, SbertEvaluator, MoverEvaluator
from pathlib import Path


class KoreanTokenizer(Tokenizer):
    def __init__(self):
        self.okt = Okt()

    def tokenize(self, text):
        return self.okt.morphs(text)


if __name__ == '__main__':
    start_time = time.time()

    results_dir = (Path(__file__).resolve().parent / ".." / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "score").mkdir(parents=True, exist_ok=True)  # ✅ 저장 폴더 보장

    selected_modes = ['partial_10', 'raw_llm']
    selected_metrics = ['rouge1', 'rougeL', 'bert', 'sbert', 'bleu']

    # ✅ 원본 BaseEvaluator 시그니처를 만족시키기 위한 최소 변수(실제로는 cache hit이라 사용 안 됨)
    qa_mode = "cleaned"
    sample_size = None
    qa_data_path = {qa_mode: "__unused__"}  # cache 주입 후 get_data가 파일을 안 읽으므로 더미로 OK

    summary = []

    for mode in selected_modes:
        # --------------------------------------------------
        # ✅ results/qag/{mode}.json 로드해서 (q,r,g,id,info) 구성
        # --------------------------------------------------
        qag_path = results_dir / "qag" / f"{mode}.json"
        if not qag_path.exists():
            raise FileNotFoundError(f"QAG file not found: {qag_path}")

        with open(qag_path, "r", encoding="utf-8") as f:
            recs_qag = json.load(f)

        questions  = [r["question"] for r in recs_qag]
        references = [r["reference"] for r in recs_qag]
        generated  = [r["generated"] for r in recs_qag]
        qa_id      = [r["id"] for r in recs_qag]
        infos      = [r.get("info") for r in recs_qag]  # 없으면 None
        # --------------------------------------------------

        for metric in selected_metrics:
            EvCls = {
                'rouge1': Rouge1Evaluator,
                'rougeL': RougeLEvaluator,
                'bert':   BertEvaluator,
                'sbert':  SbertEvaluator,
                'mover':  MoverEvaluator,
                'bleu':   BleuEvaluator
            }[metric]

            # ✅ 평가-only: expert는 사용하지 않으므로 None
            ev = EvCls(expert=None, qa_data_path=qa_data_path[qa_mode], sample_size=sample_size)

            # ✅ qag json에서 로드한 값으로 캐시 주입
            ev._cache[mode] = (questions, references, generated, qa_id)
            ev._infos[mode] = infos

            out_path = results_dir / "score" / f"score_{mode}_{metric}.json"
            print(f"[RUN] {mode} {metric} start (score-only)", flush=True)

            # ✅ BaseEvaluator에 구현된 함수 사용 (eval 아님)
            ev.save_score(mode, str(out_path))

            with open(out_path, "r", encoding="utf-8") as f:
                recs = json.load(f)

            # 저장 키가 metric과 동일하다는 가정(현재 selected_metrics 기준 OK)
            values = [row[metric] for row in recs]
            summary.append({
                'mode': mode,
                'metric': metric,
                'average': float(np.mean(values)),
                'std': float(np.std(values))
            })

        torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(summary)
    df_avg = df.pivot(index='mode', columns='metric', values='average')
    df_std = df.pivot(index='mode', columns='metric', values='std')

    with pd.ExcelWriter(results_dir / 'summary.xlsx') as writer:
        df_avg.to_excel(writer, sheet_name='average')
        df_std.to_excel(writer, sheet_name='std')

    print(f"✅ 모든 평가 완료: {results_dir / 'summary.xlsx'} (average/std 시트 포함)")
    elapsed = time.time() - start_time
    print(f"⏱ 전체 평가 완료: {elapsed/60:.2f}분")
