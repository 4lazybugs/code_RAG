import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import pandas as pd
import torch
from tqdm import tqdm
from bert_score import score as bert_score
from rouge_score.rouge_scorer import RougeScorer
import evaluate

from sentence_transformers import SentenceTransformer, util
from abc import ABC, abstractmethod
import time
import gc
import openai
import re, math
import numpy as np
from collections import Counter
from bleurt import score

from src.utils import get_config
import src.store_db as store_db  # 모듈 자체를 import
CFG = get_config()
# store_db 전역 임베딩 모델 주입
store_db.EMBED_MODEL = getattr(CFG, "embedor_model_name", None)

# Expert 클래스들
from rag import AllExpert, PartialExpert, SqlExpert, RawLlmExpert, AdaptiveExpert, SelfAskExpert
from src.store_db import load_stores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
openai.api_key = os.getenv("OPENAI_API_KEY")

from konlpy.tag import Okt
from rouge_score.tokenizers import Tokenizer

class KoreanTokenizer(Tokenizer):
    def __init__(self):
        self.okt = Okt()

    def tokenize(self, text):
        return self.okt.morphs(text)

class BaseEvaluator(ABC):
    """
    추상 베이스 평가기 클래스:
      - get_data, generate: 공통 구현
      - save_json(): 개별 문항 JSON 저장 (adaptive 모드의 info 포함)
      - eval(): metric별 평가 로직 구현
      - compute_scores(): metric별 점수 계산 (새로 추가)
    """
    def __init__(self, expert, qa_data_path: str, sample_size: int = None):
        self.args = get_config()
        self.sbert_model_name = self.args.sbert_model_name
        self.expert = expert
        self.qa_data_path = qa_data_path
        self.sample_size = sample_size
        self._cache = {}
        self._infos = {}  # adaptive 모드의 info 저장

        # scorers / models
        self.rouge1_scorer = RougeScorer(['rouge1'], use_stemmer=False)
        self.rougeL_scorer = RougeScorer(['rougeL'], use_stemmer=False)
        self.sbert_model   = SentenceTransformer(self.sbert_model_name, device=str(device))
        self.f1_metric     = evaluate.load("squad")

    def get_data(self, mode: str):
        if mode not in self._cache:
            with open(self.qa_data_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            if self.sample_size and self.sample_size < len(dataset):
                import random
                random.seed(42) 
                dataset = random.sample(dataset, self.sample_size)

            questions  = [item['question'] for item in dataset]
            references = [str(item['answer']) for item in dataset]
            generated  = self.generate(questions, mode)
            self._cache[mode] = (questions, references, generated)
        return self._cache[mode]

    def generate(self, questions: list, mode: str) -> list:
        preds = []
        infos = []
        for q in tqdm(questions, desc=f"Generating ({mode})"):
            res = self.expert.handle(q)
            if isinstance(res, tuple):
                ans, info = res
            else:
                ans, info = res, None
            preds.append(ans)
            infos.append(info)
        self._infos[mode] = infos
        return preds

    def save_json(self, mode: str,
                  questions: list,
                  references: list,
                  generated: list,
                  metrics: dict,
                  output_path: str):
        records = []
        infos = self._infos.get(mode, [None] * len(questions))
        n = len(questions)
        for i in range(n):
            rec = {
                'question':  questions[i],
                'reference': references[i],
                'generated': generated[i],
            }
            for m_name, m_vals in metrics.items():
                rec[m_name] = float(m_vals[i])
            if infos[i] is not None:
                rec['info'] = infos[i]
            records.append(rec)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[MODE={mode.upper()}] Detailed results saved to {output_path}")

    @abstractmethod
    def compute_scores(self, references: list, generated: list) -> list:
        pass

    def eval(self, mode: str, output_path: str):
        q, r, g = self.get_data(mode)
        orig = self.__class__.__name__.replace('Evaluator','').lower()
        mapping = {
            'rouge1': 'rouge1',
            'rougel': 'rougeL',
            'bert':   'bert',
            'sbert':  'sbert',
            'mover':  'mover', 
            'bleurt': 'bleurt',
        }
        metric_name = mapping.get(orig)
        if metric_name is None:
            raise ValueError(f"Unknown evaluator class: {self.__class__.__name__}")

        scores = self.compute_scores(r, g)
        self.save_json(mode, q, r, g, {metric_name: scores}, output_path)


class Rouge1Evaluator(BaseEvaluator):
    metric_key = 'rouge1'
    def __init__(self, expert, qa_data_path, sample_size=None):
        super().__init__(expert, qa_data_path, sample_size)
        self.rouge1_scorer = RougeScorer(['rouge1'], use_stemmer=False)

    def compute_scores(self, references: list, generated: list) -> list:
        return [self.rouge1_scorer.score(ref, gen)['rouge1'].fmeasure
                for ref, gen in zip(references, generated)]


class RougeLEvaluator(BaseEvaluator):
    metric_key = 'rougeL'
    def __init__(self, expert, qa_data_path, sample_size=None):
        super().__init__(expert, qa_data_path, sample_size)
        self.rougeL_scorer = RougeScorer(['rougeL'], use_stemmer=False)

    def compute_scores(self, references: list, generated: list) -> list:
        return [self.rougeL_scorer.score(ref, gen)['rougeL'].fmeasure
                for ref, gen in zip(references, generated)]


class BertEvaluator(BaseEvaluator):
    metric_key = 'bert'
    def compute_scores(self, references: list, generated: list) -> list:
        P, R, F1 = bert_score(
            generated, references,
            #lang='ko',
            model_type= self.args.bert_model_name,
            device=device,
            batch_size=16,
            rescale_with_baseline=False
        )
        return F1.cpu().numpy().tolist()


class SbertEvaluator(BaseEvaluator):
    metric_key = 'sbert'
    def compute_scores(self, references: list, generated: list) -> list:
        ref_emb = self.sbert_model.encode(references, convert_to_tensor=True, batch_size=16)
        gen_emb = self.sbert_model.encode(generated, convert_to_tensor=True, batch_size=16)
        sims = util.cos_sim(gen_emb, ref_emb).diagonal().cpu().numpy().tolist()
        return sims


class BleurtEvaluator(BaseEvaluator):
    metric_key = 'bleurt'
    def __init__(self, expert, qa_data_path, sample_size=None):
        super().__init__(expert, qa_data_path, sample_size)
        self.metric = evaluate.load("bleurt", config_name="bleurt-20", module_type="metric")

    def compute_scores(self, references: list, generated: list) -> list:
        result = self.metric.compute(predictions=generated, references=references)
        return result["scores"]


class MoverEvaluator(BaseEvaluator):
    metric_key = 'mover'
    _tok_pat = re.compile(r"\w+|\S", flags=re.UNICODE)

    def _tokenize(self, text: str):
        return self._tok_pat.findall(text.lower())

    def _get_idf_dict(self, texts):
        n = len(texts)
        df = Counter()
        for t in texts:
            df.update(set(self._tokenize(t)))
        return {w: math.log((n + 1) / (df[w] + 1)) for w in df}

    def _embed_tokens(self, tokens):
        if not tokens:
            return np.zeros((1, self.sbert_model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.sbert_model.encode(tokens, convert_to_numpy=True, show_progress_bar=False)

    def _sinkhorn_wasserstein(self, w_h, w_r, C, eps=0.1, n_iter=100):
        device = torch.device("cpu")
        a = torch.from_numpy(w_h).to(device)
        b = torch.from_numpy(w_r).to(device)
        M = torch.from_numpy(C).to(device)
        K = torch.exp(-M / eps)
        u = torch.ones_like(a) / a.size(0)
        v = torch.ones_like(b) / b.size(0)
        for _ in range(n_iter):
            u = a / (K @ v + 1e-12)
            v = b / (K.t() @ u + 1e-12)
        P = torch.diag(u) @ K @ torch.diag(v)
        dist = (P * M).sum().item()
        return dist

    def _one_pair_score(self, ref, hyp, idf_ref, idf_hyp, stop_words=None):
        r_toks = self._tokenize(ref)
        h_toks = self._tokenize(hyp)
        if stop_words:
            r_toks = [t for t in r_toks if t not in stop_words]
            h_toks = [t for t in h_toks if t not in stop_words]
        if len(r_toks) == 0 or len(h_toks) == 0:
            return 0.0

        r_emb = self._embed_tokens(r_toks)
        h_emb = self._embed_tokens(h_toks)
        C = np.linalg.norm(h_emb[:, None, :] - r_emb[None, :, :], ord=2, axis=-1)
        w_r = np.array([idf_ref.get(t, 1.0) for t in r_toks], dtype=np.float64)
        w_h = np.array([idf_hyp.get(t, 1.0) for t in h_toks], dtype=np.float64)
        w_r = w_r / (w_r.sum() + 1e-12)
        w_h = w_h / (w_h.sum() + 1e-12)
        dist = self._sinkhorn_wasserstein(w_h, w_r, C, eps=0.1, n_iter=50)
        return 1.0 - float(dist)

    def compute_scores(self, references: list, generated: list) -> list:
        idf_ref = self._get_idf_dict(references)
        idf_hyp = self._get_idf_dict(generated)
        return [self._one_pair_score(r, g, idf_ref, idf_hyp) for r, g in zip(references, generated)]


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    start_time = time.time()

    # 1) 파라미터 입력
    qa_mode = 'soil'          # 데이터셋 선택: 'qna' / 'crop' / 'soil' / 'bugs' / 'farm'
    retriever_mode = 'soil'   # 리트리버 선택: 'qna' / 'crop' / 'soil' / 'bugs' / 'farm'
    
    selected_modes = [
        #'partial_1',
        #'partial_3',
        #'partial_5',
        'partial_10',
        'selfask_1',
        #'selfask_3',
        #'selfask_5',
        'raw_llm'
    ]
    selected_metrics = ['rouge1','rougeL','bert','sbert']
    sample_size      = 1  # None이면 전체 데이터셋 사용
    qa_data_path = {
        'qna':  'qa_data/qa_agriculture.json',
        'crop': 'qa_data/qa_crop.json',
        'soil': 'qa_data/qa_soil_llama400b.json',
        #'soil': 'qa_data/qa_soil_gemini_100.json',
        'bugs': 'qa_data/qa_bug_gpt5_advanced_kor.json',
        'farm': 'qa_data/qa_farm.json'
    }

    # 2) 초기 retriever 세팅 (dict 반환 사용)
    stores = load_stores(ndocs=10)
    retr_map = {k: stores[k] for k in ['qna','crop','soil','bugs','farm'] if k in stores}
    qna_sql = stores['qna_sql']
    crop_sql = stores['crop_sql']

    # 3) (선택) AdaptiveExpert는 필요시만 생성
    # adaptive = AdaptiveExpert(retr_map, retriever_mode)

    summary = []

    for mode in selected_modes:
        # 1) 모드별 Expert 설정
        if mode.startswith('adaptive_'):
            k = int(mode.split('_')[1])
            stores_k = load_stores(ndocs=k)
            retr = stores_k[retriever_mode]
            adaptive.retriever = retr
            adaptive.args.ndocs = k
            expert = adaptive

        elif mode.startswith('partial_'):
            k = int(mode.split('_')[1])
            stores_k = load_stores(ndocs=k)
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
        elif mode == 'sql':
            expert = SqlExpert(retr_map, retriever_mode, qna_sql, crop_sql)
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
                # 'bleurt': BleurtEvaluator,
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
