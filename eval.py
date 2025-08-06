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
    def __init__(self, expert, qa_data_path: str, sample_size: int = None,
                 sbert_model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        self.expert = expert
        self.qa_data_path = qa_data_path
        self.sample_size = sample_size
        self._cache = {}
        self._infos = {}  # adaptive 모드의 info 저장

        # scorers / models
        self.rouge1_scorer = RougeScorer(['rouge1'], use_stemmer=False)
        self.rougeL_scorer = RougeScorer(['rougeL'], use_stemmer=False)
        self.sbert_model   = SentenceTransformer(sbert_model_name, device=str(device))
        self.f1_metric     = evaluate.load("squad")

    def get_data(self, mode: str):
        if mode not in self._cache:
            with open(self.qa_data_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            if self.sample_size and self.sample_size < len(dataset):
                import random
                random.seed(42) 
                dataset = random.sample(dataset, self.sample_size)

            # question 그대로 넘김 (instruction 없음)
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
        # adaptive 모드면 info를 저장해두고, 아니면 None 리스트
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
            # metric 값 추가
            for m_name, m_vals in metrics.items():
                rec[m_name] = float(m_vals[i])
            # adaptive 모드의 info가 있을 때만 JSON에 포함
            if infos[i] is not None:
                rec['info'] = infos[i]
            records.append(rec)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[MODE={mode.upper()}] Detailed results saved to {output_path}")

    @abstractmethod
    def compute_scores(self, references: list, generated: list) -> list:
        """metric별 점수 계산"""
        pass

    def eval(self, mode: str, output_path: str):
        q, r, g = self.get_data(mode)

        # 클래스 이름 기반 추출 후, 명시적 매핑
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
        korean_tokenizer = KoreanTokenizer()
        self.rouge1_scorer = RougeScorer(['rouge1'], use_stemmer=False, tokenizer=korean_tokenizer)

    def compute_scores(self, references: list, generated: list) -> list:
        return [self.rouge1_scorer.score(ref, gen)['rouge1'].fmeasure
                for ref, gen in zip(references, generated)]


class RougeLEvaluator(BaseEvaluator):
    metric_key = 'rougeL'

    def __init__(self, expert, qa_data_path, sample_size=None):
        super().__init__(expert, qa_data_path, sample_size)
        korean_tokenizer = KoreanTokenizer()
        self.rougeL_scorer = RougeScorer(['rougeL'], use_stemmer=False, tokenizer=korean_tokenizer)

    def compute_scores(self, references: list, generated: list) -> list:
        return [self.rougeL_scorer.score(ref, gen)['rougeL'].fmeasure
                for ref, gen in zip(references, generated)]



class BertEvaluator(BaseEvaluator):
    metric_key = 'bert'
    
    def compute_scores(self, references: list, generated: list) -> list:
        _, _, F1 = bert_score(
            generated, references,
            lang='en',
            model_type='bert-base-uncased',
            device=device,
            batch_size=16,
            verbose=False
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
        # 정확한 BLEURT 모델 지정 (예전처럼)
        self.metric = evaluate.load("bleurt", config_name="bleurt-20", module_type="metric")

    def compute_scores(self, references: list, generated: list) -> list:
        result = self.metric.compute(
            predictions=generated,
            references=references
        )
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

    # ---- Torch Sinkhorn OT ----
    def _sinkhorn_wasserstein(self, w_h, w_r, C, eps=0.1, n_iter=100):
        """
        w_h: (m,) hypothesis weights
        w_r: (n,) reference weights
        C  : (m,n) cost matrix (numpy)
        returns scalar Wasserstein distance
        """
        # to torch
        device = torch.device("cpu")
        a = torch.from_numpy(w_h).to(device)         # (m,)
        b = torch.from_numpy(w_r).to(device)         # (n,)
        M = torch.from_numpy(C).to(device)           # (m,n)

        K = torch.exp(-M / eps)                      # (m,n)
        u = torch.ones_like(a) / a.size(0)
        v = torch.ones_like(b) / b.size(0)

        for _ in range(n_iter):
            u = a / (K @ v + 1e-12)
            v = b / (K.t() @ u + 1e-12)

        P = torch.diag(u) @ K @ torch.diag(v)        # transport plan
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

        r_emb = self._embed_tokens(r_toks)  # (n,d)
        h_emb = self._embed_tokens(h_toks)  # (m,d)

        # Euclidean cost
        C = np.linalg.norm(h_emb[:, None, :] - r_emb[None, :, :], ord=2, axis=-1)  # (m,n)

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

    # 1) 파라미터 입력 필수!!
    qa_mode = 'soil'
    retriever_mode   = 'soil'
    selected_modes = [
        #'adaptive_3','adaptive_5',
        #'selfask_1',
        #'selfask_3',
        #'selfask_5'#,  
        #'partial_1','partial_3','partial_5',
        #'partial_10',
        #'partial_15',
        'raw_llm'
    ]
    selected_metrics = ['rouge1','rougeL','bert','sbert']
    sample_size      = 100
    qa_data_path = {
        'qna':  'qa_data/qa_agriculture.json',
        'crop': 'qa_data/qa_crop.json',
        'soil': 'qa_data/qa_soil_llama400b_kor.json'
    }

    # 2) k=1로 초기 retriever 세팅
    retr_qna, retr_crop, retr_soil, qna_sql, crop_sql = load_stores(ndocs=10)
    retr_map = {'qna': retr_qna, 'crop': retr_crop, 'soil': retr_soil}

    # 3) AdaptiveExpert는 한 번만 생성 (13B 모델 로딩)
    #adaptive = AdaptiveExpert(retr_map, retriever_mode)

    summary = []

    for mode in selected_modes:
        # -----------------------
        # 1) 모드별 Expert 설정 (매번 새로 생성)
        # -----------------------
        if mode.startswith('adaptive_'):
            k = int(mode.split('_')[1])
            r_q, r_c, r_s, _, _ = load_stores(ndocs=k)
            retr = {"qna": r_q, "crop": r_c, "soil": r_s}[retriever_mode]
            
            # 기존 인스턴스의 retriever와 args만 교체
            adaptive.retriever = retr
            adaptive.args.ndocs = k
            expert = adaptive
            
        elif mode.startswith('partial_'):
            k = int(mode.split('_')[1])
            r_q, r_c, r_s, _, _ = load_stores(ndocs=k)
            retr = {"qna": r_q, "crop": r_c, "soil": r_s}[retriever_mode]
            
            expert = PartialExpert(retr_map, retriever_mode)
            expert.retriever = retr
            
        elif mode.startswith('selfask'):
            k = int(mode.split('_')[1])  # 예: 'selfask_5' → 5
            expert = SelfAskExpert(retr_map, retriever_mode, max_iter=k)
            
        elif mode == 'all':
            expert = AllExpert(retr_map, retriever_mode)
        elif mode == 'raw_llm':
            expert = RawLlmExpert(retr_map, retriever_mode)
        elif mode == 'sql':
            expert = SqlExpert(retr_map, retriever_mode, qna_sql, crop_sql)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # -----------------------
        # 2) 한 번만 답 생성
        # -----------------------
        base_ev = Rouge1Evaluator(
            expert,
            qa_data_path=qa_data_path[qa_mode],
            sample_size=sample_size
        )
        questions, references, generated = base_ev.get_data(mode)

        # -----------------------
        # 3) metric별 평가
        # -----------------------
        for metric in selected_metrics:
            EvCls = {
                'rouge1': Rouge1Evaluator,
                'rougeL': RougeLEvaluator,
                'bert':   BertEvaluator,
                'sbert':  SbertEvaluator,
                'mover':  MoverEvaluator,  # <- 추가
                #'bleurt': BleurtEvaluator,
            }[metric]
            ev = EvCls(expert, qa_data_path[retriever_mode], sample_size)
            ev._cache[mode] = (questions, references, generated)
            ev._infos[mode] = base_ev._infos.get(mode, [None]*len(questions))

            out_path = f"results/{mode}_{metric}.json"
            ev.eval(mode, out_path)

            recs = json.load(open(out_path, 'r', encoding='utf-8'))
            # metric 값들만 리스트로 추출
            values = [r[metric] for r in recs]
            # 평균과 표준편차 계산 (population std)
            avg = float(np.mean(values))
            std = float(np.std(values))
            summary.append({
                'mode': mode,
                'metric': metric,
                'average': avg,
                'std': std
            })

        # -----------------------
        # 4) 평가 끝난 후 expert 강제 정리 (모든 모드에 적용)
        # -----------------------
        # AdaptiveExpert는 재사용하므로 삭제하지 않음
        if not mode.startswith('adaptive_'):
            del expert
            torch.cuda.empty_cache()
            gc.collect()

    # -----------------------
    # 5) 결과 저장
    # -----------------------
    # DataFrame 생성
    df = pd.DataFrame(summary)
    # 평균 시트
    df_avg = df.pivot(index='mode', columns='metric', values='average')
    # 표준편차 시트
    df_std = df.pivot(index='mode', columns='metric', values='std')

    os.makedirs('results', exist_ok=True)
    # 두 시트를 가진 Excel 파일로 저장
    with pd.ExcelWriter('results/summary.xlsx') as writer:
        df_avg.to_excel(writer, sheet_name='average')
        df_std.to_excel(writer, sheet_name='std')
    print("✅ 모든 평가 완료: results/summary.xlsx (average/std 시트 포함)")

    elapsed = time.time() - start_time
    print(f"⏱ 전체 평가 완료: {elapsed/60:.2f}분")
