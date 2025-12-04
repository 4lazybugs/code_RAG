from abc import ABC, abstractmethod
from utils import get_config
from rouge_score.rouge_scorer import RougeScorer
from sentence_transformers import SentenceTransformer
import evaluate
import json
import os
from tqdm import tqdm
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print(f"[MODE={mode.upper()}] Detailed ../results saved to {output_path}")

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
            'bleu':   'bleu',
        }
        metric_name = mapping.get(orig)
        if metric_name is None:
            raise ValueError(f"Unknown evaluator class: {self.__class__.__name__}")

        scores = self.compute_scores(r, g)
        self.save_json(mode, q, r, g, {metric_name: scores}, output_path)