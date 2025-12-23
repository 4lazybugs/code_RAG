from abc import ABC, abstractmethod
from utils import get_config
from rouge_score.rouge_scorer import RougeScorer
from sentence_transformers import SentenceTransformer
import evaluate
import json
import os
from tqdm import tqdm
import torch

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

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
            qa_id = [item['id'] for item in dataset]
            #breakpoint()
            self._cache[mode] = (questions, references, generated, qa_id)
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
            
            # ✅ 여기만 추가: ans가 AIMessage 등일 때 문자열로 정규화
            if hasattr(ans, "content"):          # LangChain AIMessage/BaseMessage
                ans = ans.content
            elif isinstance(ans, dict):          # 혹시 dict로 오는 경우
                ans = ans.get("content") or ans.get("text") or ans.get("answer") or str(ans)
            ans = "" if ans is None else str(ans)

            preds.append(ans)
            infos.append(info)
        self._infos[mode] = infos
        return preds

    @abstractmethod
    def compute_scores(self, references: list, generated: list) -> list:
        pass

    def save_score(self, mode: str, output_path: str):
        q, r, g, qa_id = self._cache[mode]
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

        scores = self.compute_scores(r, g) # (references, generated)의 평가지표 점수(bert, sbert 등) 계산
        self.save_json(mode, qa_id, q, r, g, {metric_name: scores}, output_path)
