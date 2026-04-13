from typing import List, Union, Sequence
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 512):
        self.model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, docs: List[str]) -> List[float]:
        if not docs:
            return []

        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs)

        # numpy array여도 list로 맞춰서 반환
        return [float(s) for s in scores]