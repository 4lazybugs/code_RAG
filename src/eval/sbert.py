from __future__ import annotations
from typing import Dict, Any
from sentence_transformers import util, SentenceTransformer
import torch
from .base import Evaluator

device = "cuda" if torch.cuda.is_available() else "cpu"

class SBERTEvaluator(Evaluator):
    def __init__(self, sbert_model_name):
        super().__init__()
        self.sbert_model = SentenceTransformer(sbert_model_name, device=device)
        self.metric_key = "SBERT"

    def score_once(self, data: Dict[str, Any]) -> float:
        ref = self._normalize(self._get_field(data, "reference", "answer"))
        gen = self._normalize(self._get_field(data, "generated", "gen_answer"))

        # ref와 gen중 둘 중 하나라도 없다면 보수적으로 0점 return
        if not ref: return 0.0
        if not gen: return 0.0

        with torch.no_grad():
            ref_emb = self.sbert_model.encode([ref], convert_to_tensor=True, normalize_embeddings=True)
            gen_emb = self.sbert_model.encode([gen], convert_to_tensor=True, normalize_embeddings=True)
            # 코사인 유사도 계산
            sim = util.cos_sim(gen_emb, ref_emb)[0, 0] 

        return float(sim.cpu().item())