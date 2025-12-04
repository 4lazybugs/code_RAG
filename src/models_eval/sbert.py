from .base import BaseEvaluator
from sentence_transformers import util

class SbertEvaluator(BaseEvaluator):
    metric_key = 'sbert'
    def compute_scores(self, references: list, generated: list) -> list:
        ref_emb = self.sbert_model.encode(references, convert_to_tensor=True, batch_size=16)
        gen_emb = self.sbert_model.encode(generated, convert_to_tensor=True, batch_size=16)
        sims = util.cos_sim(gen_emb, ref_emb).diagonal().cpu().numpy().tolist()
        return sims
