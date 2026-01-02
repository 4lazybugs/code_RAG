from .base import Evaluator
from sentence_transformers import util, SentenceTransformer
import torch
from load_params import get_config

CFG = get_config()

device = "cuda" if torch.cuda.is_available() else "cpu"   # 원하시면 cpu로 고정해도 됨

class SbertEvaluator(Evaluator):
    metric_key = "sbert"

    def __init__(self):
        super().__init__()

        # ✅ config에서 읽힌 args를 Evaluator가 보관한다고 가정(self.args)
        self.sbert_model = SentenceTransformer(
            CFG.sbert_model_name,
            device=device
        )

    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs: list) -> list:
        with torch.no_grad():
            ref_emb = self.sbert_model.encode(
                references,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=16
            )
            gen_emb = self.sbert_model.encode(
                generated,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=16
            )
            sims = util.cos_sim(gen_emb, ref_emb).diagonal()
        return sims.cpu().numpy().tolist()
