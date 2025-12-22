from .base import BaseEvaluator
from sentence_transformers import util, SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"   # 원하시면 cpu로 고정해도 됨

class SbertEvaluator(BaseEvaluator):
    metric_key = "sbert"

    def __init__(self, expert, qa_data_path=None, sample_size=None):
        super().__init__(expert, qa_data_path=qa_data_path, sample_size=sample_size)

        # ✅ config에서 읽힌 args를 BaseEvaluator가 보관한다고 가정(self.args)
        self.sbert_model = SentenceTransformer(
            self.args.sbert_model_name,
            device=device
        )

    def compute_scores(self, references: list, generated: list) -> list:
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
