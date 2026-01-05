from .base import Evaluator, add_metric_key
from sentence_transformers import util, SentenceTransformer
import torch
from load_params import get_config

CFG = get_config()
device = "cuda" if torch.cuda.is_available() else "cpu"

def _clean(x) -> str:
    if x is None:
        return ""
    return str(x).strip()

@add_metric_key("sbert")
class SbertEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.sbert_model = SentenceTransformer(CFG.sbert_model_name, device=device)

    def score_once(self, data: dict) -> float:
        raise NotImplementedError("SBERT는 배치 평가를 사용. score_all()을 호출.")

    def score_all(self, batch_data: list[dict]) -> list[float]:
        references = [_clean(s.get("reference")) for s in batch_data]
        generated  = [_clean(s.get("generated")) for s in batch_data]

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
