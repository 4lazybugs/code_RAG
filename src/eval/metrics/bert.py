from .base import Evaluator
from bert_score import score as bert_score
import re
import torch
from load_params import get_config

CFG = get_config()

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

def _clean_for_bert(x: str, max_chars: int = 3000) -> str:
    if x is None:
        return ""
    return str(x).strip()[:max_chars]


class BertEvaluator(Evaluator):
    metric_key = "bert"

    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs: list) -> list:
        references = [_clean_for_bert(t) for t in references]
        generated  = [_clean_for_bert(t) for t in generated]

        with torch.no_grad():
            P, R, F1 = bert_score(
                generated, references,
                lang="ko",
                model_type=CFG.bert_model_name,
                device=device,
                batch_size=1,                  # ✅ 16 -> 1
                rescale_with_baseline=False,   # ✅ True -> False (아래 설명)
            )
        return F1.cpu().numpy().tolist()
