from .base import BaseEvaluator
from bert_score import score as bert_score
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertEvaluator(BaseEvaluator):
    metric_key = 'bert'
    def compute_scores(self, references: list, generated: list) -> list:
        P, R, F1 = bert_score(
            generated, references,
            lang='ko',
            model_type= self.args.bert_model_name,
            device=device,
            batch_size=16,
            rescale_with_baseline=True
        )
        return F1.cpu().numpy().tolist()