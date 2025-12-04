from .base import BaseEvaluator
import evaluate

class BleurtEvaluator(BaseEvaluator):
    metric_key = 'bleurt'
    def __init__(self, expert, qa_data_path, sample_size=None):
        super().__init__(expert, qa_data_path, sample_size)
        self.metric = evaluate.load("bleurt", config_name="bleurt-20", module_type="metric")

    def compute_scores(self, references: list, generated: list) -> list:
        result = self.metric.compute(predictions=generated, references=references)
        return result["scores"]