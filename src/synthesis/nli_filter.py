import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


class NLIFilter:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = device
        self.model.to(device)

    def predict(self, context: str, declarative: str) -> tuple[str, float]:
        """context와 declarative sentence 간 NLI 예측"""
        inputs = self.tokenizer(
            context,
            declarative,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        pred  = torch.argmax(probs).item()
        return LABEL_MAP[pred], round(probs[pred].item(), 4)

    def filter_batch(self, qa2d_list: list[dict]) -> tuple[list[dict], list[dict]]:
        all_results = []
        entailed    = []

        for i, item in enumerate(qa2d_list):
            context = item.get("agentic_chunk", "")   # context 대신 agentic_chunk 직접 사용
            label, score = self.predict(context, item["declarative"])
            result = {**item, "label": label, "score": score}
            all_results.append(result)
            if label == "entailment":
                entailed.append(result)

        return all_results, entailed