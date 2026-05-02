import torch
from typing import Callable
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

NLIStrategy = Callable[[dict], tuple[str, str]]


# ── 전략 함수들 ──────────────────────────────────────────────────
def nli_CD(item: dict) -> tuple[str, str]:
    premise = " ".join(filter(None, [
        item.get("md_summary", ""),
        item.get("raw_chunk", ""),
    ]))
    return premise, item["declarative"]


def nli_QA(item: dict) -> tuple[str, str]:
    return item["question"], item["answer"]


# ── NLIFilter ────────────────────────────────────────────────────
class NLIFilter:
    def __init__(self, model_name: str, strategy: NLIStrategy, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.strategy = strategy

    def predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
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

    def filter_batch(self, qa2d_list: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        all_results, entailed, neutral, contradiction = [], [], [], []

        for item in qa2d_list:
            premise, hypothesis = self.strategy(item)

            inputs = self.tokenizer(
                premise, hypothesis,
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

            result = {
                **item,
                "score_entailment":      round(probs[0].item(), 4),
                "score_neutral":         round(probs[1].item(), 4),
                "score_contradiction":   round(probs[2].item(), 4),
            }
            all_results.append(result)
            if LABEL_MAP[pred] == "entailment":
                entailed.append(result)
            elif LABEL_MAP[pred] == "neutral":
                neutral.append(result)
            else:
                contradiction.append(result)

        return all_results, entailed, neutral, contradiction