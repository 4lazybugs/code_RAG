import json
import re
from typing import List, Tuple, Literal

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Judge_HF:
    def __init__(
        self,
        relv_prompt,
        faith_prompt,
        model_name: str,
        max_new_tokens: int = 128,
    ):
        self.max_new_tokens = max_new_tokens
        self.relv_prompt = relv_prompt
        self.faith_prompt = faith_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _select_prompt(self, prompt_type: Literal["relv", "faith"]):
        if prompt_type == "relv":
            return self.relv_prompt
        if prompt_type == "faith":
            return self.faith_prompt

    def _extract_json(self, text: str) -> Tuple[int, float]:
        text = text.strip()

        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                score = max(0.0, min(1.0, float(obj.get("score", 0.0))))
                label = int(obj.get("label", 1 if score >= 0.5 else 0))
                return (1 if label == 1 else 0), score
            except Exception:
                pass

        num_match = re.search(r"([01](?:\.\d+)?)", text)
        if num_match:
            score = max(0.0, min(1.0, float(num_match.group(1))))
            return (1 if score >= 0.5 else 0), score

        return 0, 0.0

    @torch.no_grad()
    def _score_one(
        self,
        doc: str,
        claim: str,
        prompt_type: Literal["relv", "faith"],
    ) -> Tuple[int, float]:
        prompt = self._select_prompt(prompt_type).format(doc=doc, claim=claim)

        messages = [
            {"role": "system", "content": "You are a judge."},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return self._extract_json(gen_text)

    def score(
        self,
        docs: List[str],
        claims: List[str],
        prompt_type: Literal["relv", "faith"],
    ) -> Tuple[List[int], List[float], List[str], List[List[float]]]:
        if len(docs) != len(claims):
            raise ValueError("docs와 claims의 길이는 같아야 합니다.")

        pred_labels, probs, used_chunks, support_prob_per_chunk = [], [], [], []

        for doc, claim in zip(docs, claims):
            label, score = self._score_one(doc, claim, prompt_type)
            pred_labels.append(label)
            probs.append(score)
            used_chunks.append(doc)
            support_prob_per_chunk.append([score])

        return pred_labels, probs, used_chunks, support_prob_per_chunk