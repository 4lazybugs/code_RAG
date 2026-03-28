import json
import re
from typing import List, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Judge_LM:
    """
    MiniCheck 대체용 Judge 클래스
    - score(docs=[...], claims=[...]) 인터페이스 유지
    - 반환 형식:
      (pred_label, prob, used_chunk, support_prob_per_chunk)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir: str = "./ckpts",
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _build_prompt(self, doc: str, claim: str) -> str:
        return f"""
You are a careful evaluator.

Task:
Given a DOCUMENT and a CLAIM, determine how strongly the DOCUMENT supports the CLAIM.

Return ONLY valid JSON in this exact format:
{{"score": <float between 0 and 1>, "label": <0 or 1>}}

Scoring guide:
- 1.0 = fully supported by the document
- 0.8 = mostly supported
- 0.5 = partially supported / uncertain
- 0.2 = weakly supported
- 0.0 = not supported or contradicted

Rules:
- Use only the document.
- Do not use outside knowledge.
- If the claim is not clearly supported, give a low score.
- Output JSON only.

DOCUMENT:
{doc}

CLAIM:
{claim}
""".strip()

    def _extract_json(self, text: str) -> Tuple[int, float]:
        """
        모델 출력에서 JSON 추출
        실패 시 보수적으로 낮은 점수 반환
        """
        text = text.strip()

        # 가장 먼저 JSON 블록 찾기
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                score = float(obj.get("score", 0.0))
                label = int(obj.get("label", 1 if score >= 0.5 else 0))
                score = max(0.0, min(1.0, score))
                label = 1 if label == 1 else 0
                return label, score
            except Exception:
                pass

        # fallback: 숫자만 찾아보기
        num_match = re.search(r'([01](?:\.\d+)?)', text)
        if num_match:
            score = float(num_match.group(1))
            score = max(0.0, min(1.0, score))
            label = 1 if score >= 0.5 else 0
            return label, score

        return 0, 0.0

    @torch.no_grad()
    def _score_one(self, doc: str, claim: str) -> Tuple[int, float]:
        prompt = self._build_prompt(doc, claim)

        messages = [
            {"role": "system", "content": "You are a precise factual judge."},
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
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        label, score = self._extract_json(gen_text)
        return label, score

    def score(
        self,
        docs: List[str],
        claims: List[str],
    ) -> Tuple[List[int], List[float], List[str], List[List[float]]]:
        """
        MiniCheck와 동일한 반환 형식 맞춤:
        pred_label, prob, used_chunk, support_prob_per_chunk
        """
        if len(docs) != len(claims):
            raise ValueError("docs와 claims의 길이는 같아야 합니다.")

        pred_labels = []
        probs = []
        used_chunks = []
        support_prob_per_chunk = []

        for doc, claim in zip(docs, claims):
            label, score = self._score_one(doc, claim)

            pred_labels.append(label)
            probs.append(score)
            used_chunks.append(doc)
            support_prob_per_chunk.append([score])

        return pred_labels, probs, used_chunks, support_prob_per_chunk