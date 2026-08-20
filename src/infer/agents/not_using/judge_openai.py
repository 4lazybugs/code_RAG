import json
import re
from typing import List, Tuple, Literal

from openai import OpenAI


class Judge_OpenAI:
    """
    Judge_LM의 OpenAI API 버전.
    score() 인터페이스는 Judge_LM과 완전히 동일 — Gateway_agent 수정 불필요.
    label은 반환하지 않고 score만 반환, label 판단은 상위(Gateway_agent)에서 threshold로 수행.
    """

    def __init__(
        self,
        relv_prompt,
        faith_prompt,
        model_name: str,
        max_tokens: int = 128,
    ):
        self.relv_prompt = relv_prompt
        self.faith_prompt = faith_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = OpenAI()  # OPENAI_API_KEY 환경변수에서 자동 로딩

    def _select_prompt(self, prompt_type: Literal["relv", "faith"]) -> str:
        if prompt_type == "relv":
            return self.relv_prompt
        if prompt_type == "faith":
            return self.faith_prompt
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    def _extract_json(self, text: str) -> Tuple[int, float]:
        """
        Judge_LM과 동일한 파싱 로직.
        단, score만 신뢰하고 label은 score >= 0.5 기준으로 내부 생성
        (어차피 상위에서 threshold로 재판단하므로 내부 label은 보조용).
        """
        text = text.strip()

        # 1순위: JSON 파싱
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                score = max(0.0, min(1.0, float(obj.get("score", 0.0))))
                label = int(obj.get("label", 1 if score >= 0.5 else 0))
                return (1 if label == 1 else 0), score
            except Exception:
                pass

        # 2순위: fallback — "score": 0.xx 패턴만 찾기 (Judge_LM보다 안전한 버전)
        score_match = re.search(r'"score"\s*:\s*([01](?:\.\d+)?)', text)
        if score_match:
            score = max(0.0, min(1.0, float(score_match.group(1))))
            return (1 if score >= 0.5 else 0), score

        # 3순위: 완전 fallback — 0 처리 (Judge_LM의 num_match보다 보수적)
        return 0, 0.0

    def _score_one(
        self,
        doc: str,
        claim: str,
        prompt_type: Literal["relv", "faith"],
    ) -> Tuple[int, float]:
        prompt = self._select_prompt(prompt_type).format(doc=doc, claim=claim)

        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0,                   # judge는 항상 deterministic
            response_format={"type": "json_object"},  # GPT-4o JSON mode: 파싱 안정성 ↑
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a judge. "
                        "Return only valid JSON: "
                        '{"score": <float 0.0~1.0>}. '
                        "Do not output any other text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        gen_text = response.choices[0].message.content or ""
        return self._extract_json(gen_text)

    def score(
        self,
        docs: List[str],
        claims: List[str],
        prompt_type: Literal["relv", "faith"],
    ) -> Tuple[List[int], List[float], List[str], List[List[float]]]:
        """Judge_LM.score()와 완전히 동일한 시그니처 및 반환 구조."""
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