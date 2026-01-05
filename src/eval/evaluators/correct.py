from .base import Evaluator, add_metric_key, get_config
from dotenv import load_dotenv
import os
import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


@add_metric_key("correct")
class CorrectnessEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다.")

        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "당신은 질의응답 시스템의 정답성(correctness)을 평가하는 엄격한 평가자입니다.\n"
                        "GENERATED ANSWER가 REFERENCE ANSWER에 비추어 올바른지 평가하십시오.\n"
                        "규칙:\n"
                        "1) 판단 기준은 REFERENCE ANSWER이며, 외부 지식은 사용하지 마십시오.\n"
                        "2) 표현이 달라도 의미적으로 동일하면 정답으로 간주할 수 있습니다.\n"
                        "3) 핵심 정보가 누락되거나 질문에 충분히 답하지 못하면 점수를 낮추십시오.\n"
                        "4) REFERENCE와 명백히 충돌하거나 잘못된 주장을 하면 점수는 0입니다.\n"
                        "5) 점수는 다음 중 하나만 선택하십시오: 1.0, 0.75, 0.5, 0.25, 0.0\n"
                        "   - 1.0: 의미적으로 완전히 동일하며 충분히 답함\n"
                        "   - 0.75: 대체로 맞지만 사소한 누락/모호함이 있음\n"
                        "   - 0.5: 부분적으로 맞지만 중요한 정보 누락 또는 일부 오류가 있음\n"
                        "   - 0.25: 조금만 맞고 대부분 틀리거나 질문에 거의 답하지 못함\n"
                        "   - 0.0: REFERENCE와 충돌하거나 전반적으로 오답\n"
                        "출력은 반드시 유효한 JSON 하나로만 반환하십시오.\n"
                        "JSON 키: score (float), verdict (string), rationale (string)\n"
                        "verdict는 다음 중 하나: correct, mostly_correct, partially_correct, weakly_correct, incorrect\n"
                    ),
                ),
                (
                    "user",
                    (
                        "QUESTION:\n{question}\n\n"
                        "REFERENCE ANSWER:\n{reference}\n\n"
                        "GENERATED ANSWER:\n{generated}\n\n"
                        "JSON만 반환하십시오."
                    ),
                ),
            ]
        )
        self.chain = self.prompt | self.llm

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if "{" in text and "}" in text:
            text = text[text.find("{") : text.rfind("}") + 1]
        try:
            return json.loads(text)
        except Exception:
            return {"score": 0.0, "verdict": "parse_error", "rationale": text[:500]}

    def _normalize_score(self, score: Any) -> float:
        allowed = [0.0, 0.25, 0.5, 0.75, 1.0]
        try:
            s = float(score)
        except Exception:
            return 0.0
        s = max(0.0, min(1.0, s))
        return min(allowed, key=lambda a: abs(a - s))

    def score_once(self, data: Dict[str, Any]) -> float:
        """
        data에서 꺼내 사용:
          - reference: GT 답변
          - generated: 모델 답변
          - question: (옵션) 있으면 사용, 없으면 ""
        """
        ref = self._normalize(data.get("reference"))
        gen = self._normalize(data.get("generated"))
        q = self._normalize(data.get("question"))  # 없으면 ""

        res = self.chain.invoke(
            {
                "question": q,
                "reference": ref,
                "generated": gen,
            }
        )

        content = getattr(res, "content", None)
        if content is None:
            content = str(res)

        obj = self._safe_parse_json(content)
        return self._normalize_score(obj.get("score", 0.0))
