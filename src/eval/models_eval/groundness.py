from .base import BaseEvaluator
from dotenv import load_dotenv
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class GroundEvaluator(BaseEvaluator):
    metric_key = "ground"  # groundedness/faithfulness 점수

    def __init__(self, k=5, max_context_chars=12000, max_doc_chars=4000):
        super().__init__()

        self.k = k
        self.max_context_chars = max_context_chars
        self.max_doc_chars = max_doc_chars

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다.")

        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "당신은 RAG 답변을 엄격하게 평가하는 평가자다.\n"
                            "제공된 CONTEXT만을 근거로, ANSWER가 CONTEXT에 의해 완전히 뒷받침되는지 판단하라.\n"
                            "규칙:\n"
                            "1) CONTEXT만 사용하라. 외부 지식을 사용하지 마라.\n"
                            "2) ANSWER에 CONTEXT로 뒷받침되지 않는 주장(클레임)이 하나라도 포함되면 점수를 낮춰라.\n"
                            "3) ANSWER가 CONTEXT와 모순되면 점수는 반드시 0이어야 한다.\n"
                            "4) 출력은 반드시 다음 키를 갖는 유효한 JSON이어야 한다: score (0..1), verdict, rationale.\n"
                            "5) 점수는 다음 5단계 척도로 부여하라:\n"
                            "   - 1.0: ANSWER의 모든 주장이 CONTEXT에 의해 명시적으로 그리고 완전하게 뒷받침된다.\n"
                            "   - 0.75: 대부분의 주장이 CONTEXT로 뒷받침되지만, 일부는 근거가 약하거나(모호) 경미한 누락이 있다.\n"
                            "   - 0.5: ANSWER는 부분적으로만 CONTEXT로 뒷받침되며, 여러 개의 주장에 근거가 없다.\n"
                            "   - 0.25: ANSWER의 일부만 CONTEXT로 뒷받침되고, 대부분의 주장은 근거가 부족하다.\n"
                            "   - 0.0: ANSWER가 거의 뒷받침되지 않거나, CONTEXT와 직접적으로 모순된다.\n"
                    ),
                ),
                (
                    "user",
                    (
                        "CONTEXT:\n{context}\n\n"
                        "ANSWER:\n{answer}\n\n"
                        "Return JSON only."
                    ),
                ),
            ]
        )
        self.chain = self.qa_prompt | self.llm

    @staticmethod
    def _canon(s: str) -> str:
        return (s or "").strip()

    def _build_context(self, gen_docs):
        """
        gen_docs: list[dict]  (예: [{"rank":1,"filename":"...","content":"..."}, ...])
        - rank 기준 정렬 후 top-k만 사용
        - content가 있으면 content 우선, 없으면 filename만이라도 남김
        - 길이 제한 적용
        """
        retrieved_sorted = sorted(gen_docs or [], key=lambda x: x.get("rank", 10**9))
        topk = retrieved_sorted[: self.k]

        parts = []
        total = 0
        for item in topk:
            fn = self._canon(item.get("filename", ""))
            content = item.get("content", None)

            if content is None:
                # content가 없으면 filename이라도 컨텍스트에 남겨서 디버깅 가능하게 함
                block = f"[DOC: {fn}]\n(NO_CONTENT)\n"
            else:
                content = self._canon(str(content))[: self.max_doc_chars]
                block = f"[DOC: {fn}]\n{content}\n"

            if total + len(block) > self.max_context_chars:
                break

            parts.append(block)
            total += len(block)

        return "\n---\n".join(parts).strip()

    def _safe_parse_json(self, text: str):
        text = (text or "").strip()
        if "{" in text and "}" in text:
            text = text[text.find("{"): text.rfind("}") + 1]
        try:
            return json.loads(text)
        except Exception:
            return {"score": 0.0, "verdict": "parse_error", "rationale": text[:500]}

    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs) -> list:
        """
        groundedness/faithfulness:
        - 기준: gen_docs(=모델이 본 retrieved context) 컨텐츠
        - references/ref_docs는 여기서는 사용하지 않음 (정답성 평가가 아님)

        반환: [score]  (0~1)
        """
        # generated는 메인에서 [gen_ans[i]] 형태로 들어오므로 첫 원소 사용
        answer = generated[0] if isinstance(generated, list) and generated else str(generated)

        context = self._build_context(gen_docs)

        # 컨텍스트가 비면 groundedness 평가 불가 -> 0 처리
        if not context:
            return [0.0]

        res = self.chain.invoke({"context": context, "answer": answer})
        content = getattr(res, "content", None)
        if content is None:
            content = str(res)

        obj = self._safe_parse_json(content)
        score = obj.get("score", 0.0)

        try:
            score = float(score)
        except Exception:
            score = 0.0

        score = max(0.0, min(1.0, score))
        return [score]
