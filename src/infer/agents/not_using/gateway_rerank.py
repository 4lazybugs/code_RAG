from dataclasses import dataclass
from typing import Any, Dict

from .base import BaseModel
from .reranker import Reranker

@dataclass
class RankGate_agent(BaseModel):
    reranker: Reranker
    RAG_agent: Any
    SOTA_agent: Any = None

    rerank_thre: float = 0.5

    def rerank_prob(self, question: str, context: str) -> float:
        """
        question-context relevance를 reranker 하나로 평가.
        reranker 구현체에 따라 반환 형식이 다를 수 있으므로
        float / list / tuple 정도는 유연하게 처리.
        """
        result = self.reranker.score(query=question, docs=[context])

        if isinstance(result, (int, float)):
            score = float(result)

        elif isinstance(result, list):
            score = float(result[0])

        elif isinstance(result, tuple):
            # 예: (scores, ...)
            first = result[0]
            if isinstance(first, list):
                score = float(first[0])
            else:
                score = float(first)

        else:
            raise TypeError(
                f"Unsupported reranker.score output type: {type(result)}"
            )

        print(f"================= Reranker score: {score:.4f} ====================")
        return score

    def answer_once(self, raw_input: Dict[str, Any] = None):
        # 1) 우선 RAG 수행
        output_dic = self.RAG_agent.answer_once(raw_input)

        question = output_dic["question"]

        contents = []
        for doc in output_dic["retrieved"]:
            content = doc.get("content", [])
            if isinstance(content, list):
                contents.extend(content)
            elif isinstance(content, str):
                contents.append(content)

        context = "\n\n".join(contents)

        # 1) reranker score 계산
        rerank_score = self.rerank_prob(question=question, context=context)
        output_dic["judge_score"][0]["rerank_score"] = rerank_score

        # 2) threshold 기준으로 routing
        if rerank_score >= self.rerank_thre:
            print("==== Reranker Good → QWEN+RAG 반환 =======")
            return output_dic

        print("==== Reranker Bad → SOTA fallback =======\n")
        if self.SOTA_agent is not None:
            return self.SOTA_agent.answer_once(output_dic)

        return output_dic