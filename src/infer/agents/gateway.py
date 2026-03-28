from dataclasses import dataclass, field

from typing import Any, Dict, Tuple, List
from .base import BaseModel
from src.infer.qa_type.base import QAtype

@dataclass
class Gateway_agent(BaseModel):

    Router_agent: Any
    LLM_agent: Any
    judge_lm: Any
    RAG_agent: Any
    SOTA_agent: Any = None

    know_thres: float = 0.8
    relv_thre: float = 0.5
    faith_thre: float = 0.5

    def relv_prob(self, question: str, context: str) -> float:

        _, prob, _, _ = self.judge_lm.score(docs=[context], claims=[question])
        '''
            반환값 4개는:(pred_label, prob, used_chunk, support_prob_per_chunk)
            ↑ 0 or 1    ↑ 확률값  ↑ 사용된 청크   ↑ 청크별 확률
        '''
        score = float(prob[0])
        print(f"================= Context Relevance score: {score:.4f} ====================")
        return score

    def faith_prob(self, answer: str, context: str) -> float:

        _, prob, _, _ = self.judge_lm.score(docs=[context], claims=[answer])
        '''
            반환값 4개는:(pred_label, prob, used_chunk, support_prob_per_chunk)
            ↑ 0 or 1    ↑ 확률값  ↑ 사용된 청크   ↑ 청크별 확률
        '''
        score = float(prob[0])
        print(f"================= Answer Faithfullness score: {score:.4f} ====================")
        return score

    def answer_once(self, raw_input: Dict[str, Any] = None):
        output_dic = self.Router_agent.answer_once(raw_input)

        know_prob = output_dic['judge_score'][0]['know_prob']
        print(f"==== Naive_LLM이 질문에 대해 알고 있을 확률은 {know_prob}이다. =======")

        # Know → LLM 직답
        if float(know_prob) > self.know_thres:
            output_dic = self.LLM_agent.answer_once(output_dic)

        # Don't Know → RAG 후 평가
        else:
            #breakpoint()
            output_dic = self.RAG_agent.answer_once(output_dic)
            
            question = output_dic["question"]
            gen_ans = output_dic["generated"]
            contents = []
            for doc in output_dic["retrieved"]:
                contents.extend(doc["content"]) # extend는 리스트 이어붙이는 메서드
            context = "\n\n".join(contents) 
            if not context.strip(): # "context가 비어있거나, 공백만 있는 경우"
                return 0.0
        
            relv_score = self.relv_prob(question, context)
            output_dic["judge_score"][0]["relv_score"] = relv_score
    
            # Context Relevance Bad → GPT fallback
            if relv_score <= self.relv_thre:
                print("==== Context Relevance Bad → GPT fallback =======\n")
                output_dic = self.SOTA_agent.answer_once(output_dic)

            else:
                # Context Relevance Good → Faithfulness 체크
                faith_score = self.faith_prob(gen_ans, context)

                # Faithfulness Good → QWEN+RAG 답변 그대로 반환
                if faith_score > self.faith_thre:
                    print("==== Faithfulness Good → QWEN+RAG 반환 =======")

                # Faithfulness Bad → GPT fallback
                else:
                    print("==== Faithfulness Bad → GPT fallback =======\n")
                    output_dic = self.SOTA_agent.answer_once(output_dic)
                
                output_dic["judge_score"][0]["faith_score"] = faith_score
        
        return output_dic
