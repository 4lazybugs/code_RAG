from abc import ABC, abstractmethod
import os
import sqlite3
import pandas as pd
import re
import io
import contextlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import torch, gc

from src.prompt import (
    rag_prompt,
    reform_prompt,
    text2sql_prompt,
    naive_llm_prompt,
    suff_check_prompt,
    followup_prompt,
    final_answer_prompt
)
from src.store_db import load_stores, load_all_docs
from src.utils import load_special_tokens, get_config, control_tokens, PROMPT_DICT

from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"


class BaseExpert(ABC):
    """
    추상 베이스 클래스:
      - setup(): 리트리버·LLM·체인 초기화
      - handle(): 질문에 대해 응답 생성
    """
    def __init__(self, retriever_map: Dict[str, Any], retriever_mode: str):
        self.args = get_config()
        self.retriever_map = retriever_map
        self.setup(retriever_mode)
        
    @abstractmethod
    def setup(self, retriever_mode: str) -> None:
        """Initialize retriever, LLM, and chains"""
        pass

    @abstractmethod
    def handle(self, question: str) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Handle a question and return response"""
        pass

# --------------------------------------------------

class AllExpert(BaseExpert):
    """Expert that uses all documents for RAG (accuracy-focused)"""
    
    def setup(self, retriever_mode: str) -> None:
        model = OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)
        if retriever_mode not in self.retriever_map:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=self.retriever_map[retriever_mode],
            return_source_documents=False
        )

    def handle(self, question: str) -> str:
        try:
            docs = load_all_docs()
            res = self.qa_chain.invoke({
                "input_documents": docs,
                "query": question
            })
            return res["result"].strip()
        except Exception as e:
            return f"[오류]: {str(e)}"

# --------------------------------------------------

class PartialExpert(BaseExpert):
    """Expert that uses top-K snippets for RAG (speed/partial answers)"""
    
    def setup(self, retriever_mode: str) -> None:
        self.model = OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)
        if retriever_mode not in self.retriever_map:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")
        self.retriever = self.retriever_map[retriever_mode]
        self.rag_chain = rag_prompt | self.model

    def handle(self, question: str) -> str:
        try:
            snippets = self.retriever.get_relevant_documents(question)
            review = "\n\n".join(d.page_content for d in snippets)
            result = self.rag_chain.invoke({"reviews": review, "question": question})
            return result
        except Exception as e:
            return f"[오류]: {str(e)}"

# --------------------------------------------------

class SqlExpert(BaseExpert):
    """Expert that converts text to SQL and executes queries"""
    
    def __init__(self, retriever_map: Dict[str, Any], retriever_mode: str, 
                 qna_sql_path: str, crop_sql_path: str):
        self.qna_sql_path = qna_sql_path
        self.crop_sql_path = crop_sql_path
        super().__init__(retriever_map, retriever_mode)

    def setup(self, retriever_mode: str) -> None:
        self.text2sql = OllamaLLM(model="sqlcoder:15b", temperature=0.2, top_p=0.9, top_k=40)
        self.reform = reform_prompt | OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)
        self.ans_chain = naive_llm_prompt | OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)

    def handle(self, question: str) -> str:
        try:
            instr = self.reform.invoke({"question": question})
            raw_sql = self.text2sql.invoke({"instruction": instr})
            sql = re.sub(r"```(?:sql)?```", "", raw_sql).strip()

            conn = sqlite3.connect(self.crop_sql_path)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            
            csv = df.to_csv(index=False)
            return self.ans_chain.invoke({"question": question, "query": sql, "csv": csv})
        except Exception as e:
            return f"[SQL 오류]: {str(e)}"

# --------------------------------------------------

class RawLlmExpert(BaseExpert):
    """Expert that uses raw LLM without retrieval"""
    
    def setup(self, retriever_mode: str) -> None:
        self.model = OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)

    def handle(self, question: str) -> str:
        try:
            return self.model.invoke(question).strip()
        except Exception as e:
            return f"[오류]: {str(e)}"

# --------------------------------------------------
#### 참고: https://github.com/AkariAsai/self-rag ####

class AdaptiveExpert(BaseExpert):
    """완전 vLLM 기반 Adaptive gating + RAG 전문가"""
    model_dir = os.getenv("MODEL_DIR", "./selfrag_llama2_7b")

    def setup(self, retriever_mode: str) -> None:
        self.model_dir = os.getenv("MODEL_DIR", "./selfrag_llama2_7b")
        self.model_dir = os.path.expanduser(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True, local_files_only=True, repo_type="model"
        )
        self.critic_llm = LLM(
            model=self.model_dir,
            model_impl="transformers",
            trust_remote_code=True,
            dtype="half",
            gpu_memory_utilization=0.9,
            max_num_seqs=1,
            max_model_len=1824
        )
        self.rag_tpl = rag_prompt
        self.args = get_config()
        self.gate_sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=25, logprobs=20)
        self.ans_sampling  = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256, logprobs=None)

        if retriever_mode not in self.retriever_map:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")
        self.retriever = self.retriever_map[retriever_mode]

        (self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens) = load_special_tokens(
            self.tokenizer, use_grounding=True, use_utility=True
        )
        if self.ret_tokens:
            self.special_token_map = {
                "ret_tokens": self.ret_tokens,
                "rel_tokens": self.rel_tokens,
                "grd_tokens": self.grd_tokens,
                "ut_tokens": self.ut_tokens,
            }

    def _to_float(self, lp, default: float = -5.0) -> float:
        val = getattr(lp, "logprob", lp)
        try:
            return float(val)
        except:
            return default

    def _call_adaptive(self, prompt: str, evidences: list):
        a = self.args
        out = self.critic_llm.generate([prompt], self.gate_sampling, use_tqdm=False)[0]
        lp0    = out.outputs[0].logprobs[0]
        ret_lp = self._to_float(lp0.get(self.ret_tokens["[Retrieval]"]))
        no_lp  = self._to_float(lp0.get(self.ret_tokens["[No Retrieval]"]))
        p_ret  = float(np.exp(ret_lp))
        p_no   = float(np.exp(no_lp))
        gate_p = p_ret / (p_ret + p_no + 1e-12)

        if gate_p <= a.threshold:
            return prompt, {"gating": {"p_ret": p_ret, "p_no": p_no, "used": False}}, False

        if evidences:
            best = evidences[0]
            final_prompt = (
                prompt + f"[Retrieval]<paragraph>{best['title']}\n{best['text']}</paragraph>\n\nAnswer:"
            )
            return final_prompt, {"gating": {"used": True}}, True

        return prompt, {"gating": {"used": False, "no_evidence": True}}, False

    def handle(self, question: str):
        prompt0 = PROMPT_DICT["prompt_no_input"].format_map({"instruction": question})
        docs    = self.retriever.get_relevant_documents(question)
        ev      = [{"title": d.metadata.get("source",""), "text": d.page_content} for d in docs]

        final_prompt, info, used = self._call_adaptive(prompt0, ev)
        if used:
            prompt_text = self.rag_tpl.format(reviews=ev[0]["text"], question=question)
        else:
            prompt_text = prompt0 + "\n\nAnswer:"

        outputs = self.critic_llm.generate([prompt_text], self.ans_sampling)
        answer = outputs[0].outputs[0].text.strip()
        return answer, info

# --------------------------------------------------

class SelfAskExpert(BaseExpert):
    """Expert using Self-Ask: sufficiency check → follow-up questions → CoT final answer"""

    def __init__(self, retriever_map: Dict[str, Any], retriever_mode: str, max_iter: int = 3):
        self.max_iter = max_iter
        super().__init__(retriever_map, retriever_mode)

    def setup(self, retriever_mode: str) -> None:
        self.model = OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)
        if retriever_mode not in self.retriever_map:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")
        self.retriever = self.retriever_map[retriever_mode]
        self.rag_chain = rag_prompt | self.model

    def _format_prompt(self, template: str, **kwargs) -> str:
        try:
            return template.format(**kwargs)
        except:
            result = template
            for key, value in kwargs.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result

    def _check_sufficiency(self, question: str, context: str) -> bool:
        prompt = self._format_prompt(suff_check_prompt, original_question=question, context=context)
        try:
            response = self.model.invoke(prompt).strip().lower()
            return response.startswith("yes")
        except Exception as e:
            print(f"[ERROR] 충분성 검사 실패: {e}")
            return True

    def _generate_followup(self, question: str, context: str) -> str:
        prompt = self._format_prompt(followup_prompt, original_question=question, context=context)
        try:
            return self.model.invoke(prompt).strip()
        except Exception as e:
            print(f"[ERROR] 후속 질문 생성 실패: {e}")
            return ""

    def _answer_with_rag(self, question: str) -> str:
        try:
            docs = (self.retriever.invoke(question)
                    if hasattr(self.retriever, 'invoke')
                    else self.retriever.get_relevant_documents(question))
            print(f"[DEBUG] retrieved docs for '{question}':")
            for i, d in enumerate(docs):
                print(f"- {i+1}. {d.metadata.get('source', '')} | {d.page_content[:100]}...")
            if not docs:
                return self.model.invoke(question).strip()
            review = "\n\n".join(d.page_content for d in docs)
            result = self.rag_chain.invoke({"reviews": review, "question": question})
            return result.strip() if result else ""
        except Exception as e:
            print(f"[ERROR] RAG 답변 실패: {e}")
            return f"답변 생성 오류: {str(e)}"

    def handle(self, question: str) -> str:
        print(f"[SelfAsk] 시작: {question}")
        if self._check_sufficiency(question, ""):
            print("[SelfAsk] 충분도 검사 통과 - 직접 RAG")
            snippets = self.retriever.get_relevant_documents(question)
            review = "\n\n".join(d.page_content for d in snippets)
            return self.rag_chain.invoke({"reviews": review, "question": question}).strip()

        snippets = self.retriever.get_relevant_documents(question)
        review = "\n\n".join(d.page_content for d in snippets)

        qas: List[Tuple[str, str]] = []
        context: List[str] = []

        for iteration in range(self.max_iter):
            print(f"[SelfAsk] 반복 {iteration + 1}/{self.max_iter}")
            ctx = "\n".join(context)
            if self._check_sufficiency(question, ctx):
                print("[SelfAsk] 충분한 정보 확보")
                break
            followup = self._generate_followup(question, ctx)
            if not followup:
                print("[SelfAsk] 후속 질문 생성 실패")
                break
            print(f"[SelfAsk] 후속 질문: {followup}")
            answer = self._answer_with_rag(followup)
            if not answer:
                print("[SelfAsk] 답변 생성 실패")
                continue
            print(f"[SelfAsk] 답변: {answer[:50]}...")
            qas.append((followup, answer))
            context.append(f"Q: {followup}\nA: {answer}")

        if not qas:
            print("[SelfAsk] Q&A 없음 - Partial fallback")
            return self.rag_chain.invoke({"reviews": review, "question": question}).strip()

        print(f"[SelfAsk] 최종 답변 생성 ({len(qas)}개 Q&A)")
        qa_history = "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(qas))
        final_prompt = self._format_prompt(final_answer_prompt, original_question=question, qa_history=qa_history)
        try:
            final_answer = self.model.invoke(final_prompt).strip()
            print("[SelfAsk] 완료")
            return final_answer if final_answer else qas[-1][1]
        except Exception as e:
            print(f"[ERROR] 최종 답변 실패: {e}")
            return qas[-1][1] if qas else f"실패: {str(e)}"

# --------------------------------------------------

def create_expert_instances(retriever_map: Dict[str, Any], retriever_mode: str, 
                           qna_sql_path: str, crop_sql_path: str) -> Dict[str, BaseExpert]:
    """Factory function to create expert instances"""
    experts = {}
    experts["all"] = AllExpert(retriever_map, retriever_mode)
    experts["partial"] = PartialExpert(retriever_map, retriever_mode)
    experts["sql"] = SqlExpert(retriever_map, retriever_mode, qna_sql_path, crop_sql_path)
    #experts["adaptive"] = AdaptiveExpert(retriever_map, retriever_mode)
    experts["raw_llm"] = RawLlmExpert(retriever_map, retriever_mode)
    experts["self_ask"] = SelfAskExpert(retriever_map, retriever_mode)

    # (옵션) self_ask_5 같은 변형 처리하려면 여기 확장
    return experts


def get_help_text() -> str:
    return """
[모드별 사용 가이드]
mode       | 설명
-----------------------------------------------
all        | 전체 DB를 대상으로 RAG (정확도 우선)
partial    | 상위 K개 snippet만 RAG (속도/부분답변)
sql        | text→SQL→실행→해석 (테이블 기반 추천)
raw_llm    | LLM 자유 응답 (retrieval 없이)
self_ask   | Self-Ask 방식: 충분도 검사→추가질문→답변 집계
(종료: q)
""".strip()

# adaptive   | Adaptive gating + RAG (Self-RAG 방식)


def main():
    """Main interactive loop"""
    try:
        ndocs = 15

        # dict 기반 stores 사용
        stores = load_stores(ndocs=ndocs)
        retriever_map = {k: stores[k] for k in ['qna','crop','soil','bugs','farm'] if k in stores}
        retriever_mode = "qna"  # 필요시 'soil', 'bugs', 'farm' 등으로 변경

        # SQL 경로
        qna_sql_path  = stores['qna_sql']
        crop_sql_path = stores['crop_sql']

        expert_instances = create_expert_instances(retriever_map, retriever_mode, qna_sql_path, crop_sql_path)
        help_text = get_help_text()

        while True:
            print(help_text)
            mode = input("Mode → all/partial/sql/raw_llm/self_ask (q to quit): ").strip().lower()
            if mode == "q":
                break
            if mode not in expert_instances:
                print("Invalid mode. 다시 선택하세요.")
                continue

            question = input("Question → ").strip()
            if question.lower() == "q":
                break

            expert = expert_instances[mode]
            try:
                res = expert.handle(question)
                answer = res[0] if isinstance(res, tuple) else res
                print(f"\n[Answer]\n{answer}\n")
            except Exception as e:
                print(f"\n[오류 발생]: {str(e)}\n")

    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n[시스템 오류]: {str(e)}")


if __name__ == "__main__":
    main()
