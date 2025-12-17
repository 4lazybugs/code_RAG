from .base import BaseExpert
from langchain_ollama.llms import OllamaLLM
# from langchain.chains import RetrievalQA  # 제거: 최신 langchain에서 없음 + 여기서 사용하지 않음
from typing import Dict, Any, List, Tuple
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------

# (프롬프트들은 원본 그대로 유지)
# rag_prompt, reform_prompt, text2sql_prompt, naive_llm_prompt ...
# suff_check_prompt, followup_prompt, final_answer_prompt ...

# RAG prompts for all/partial modes
rag_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문 상담사입니다. 아래 참고 자료를 바탕으로 질문에 답변하세요.

**답변 작성 규칙:**
1. 참고 자료의 내용을 **이해하고 재구성**하여 자연스럽고 명확하게 설명하세요
2. 답변은 **핵심 정보만 간결하게**, 불필요한 배경설명·문장 반복을 피하세요  
3. 문장은 **짧고 직관적**으로 작성하고, 장문·장황한 서술을 하지 마세요
4. 원문을 그대로 복사하거나 표/목록 형식을 그대로 옮기지 마세요
5. 문서 출처, 페이지 번호, 파일명 등은 언급하지 마세요

---
[참고 자료]
{reviews}

---
[질문]
{question}

[답변]
""")

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

    # ---- 추가: retriever 호출을 한 곳으로 통일 (최소 변경) ----
    def _retrieve_docs(self, query: str):
        """
        retriever 구현체마다 지원 메서드가 달라서 안전하게 통일.
        - 최신: retriever.invoke(query)
        - 구형: retriever.get_relevant_documents(query)
        - 일부: invoke({"query": ...}) 형태만 받는 경우도 있어 예외 처리
        """
        if hasattr(self.retriever, "invoke"):
            try:
                return self.retriever.invoke(query)
            except Exception:
                return self.retriever.invoke({"query": query})
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(query)
        raise TypeError(f"Retriever does not support invoke/get_relevant_documents: {type(self.retriever)}")

    def _format_prompt(self, template: str, **kwargs) -> str:
        try:
            return template.format(**kwargs)
        except Exception:
            result = template
            for key, value in kwargs.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result

    def _check_sufficiency(self, question: str, context: str) -> bool:
        prompt = self._format_prompt(suff_check_prompt, original_question=question, context=context)
        try:
            response = str(self.model.invoke(prompt)).strip().lower()
            return response.startswith("yes")
        except Exception as e:
            print(f"[ERROR] 충분성 검사 실패: {e}")
            return True

    def _generate_followup(self, question: str, context: str) -> str:
        prompt = self._format_prompt(followup_prompt, original_question=question, context=context)
        try:
            return str(self.model.invoke(prompt)).strip()
        except Exception as e:
            print(f"[ERROR] 후속 질문 생성 실패: {e}")
            return ""

    def _answer_with_rag(self, question: str) -> str:
        try:
            docs = self._retrieve_docs(question)

            print(f"[DEBUG] retrieved docs for '{question}':")
            for i, d in enumerate(docs or []):
                md = getattr(d, "metadata", {}) or {}
                src_store = md.get("__source_store__", "unknown_store")
                base_src  = md.get("source", "")
                score     = md.get("__score__", None)
                content   = getattr(d, "page_content", str(d))
                print(
                    f"- {i+1}. store={src_store}, score={score}, source={base_src} | "
                    f"{content[:120]}..."
                )

            if not docs:
                return str(self.model.invoke(question)).strip()

            review = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
            result = self.rag_chain.invoke({"reviews": review, "question": question})
            return str(result).strip() if result else ""

        except Exception as e:
            print(f"[ERROR] RAG 답변 실패: {e}")
            return f"답변 생성 오류: {str(e)}"

    def handle(self, question: str) -> str:
        print(f"[SelfAsk] 시작: {question}")

        # 기존 로직 보존하되, retriever 접근을 _retrieve_docs로 통일
        if self._check_sufficiency(question, ""):
            print("[SelfAsk] 충분도 검사 통과 - 직접 RAG")
            snippets = self._retrieve_docs(question)
            review = "\n\n".join(getattr(d, "page_content", str(d)) for d in (snippets or []))
            return str(self.rag_chain.invoke({"reviews": review, "question": question})).strip()

        snippets = self._retrieve_docs(question)
        review = "\n\n".join(getattr(d, "page_content", str(d)) for d in (snippets or []))

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
            return str(self.rag_chain.invoke({"reviews": review, "question": question})).strip()

        print(f"[SelfAsk] 최종 답변 생성 ({len(qas)}개 Q&A)")
        qa_history = "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(qas))
        final_prompt = self._format_prompt(final_answer_prompt, original_question=question, qa_history=qa_history)

        try:
            final_answer = str(self.model.invoke(final_prompt)).strip()
            print("[SelfAsk] 완료")
            return final_answer if final_answer else qas[-1][1]
        except Exception as e:
            print(f"[ERROR] 최종 답변 실패: {e}")
            return qas[-1][1] if qas else f"실패: {str(e)}"
