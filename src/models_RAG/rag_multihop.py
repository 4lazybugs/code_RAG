from .base import BaseExpert
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from typing import Dict, Any
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
# --------------------------------------------------

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


# Text-to-SQL LLM branch
reform_prompt = ChatPromptTemplate.from_template("""
You are an expert at clarifying user requests for SQL queries on an agricultural database.
The user may use shorthand like:
  - N70 → nitrogen level ≈ 70
  - P60 → phosphorus level ≈ 60
  - K38 → potassium level ≈ 38
  - temp25 → temperature ≈ 25°C

Rewrite the user's question to explicitly state the filters needed, for example:
  "Find crops with nitrogen around 70, phosphorus around 60, potassium around 38, temperature around 25°C."

User Question:
{question}

Rewritten Instruction:
""")

text2sql_prompt = ChatPromptTemplate.from_template("""
You are an expert SQL generator for an agricultural SQLite database.

***IMPORTANT:***  
- **Your ONLY output** must be a **single** SQL query, nothing else.  
- If your output does **not** match the pattern `(?i)^SELECT\\b[\\s\\S]*`, or contains any non-SQL text (Python, pseudocode, comments, markdown, backticks, explanations, etc.), respond exactly:
  `I don't know how to answer that in SQL.`

DATABASE SCHEMA:
TABLE crop_recommendation(
  N           REAL,   -- nitrogen level
  P           REAL,   -- phosphorus level
  K           REAL,   -- potassium level
  temperature REAL,   -- °C
  humidity    REAL,   -- %
  ph          REAL,   -- soil acidity
  rainfall    REAL,   -- mm
  label       TEXT    -- crop name (exact values as in the DB)
)

LABEL FILTERING:
- If user mentions a crop name, you **must** filter with `WHERE label = '<exact label>'`.  
- Do **not** invent, alter or pluralize labels. If unsure, output:
  `I don't know how to answer that in SQL.`

FATAL RULES:
1. Output **exactly one** SQL statement—**nothing else**.  
2. Must start with `SELECT label FROM crop_recommendation`.  
3. Numeric "around X" handling:  
   - X ≥ 1 → `BETWEEN round(X*0.9,1) AND round(X*1.1,1)`  
   - X < 1 → `BETWEEN round(X-0.05,2) AND round(X+0.05,2)`  
4. Use only listed columns; no `DISTINCT`, `JOIN`, `WITH`, no trailing semicolon.  
5. If you cannot express the request in a single `SELECT + WHERE + optional LIMIT/ORDER BY`, output:
   `I don't know how to answer that in SQL.`

FEW-SHOT EXAMPLES:
Instruction: Find crops with nitrogen around 0.3 and phosphorus around 0.2  
SELECT label FROM crop_recommendation WHERE N BETWEEN 0.25 AND 0.35 AND P BETWEEN 0.15 AND 0.25

Instruction: Top 3 crops rainfall > 250 mm and ph < 5  
SELECT label FROM crop_recommendation WHERE rainfall > 250 AND ph < 5 LIMIT 3

Instruction: Which crops grow well at temp25?  
SELECT label FROM crop_recommendation WHERE temperature BETWEEN 22.5 AND 27.5

USER INSTRUCTION:
{instruction}

SQL:
""")

naive_llm_prompt = ChatPromptTemplate.from_template(
"""You are an agricultural data interpretation expert.
User Question: {question}

Executed Query:
{query}

Results (CSV):
{csv}

Please provide a concise answer."""
)

suff_check_prompt = (
    "Original Question:\n{original_question}\n\n"
    "Collected Q&A so far:\n{context}\n\n"
    "Determine if we have gathered sufficient information for an accurate and complete answer. "
    "If there's any missing information, respond with NO. "
    "Answer only with YES or NO:"
)

followup_prompt = (
    "You are an assistant specialized in query decomposition. "
    "Review the original question and current context to analyze what information "
    "is missing or unclear. Instead of creating new questions, break down the "
    "information needs into numbered sub-points.\n\n"
    "Original Question:\n"
    "{original_question}\n\n"
    "Collected Context:\n"
    "{context}\n\n"
    "List the additional information needed or points to clarify using numbers. Example:\n"
    "1. ...\n"
    "2. ...\n"
    "3. ...\n"
    "- Do not generate direct follow-up questions."
)

final_answer_prompt = (
    "Original Question:\n{original_question}\n\n"
    "Step-by-step Q&A History:\n{qa_history}\n\n"
    "Based on this history, please provide a concise final answer:"
)


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
                src_store = d.metadata.get("__source_store__", "unknown_store")
                base_src  = d.metadata.get("source", "")
                score     = d.metadata.get("__score__", None)
                print(
                    f"- {i+1}. store={src_store}, score={score}, source={base_src} | "
                    f"{d.page_content[:120]}..."
                )
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