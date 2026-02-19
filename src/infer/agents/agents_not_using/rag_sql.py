from .base import BaseExpert
from langchain_ollama.llms import OllamaLLM
# from langchain.chains import RetrievalQA  # 제거 (더 이상 사용 불가 + 여기선 안 씀)
import re
import sqlite3
import pandas as pd
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


class SqlExpert(BaseExpert):
    """Expert that converts text to SQL and executes queries"""

    def __init__(self, retriever_map: Dict[str, Any], retriever_mode: str,
                 qna_sql_path: str, crop_sql_path: str):
        self.qna_sql_path = qna_sql_path
        self.crop_sql_path = crop_sql_path
        super().__init__(retriever_map, retriever_mode)

    def setup(self, retriever_mode: str) -> None:
        # LLMs
        text2sql_llm = OllamaLLM(model="sqlcoder:15b", temperature=0.2, top_p=0.9, top_k=40)
        reform_llm = OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)
        ans_llm = OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)

        # Chains (기존 구조 유지: prompt | llm)
        self.reform = reform_prompt | reform_llm | StrOutputParser()
        self.text2sql = text2sql_prompt | text2sql_llm | StrOutputParser()
        self.ans_chain = naive_llm_prompt | ans_llm | StrOutputParser()

    def _strip_code_fences(self, s: str) -> str:
        # ```sql ... ``` 또는 ``` ... ``` 제거
        s = re.sub(r"```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```", "", s)
        return s.strip()

    def _is_safe_sql(self, sql: str) -> bool:
        sql_s = sql.strip()
        # 규칙 기반 가드 (원하신 “최대한 보존” 원칙 하에 최소한만)
        if not re.match(r"(?is)^select\b", sql_s):
            return False
        if not re.match(r"(?is)^select\s+label\s+from\s+crop_recommendation\b", sql_s):
            return False
        if ";" in sql_s:
            return False
        if re.search(r"(?is)\b(distinct|join|with|insert|update|delete|drop|alter|pragma)\b", sql_s):
            return False
        return True

    def handle(self, question: str) -> str:
        try:
            instr = self.reform.invoke({"question": question})
            instr = str(instr).strip()

            raw_sql = self.text2sql.invoke({"instruction": instr})
            raw_sql = str(raw_sql)
            sql = self._strip_code_fences(raw_sql)

            if sql == "I don't know how to answer that in SQL." or not self._is_safe_sql(sql):
                return "I don't know how to answer that in SQL."

            conn = sqlite3.connect(self.crop_sql_path)
            try:
                df = pd.read_sql_query(sql, conn)
            finally:
                conn.close()

            csv = df.to_csv(index=False)
            out = self.ans_chain.invoke({"question": question, "query": sql, "csv": csv})
            return str(out).strip()

        except Exception as e:
            return f"[SQL 오류]: {str(e)}"
