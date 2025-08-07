from langchain_core.prompts import ChatPromptTemplate

# —————— all/partial RAG용 LLM 분기 ——————
# 1) 사용자 질문을 리포뮬레이션
rag_prompt = ChatPromptTemplate.from_template("""
당신은 농업 Q&A 전문가입니다.
아래 제공된 스니펫만을 사용하여 답변하세요.  
제공된 정보에 답이 없으면 다음과 같이 답변하세요:
"주어진 정보 가지고는 판단하기가 어렵습니다."

스니펫:
{reviews}

질문:
{question}

답변 (한국어):
""")


# —————— Text-2-SQL용 LLM 분기 ——————
# 1) 사용자 질문을 리포뮬레이션
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

# 2) 리포뮬레이션된 지시를 SQL로 변환


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
3. Numeric “around X” handling:  
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
"""당신은 농업 데이터 해석 전문가입니다.
사용자 질문: {question}

실행한 쿼리:
{query}

결과 (CSV):
{csv}

간결하게 한국어로 답변하세요."""
)


adaptive_inst = "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."


suff_check_prompt = (
    "원래 질문:\n{original_question}\n\n"
    "지금까지 수집된 Q&A:\n{context}\n\n"
    "정확하고 완전한 답변에 필요한 모든 정보가 충분히 모였는지 판단하세요. "
    "조금이라도 부족하면 NO라고 답하세요. "
    "반드시 YES 또는 NO만 답변하세요:"
)

followup_prompt = (
    "당신은 질의 재구성에 특화된 어시스턴트입니다. "
    "원래 질문과 현재까지의 맥락을 보고, 새로운 질문을 만드는 것이 아니라 "
    "정확한 답변을 위해 어떤 정보가 부족하거나 불명확한지 분석하고, "
    "이를 각각의 '정보 요구' 또는 하위 질문으로 나눠서 번호로 나열하세요.\n\n"
    "원래 질문:\n"
    "{original_question}\n\n"
    "수집된 맥락:\n"
    "{context}\n\n"
    "원래 질문에 답하기 위해 추가로 필요한 정보나 명확히 해야 할 점을 번호로 나열하세요. 예시:\n"
    "1. …\n"
    "2. …\n"
    "3. …\n"
    "– 직접적인 후속 질문은 생성하지 마세요."
)


final_answer_prompt = (
    "원래 질문:\n{original_question}\n\n"
    "단계별 Q&A 히스토리:\n{qa_history}\n\n"
    "이 히스토리를 바탕으로 간결하게 최종 답변을 한국어로 작성하세요:"
)