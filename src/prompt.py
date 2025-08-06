from langchain_core.prompts import ChatPromptTemplate

# —————— all/partial RAG용 LLM 분기 ——————
# 1) 사용자 질문을 리포뮬레이션
rag_prompt = ChatPromptTemplate.from_template("""
You are an expert in agricultural Q&A.
Use ONLY the snippets below.  
If the answer is not in the provided information, reply:
"I don't know based on the provided information."

Snippets:
{reviews}

Question:
{question}

Answer (Korean):
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
"""You are an expert in agricultural data interpretation.
User asked: {question}

You ran:
{query}

Results (CSV):
{csv}

Answer concisely in Korean."""
)


adaptive_inst = "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."


suff_check_prompt = (
    "Original Question:\n{original_question}\n\n"
    "Collected Q&A so far:\n{context}\n\n"
    "You must decide whether you have absolutely all information needed "
    "to answer fully and accurately. If anything is missing, answer NO. "
    "Only respond YES or NO:"
)

followup_prompt = (
    "You are an assistant specialized in query reformulation. "
    "Given an original question and any collected context, your task is NOT to ask a new question, "
    "but to analyze what specific information is missing or unclear and break the original question "
    "into smaller “information needs” or sub‑questions that, if answered, would enable a complete answer.\n\n"
    "Original Question:\n"
    "{original_question}\n\n"
    "Collected Context:\n"
    "{context}\n\n"
    "Identify and list each missing piece of information or clarification needed to answer the original question. "
    "Output as numbered items, e.g.:\n"
    "1. …\n"
    "2. …\n"
    "3. …\n"
    "– and do NOT generate a direct follow-up question."
)


final_answer_prompt = (
                "Original Question:\n{original_question}\n\n"
                "Step-by-step Q&A History:\n{qa_history}\n\n"
                "Based on this history, provide a concise final answer in Korean:"
            )