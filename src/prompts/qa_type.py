from langchain_core.prompts import ChatPromptTemplate

hotpot_short_prompt = ChatPromptTemplate.from_template(
"""
당신은 multi-hop 질의응답 시스템이다.

입력:
- 문단 A
- 문단 B
- 질문

규칙:
1. 반드시 문단 A와 문단 B를 BOTH 모두 읽어야 한다.
2. 두 문단을 함께 사용해야만 답을 도출할 수 있어야 한다.
3. 답은 반드시 context 안에 명시된 "단일 값"이어야 한다.
4. 외부 지식 사용 금지.

추가 작업:
- 문단 A에서 답에 필요한 핵심 근거 문장을 원문 그대로 길게 발췌하라.
- 문단 B에서도 동일하게 원문 그대로 길게 발췌하라.
- supporting_fact는 각각 독립적으로 이해 가능할 만큼 충분히 길게 선택하라.

출력 형식:
설명 없이 JSON 배열만 출력

[
  {{
    "generated": "...",
    "supporting_fact_a": "...",
    "supporting_fact_b": "..."
  }}
]

--------------------

문단 A:
{md_a}

문단 B:
{md_b}

질문:
{question}
"""
)

logprob_prompt_eng = ChatPromptTemplate.from_template(
"""
You are a tool that determines whether you know the answer to a given question or not.
Based on the question, output a single number between 0 and 1 representing the probability that you know the answer.

[Rules]
- You must output only a single decimal number between 0 and 1.
- A value closer to 1 means you are very confident you know the answer.
- A value closer to 0 means you do not know the answer at all.
- Do not output any explanation, reasoning, or additional text.

[Examples]
Question: What is the capital of South Korea?
Answer: 0.99

Question: Who won the Nobel Prize in Physics in 2024?
Answer: 0.45

Question: Who was the 22nd king of the Joseon dynasty?
Answer: 0.85

Question: What is the optimal reproductive temperature of strawberry root rot nematodes?
Answer: 0.10

Question: What function is used to sort a list in Python?
Answer: 0.98

[Actual Question]
Question: {question}
Answer:
"""
)


logprob_prompt = ChatPromptTemplate.from_template(
"""
당신은 주어진 질문에 대해 자신이 알고 있는지 모르는지를 판단하는 도구입니다.
질문을 보고, 해당 내용을 알고 있을 확률을 0과 1 사이의 숫자로만 답하세요.

[규칙]
- 답변은 반드시 0과 1 사이의 소수 하나만 출력합니다.
- 1에 가까울수록 확실히 알고 있음, 0에 가까울수록 전혀 모름을 의미합니다.
- 설명, 이유, 부가 텍스트는 절대 출력하지 마세요.

[예시]
질문: 대한민국의 수도는 어디인가요?
답변: 0.99

질문: 2024년 노벨 물리학상 수상자는 누구인가요?
답변: 0.45

질문: 조선시대 22대 왕은 누구인가요?
답변: 0.85

질문: 딸기뿌리썩이선충의 생식 최적온도는 몇 도인가요?
답변: 0.10

질문: 파이썬에서 리스트를 정렬하는 함수는 무엇인가요?
답변: 0.98

[실제 질문]
질문: {question}
답변:
"""
)

mcq_llm_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 질문에 답하라.

중요: 출력 형식을 반드시 지켜라.

[출력 형식 규칙]
- 정답 번호 하나만 출력한다.
  예: 1
- 선택지가 없는 문제이면: 짧은 단답(한 문장 이내)만 출력한다.
- 어떤 경우에도 다음을 출력하지 마라: 선택지 내용, 괄호, "정답", "정답 번호", 콜론, 줄바꿈, 추가 설명.

[판단 규칙]
- 선택지는 {options}들 중에만 있다.

---
[질문]
{question}

[선택지]
{options}
""")

mcq_rag_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 자료를 근거로 질문에 답하라.

중요: 출력 형식을 반드시 지켜라.

[출력 형식 규칙]
- 정답 번호 하나만 출력한다.
  예: 1
- 선택지가 없는 문제이면: 짧은 단답(한 문장 이내)만 출력한다.
- 어떤 경우에도 다음을 출력하지 마라: 선택지 내용, 괄호, "정답", "정답 번호", 콜론, 줄바꿈, 추가 설명.

[판단 규칙]
- 선택지는 {options}들 중에만 있다.

---
[참고 자료]
{context}

---
[질문]
{question}

[선택지]
{options}
""")

saq_llm_prompt_eng = ChatPromptTemplate.from_template("""
You are an agricultural expert. Answer the following question clearly and concisely.

Provide a direct answer based on your knowledge.

[Question]
{question}
""")

saq_llm_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 질문에 답하라.

중요: 출력 형식을 반드시 지켜라.

[출력 형식 규칙]
- 답변 단어 또는 수치 하나만 출력한다.
  예: 12월 15일
- 반드시 단답형(한 단어 또는 짧은 구)으로만 답한다.
- 어떤 경우에도 다음을 출력하지 마라: 괄호, "정답", "답변", 콜론, 줄바꿈, 추가 설명, 문장 형태의 답변.

---
[질문]
{question}
""")

saq_rag_prompt_eng = ChatPromptTemplate.from_template("""
You are an agricultural expert. Answer the question based on the provided reference materials.

Use the given context as the primary source for your answer. If the information is not available in the context, you may respond accordingly.

[Reference]
{context}

[Question]
{question}
""")


saq_rag_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 자료를 근거로 질문에 답하라.

중요: 출력 형식을 반드시 지켜라.

[출력 형식 규칙]
- 답변 단어 또는 수치 하나만 출력한다.
  예: 12월 15일
- 반드시 단답형(한 단어 또는 짧은 구)으로만 답한다.
- 어떤 경우에도 다음을 출력하지 마라: 괄호, "정답", "답변", 콜론, 줄바꿈, 추가 설명, 문장 형태의 답변.

[판단 규칙]
- 반드시 [참고 자료]에 명시된 정보만 사용한다.
- [참고 자료]에 답이 없으면 "모름"이라고만 출력한다.
- 추측하거나 외부 지식을 사용하지 마라.

---
[참고 자료]
{context}

---
[질문]
{question}
""")


iter_rag_prompt = ChatPromptTemplate.from_template(
"""
당신은 반복 질의응답 개선 시스템이다.

입력:
- 질문
- 기존 답변

목표:
기존 답변이 질문에 대해 충분히 정확하고 완전한지 검토한다.
답변이 부족하거나 모호한 경우, 더 나은 답변을 위한 새로운 검색 쿼리를 생성한다.

평가 기준:
1. 답변이 질문의 핵심을 직접적으로 해결하는가
2. 중요한 정보가 누락되거나 불완전하지 않은가
3. 근거 없는 추측, 불필요한 일반화, 외부 지식이 포함되지 않았는가
4. 표현이 명확하고 이해하기 쉬운가

규칙:
- 기존 답변이 충분히 정확하고 완전하다면 iter_needed를 false로 설정한다.
- 답변이 부족하거나 개선 여지가 있다면 iter_needed를 true로 설정한다.
- iter_needed가 true일 경우, 기존 답변의 부족한 부분을 보완할 수 있는 새로운 검색 쿼리를 new_query에 작성
한다.
- new_query는 retriever에 바로 입력할 수 있도록 간결하고 명확한 검색어로 작성한다.
- 새로운 사실을 만들어내지 말고, 입력된 질문과 답변 범위를 벗어나지 않는다.
- 설명, 해설, 문장 추가 없이 반드시 지정된 JSON 배열만 출력한다.

출력 형식 (JSON 배열만 출력):

[
  {{
    "iter_needed": true | false,
    "new_query": "새로운 검색 쿼리"
  }}
]

--------------------
질문:
{question}

기존 답변:
{generated}
"""
)