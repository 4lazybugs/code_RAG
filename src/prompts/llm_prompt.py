from langchain_core.prompts import ChatPromptTemplate


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