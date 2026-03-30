from langchain_core.prompts import ChatPromptTemplate

know_prompt_eng = ChatPromptTemplate.from_template(
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


know_prompt = ChatPromptTemplate.from_template(
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