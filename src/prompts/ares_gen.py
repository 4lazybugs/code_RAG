from langchain_core.prompts import ChatPromptTemplate


q_gen_prompt = ChatPromptTemplate.from_template(
"""
다음 context를 기반으로 질문을 생성하라.
context: {md}
질문은 반드시 문서에서 직접 답을 찾을 수 있어야 하고, 문서의 핵심 정보를 묻는 명확한 질문이어야 한다.

Example #1
context:
토양 pH는 작물 생장에 중요한 영향을 미친다. 대부분의 작물은 pH 6.0~7.0 범위에서 잘 자란다.

question:
대부분의 작물이 잘 자라는 토양 pH 범위는 무엇인가?

Example #2
context:
질소 비료는 식물의 잎과 줄기 생장을 촉진하는 역할을 한다.

question:
질소 비료의 주요 역할은 무엇인가?

Example #3
context:
관개는 작물이 필요한 수분을 공급하기 위해 농경지에 물을 공급하는 과정이다.

question:
관개의 목적은 무엇인가?

[출력 형식]
설명 없이 JSON 배열 하나만 출력한다.

[
  {{
    "id": {id},
    "question": "..."
  }}
]
"""
)


ans_gen_prompt = ChatPromptTemplate.from_template(
"""
다음 context와 question을 기반으로 answer를 생성하라.
context: {md}, question: {question}
answer는 반드시 context에서 직접 찾을 수 있어야 하며, 외부 지식이나 추론을 사용하지 마라.
answer는 question에 대해 간결하고 정확하게 작성하라.

Example #1
context:
토양 pH는 작물 생장에 중요한 영향을 미친다. 대부분의 작물은 pH 6.0~7.0 범위에서 잘 자란다.

question:
대부분의 작물이 잘 자라는 토양 pH 범위는 무엇인가?

answer:
pH 6.0~7.0 범위이다.

Example #2
context:
질소 비료는 식물의 잎과 줄기 생장을 촉진하는 역할을 한다.

question:
질소 비료의 주요 역할은 무엇인가?

answer:
식물의 잎과 줄기 생장을 촉진하는 것이다.

Example #3
context:
관개는 작물이 필요한 수분을 공급하기 위해 농경지에 물을 공급하는 과정이다.

question:
관개의 목적은 무엇인가?

answer:
작물이 필요한 수분을 공급하기 위한 것이다.

[출력 형식]
설명 없이 JSON 배열 하나만 출력한다.

[
  {{
    "id": {id},
    "question": "{question}",
    "answer": "..."
  }}
]
"""
)