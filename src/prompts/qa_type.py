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
