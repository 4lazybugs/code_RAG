from langchain_core.prompts import ChatPromptTemplate

md2text_prompt = ChatPromptTemplate.from_template(
"""
아래 문서 페이지의 내용을 내용 누락 없이 자연스러운 줄글로 변환하라.

[규칙]
- 표, 목록, 수치, 항목명 등 모든 정보를 빠짐없이 포함할 것
- 문서 구조(헤더, 불릿 등)는 제거하고 자연스러운 문장으로 변환
- 추론하거나 내용을 추가하지 말 것
- 텍스트만 출력하라. 다른 설명 금지.

문서:
{page_text}
"""
)

boundary_prompt = ChatPromptTemplate.from_template(
"""
아래는 매뉴얼 문서의 연속된 텍스트다.
현재까지 읽은 내용과 새로 추가된 내용 사이에 문맥 단절이 있는지 판단하라.

[중요 규칙]
1. JSON만 출력하라.
2. 코드블록 금지.
3. 문맥 단절이란 주제가 바뀌는 것을 의미한다.
4. 같은 주제라도 핵심 대상이 바뀌면 문맥 단절로 본다.

출력 스키마:
{{
  "is_boundary": true,
  "reason": "판단 이유"
}}

지금까지 읽은 내용:
{accumulated}

새로 추가된 내용:
{new_text}
"""
)

lumber_prompt = ChatPromptTemplate.from_template(
"""
아래는 매뉴얼 문서에서 순서대로 이어지는 문단들이며, 각 문단 앞에 ID가 붙어 있다.

문단들을 순서대로 읽으면서, 어느 ID부터 내용/주제가 바뀌기 시작하는지 판단하라.
그 ID는 새로운 청크의 시작점이 된다.

[중요 규칙]
1. JSON만 출력하라.
2. 코드블록 금지.
3. 주제가 바뀐다는 것은 핵심 대상, 목적, 절차 단위가 달라짐을 의미한다.
4. 같은 주제라도 핵심 대상이 바뀌면 문맥 단절로 본다.
5. boundary_id는 "직전 청크의 마지막 ID"를 의미한다. 즉 문맥 단절이 없으면
   마지막 문단의 ID를 그대로 반환하라.
6. boundary_id는 반드시 주어진 문단들의 ID 범위 안의 정수여야 한다.

출력 스키마:
{{
  "boundary_id": 0,
  "reason": "판단 이유"
}}

문단들:
{numbered_paragraphs}
"""
)

decision_prompt = ChatPromptTemplate.from_template(
"""
아래 텍스트가 RAG 청크로 유효한지 판단하라.

[중요 규칙]
1. JSON만 출력하라.
2. 코드블록 금지.
3. 문서에 실제로 있는 정보만 기준으로 판단할 것.

[판단 기준]
유효한 텍스트: 명확한 대상과 구체적 사실이 있어 질문-답변을 만들 수 있는 내용
무효한 텍스트: 아래 중 하나라도 해당하면 무효
  - 제목, 목차, 라벨로만 구성됨
  - 동일 문장이 10회 이상 반복됨
  - OCR 오류로 내용 판독 불가
  - 본문 없이 기호나 빈 셀로만 구성된 표

출력 스키마:
{{
  "is_useful": true,
  "reason": "판단 이유"
}}

텍스트:
{raw_text}
"""
)

meta_prompt = ChatPromptTemplate.from_template(
"""
아래 문서를 읽고 문서의 핵심 내용을 요약하여 추출하라.

[중요 규칙]
1. JSON만 출력하라.
2. 코드블록 금지.
3. 추론하지 말고 문서에 실제로 있는 정보만 추출할 것.

출력 스키마:
{{
  "md_summary": "문서 전체 핵심 내용 요약 (2~3문장)"
}}

문서:
{raw_text}
"""
)

agentic_prompt = ChatPromptTemplate.from_template(
"""
아래는 문서 전체 요약과 본문 텍스트다.

문서 요약:
{md_summary}

본문:
{raw_text}

요약을 참고하여 본문을 의미적으로 독립적인 자연어 청크들로 분리하라.

[중요 규칙]
1. JSON 배열만 출력하라.
2. 코드블록 금지.
3. 각 청크는 단독으로 읽어도 의미가 통해야 한다.
4. 원문 텍스트만 사용하고 추론하거나 내용을 추가하지 말 것.

출력 스키마:
[
  {{"chunk": "청크 내용"}},
  {{"chunk": "청크 내용"}}
]

"""
)