from langchain_core.prompts import ChatPromptTemplate

naive_prompt = ChatPromptTemplate.from_template(
"""
다음 context를 기반으로 Ground Truth QA 1개를 생Mapping[str, Mapping[str, Any]] = field(default_facto는다.
3) 답은 context에 있는 단일 값이어야 한다.
4) 추론이나 외부 지식 사용 금지.

[출력 형식]
설명 없이 JSON 배열 1개만 출력:

[
  {{
    "id": {id},
    "question": "...",
    "answer": "..."
  }}
]

# context
{md}
"""
)

consulting_prompt = ChatPromptTemplate.from_template(
"""
# 역할(Role)

너는 RAG DB 기반 평가용 QA를 설계하는 전문가다.
목표는 검색 정확도, 표 해석 능력, 그리고 조치사항·의견·결과 통합 이해 능력을 평가하는 것이다.


# 목표(Goal)

입력된 context를 기반으로,
다음 두 유형 중 하나를 선택해 QA를 하나만 생성하라:

## 1. 표 기반 추출 QA
- 행/열 맥락을 함께 이해해야 답할 수 있는 질문
- 단순 셀 값 복사, 빈칸, 의미 없는 숫자, 체크표시만 있는 항목 제외

## 2. 통합 이해 QA  
- 최소 2개 이상의 구간(예: 컨설팅 의견 + 조치사항)을 연결해야 답할 수 있는 질문
- 단일 문장 복사형 질문 금지


# 질문 구성 필수 구조 (Mandatory Question Structure)

모든 질문은 다음 구조를 반드시 따라야 한다:

[대상 식별 정보] + [구체적 내용 질문]

- 질문 첫 문장은 반드시 대상 식별 정보로 시작해야 한다.
- 대상 식별 정보에는 다음 중 최소 1개 이상 포함:
  · 농가명
  · 대표자명
  · 사업명
  · 문서 차수(1차/2차 등)
  · 보고서명

- "전설팀의 의견에 따르면..."처럼
  대상 없이 시작하는 질문은 생성 금지.


# 구조 위반 처리 규칙 (Structural Enforcement Rule)

- 질문이 대상 식별 정보로 시작하지 않으면,
  반드시 질문을 수정하여 재작성한 후 출력한다.
- 대상 식별 정보가 누락된 경우, context에서 적절한 식별 정보를 찾아
  질문 첫 문장에 포함시켜야 한다.
- 위 조건을 만족하지 않으면 출력하지 않는다.


# 출력 형식

설명 없이 JSON 배열 1개만 출력:

[
  {{
    "id": {id},
    "question": "...",
    "answer": "..."
  }}
]

# 내부 검증 단계 (Self-Check)

출력 전 반드시 확인:

- 두 유형 중 하나에 정확히 해당하는가?
- 질문에 대상 식별 정보가 포함되어 있는가?
- 단순 추출형으로 변질되지 않았는가?
- 답이 context에 정확히 존재하는가?
- 중복·모호성·맥락 누락이 없는가?


# context
{md}
"""
)


hotpot_prompt = ChatPromptTemplate.from_template(
"""
다음 context에서 HotpotQA 스타일 multi-hop Ground Truth 1개를 생성하라.

[핵심 규칙]
1) 반드시 서로 다른 두 문단(A, B)을 사용해야 하며,
   질문은 A와 B를 모두 읽어야만 답할 수 있어야 한다.

2) supporting_fact_a:
   - 문단 A에서 답에 필요한 핵심 근거 1개를 원문 그대로 발췌
   - 문맥상 단독 이해 가능할 만큼 충분히 길 것 (약 200 tokens 이상 권장)

3) supporting_fact_b:
   - 문단 B에서 답에 필요한 핵심 근거 1개를 원문 그대로 발췌
   - 문단 A 내용 포함 금지

4) 질문 규칙:
   - 지시어/모호한 일반명사 금지
   - 답은 context에 명시된 단일 값

[출력 형식]
설명 없이 JSON 배열 1개만 출력:

[
  {{
    "id": {id},
    "question": "...",
    "answer": "...",
    "supporting_fact_a": "...",
    "supporting_fact_b": "..."
  }}
]

# context
{md_a}
{md_b}
"""
)