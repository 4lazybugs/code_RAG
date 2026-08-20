from langchain_core.prompts import ChatPromptTemplate

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
- 추측하거나 외부 지식을 사용하지 마라.

---
[참고 자료]
{context}

---
[질문]
{question}
""")

rag_consult_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 자료를 근거로 질문에 답하라.

[판단 규칙]
- 반드시 [참고 자료]에 명시된 정보만 사용한다.
- 추측하거나 외부 지식을 사용하지 마라.
- "참고 자료에 따르면", "자료에 의하면" 등 출처를 언급하는 표현 없이, 답변은 자연어로 자연스럽게 작성한다.

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


interpretation_prompt = ChatPromptTemplate.from_template("""
아래 질문을 분석해서 JSON으로만 답하라. 다른 텍스트는 절대 포함하지 마라.

[질문]
{question}

[출력 형식]
{{
  "crop": "작물명 (모르면 빈 문자열)",
  "modelType": ["예측" 또는 "탐지" 중 해당하는 것들],
  "task": ["생산량","병충해","생육단계" 중 해당하는 것들],
  "keyword": ["질문에서 추출한 핵심 키워드들"]
}}
""")

rag_paid_consult_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 아래 [참고 자료]를 근거로 질문에 답하라.

[참고 자료]에는 두 종류의 정보가 섞여 있을 수 있다.
1) 문서/매뉴얼 기반 지식 (VDB)
2) 실시간 환경·양액·병해진단 센서 데이터 (RDB, [환경 데이터]/[양액 데이터]/[생성요소AI 데이터] 태그로 표시됨)

[판단 규칙]
- 반드시 [참고 자료]에 명시된 정보만 사용한다.
- 추측하거나 외부 지식을 사용하지 마라.
- RDB 데이터는 필드명이 코드/약어로 되어 있으므로, 아래 [RDB 필드 설명]을 반드시 참고하여 값을 정확히 해석한다.
- [RDB 필드 설명]에 없는 필드는 값의 의미를 임의로 추정하지 않는다.
- zone01은 온실1, zone02는 온실2에 해당한다. 질문의 온실 번호와 데이터의 zone 번호를 혼동하지 않는다.
- "참고 자료에 따르면", "자료에 의하면" 등 출처를 언급하는 표현 없이, 답변은 자연어로 자연스럽게 작성한다.

[RDB 필드 설명]
■ 양액 (설정/누적)
- zone01_set / zone02_set: 온실1/온실2 설정량 (목표 물량, 시간당 L 또는 cc 기준)
- zone01_total / zone02_total: 온실1/온실2의 일일 누적 급액량 (매일 자정 기준 리셋)

■ 환경 센서
- outdoor_temperature: 외부 온도
- indoor_temperature1: 내부 온도1
- indoor_humidity1: 내부 습도1
- outdoor_cumulative_solar_radiation: 외부 누적 일사량
- air_insol: 일사량

■ 병해 진단 (생성요소AI 분석 결과)
- part_id: 병해 코드 — SP01(팁번), SP02(색변이/황화증상), SD001(탄저병), SN01(정상)
- class: 진단 분류값 — 0(정상), 1(잠복기), 2(증상초기), 3(증상후기)

---
[참고 자료]
{context}

---
[질문]
{question}
""")