from langchain_core.prompts import ChatPromptTemplate

relv_prompt = ChatPromptTemplate.from_template(
"""
당신은 신중한 평가자입니다.

Task:
주어진 DOCUMENT와 QUERY를 보고, 해당 DOCUMENT가 QUERY에 답하는 데 얼마나 관련 있는지를 평가하세요.

반드시 아래 형식의 유효한 JSON만 출력하세요:
{{"score": <0과 1 사이의 실수>, "label": <0 또는 1>}}

Scoring guide:
- 1.0 = 질문에 직접적으로 도움이 되는 매우 높은 관련성
- 0.8 = 대부분 관련 있고 도움이 됨
- 0.5 = 부분적으로 관련 있음 / 어느 정도 도움이 됨
- 0.2 = 약하게 관련 있음
- 0.0 = 관련 없음 또는 질문에 도움이 되지 않음

Rules:
- 사실적 지지 여부가 아니라 "관련성"을 평가하세요.
- 반드시 문서와 질문만 사용하세요.
- 외부 지식을 사용하지 마세요.
- 문서가 질문에 완전히 답하지 않더라도 관련성이 있을 수 있습니다.
- 문서가 관련 없거나 거의 관련이 없다면 낮은 점수를 주십시오.
- 반드시 JSON만 출력하세요.

DOCUMENT:
{doc}

QUERY:
{claim}
"""
)

faith_prompt = ChatPromptTemplate.from_template(
"""
당신은 신중한 평가자입니다.

Task:
주어진 DOCUMENT와 ANSWER를 보고, DOCUMENT가 ANSWER를 얼마나 근거 있게 뒷받침하는지 평가하세요.

반드시 아래 형식의 유효한 JSON만 출력하세요:
{{"score": <0과 1 사이의 실수>, "label": <0 또는 1>}}

Scoring guide:
- 1.0 = 답변이 문서에 완전히 근거하고 있음
- 0.8 = 대부분 근거가 있으며 일부만 부족함
- 0.5 = 부분적으로 근거 있음 / 불확실함
- 0.2 = 근거가 약하며 대부분 뒷받침되지 않음
- 0.0 = 근거 없음 또는 문서와 모순됨

Rules:
- 반드시 문서만을 기준으로 판단하세요.
- 외부 지식을 사용하지 마세요.
- 답변 중 문서에 없는 내용이 포함되면 점수를 낮추세요.
- 환각(hallucination)이나 문서에 없는 추가 정보가 있으면 반드시 감점하세요.
- 반드시 JSON만 출력하세요.

DOCUMENT:
{doc}

ANSWER:
{claim}
"""
)