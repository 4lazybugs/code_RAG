from typing import Any, Dict, List
import json
import random
from tqdm import tqdm
from src.synthesis.params import Params
from langchain_core.prompts import ChatPromptTemplate

Q_TYPES = ["정의형", "이유형", "역할형", "방법형", "조건형", "비교형", "특징형"]
TOPICS = [
    "벼", "밀", "옥수수", "토양 pH", "질소 비료", "인산 비료", "칼륨 비료",
    "관개", "배수", "잡초 관리", "병해충 방제", "광합성", "온도 관리",
    "수확 후 저장", "스마트팜", "환경 센서", "시설재배", "윤작"
]
DIFFICULTIES = ["easy", "medium", "hard"]


def qa_from_prompt(params: Params, num_samples: int = 1) -> List[Dict[str, Any]]:
    llm = params.get_llm("llm")
    start_id = int(params.get_params("naive", "start_id", 0))
    prompt = params.get_params("naive", "prompt")

    chain = prompt | llm

    qas_list: List[Dict[str, Any]] = []
    cur_id = start_id

    ok_cnt = 0
    json_fail_cnt = 0
    invoke_fail_cnt = 0

    pbar = tqdm(range(num_samples), desc="single", dynamic_ncols=True)

    for idx, _ in enumerate(pbar, start=1):
        pbar.set_postfix({
            "pair": idx,
            "id": cur_id,
            "ok": ok_cnt,
            "json_fail": json_fail_cnt,
            "invoke_fail": invoke_fail_cnt,
        })

        tqdm.write(f"id={cur_id}")

        try:
            resp = chain.invoke({
                "q_type": random.choice(Q_TYPES),
                "topic": random.choice(TOPICS),
                "difficulty": random.choice(DIFFICULTIES),
            })
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            invoke_fail_cnt += 1
            tqdm.write(f"  ❌ invoke failed: {type(e).__name__}: {e}")
            cur_id += 1
            continue

        try:
            arr = json.loads(content)
        except Exception:
            json_fail_cnt += 1
            snippet = content[:300].replace("\n", "\\n")
            tqdm.write(f"  ❌ JSON parse failed. raw[:300]={snippet}")
            cur_id += 1
            continue

        if not isinstance(arr, list):
            json_fail_cnt += 1
            tqdm.write("  ❌ JSON is not a list")
            cur_id += 1
            continue

        ok_cnt += 1

        if len(arr) > 0 and isinstance(arr[0], dict):
            q_preview = str(arr[0].get("question", ""))[:120].replace("\n", " ")
            tqdm.write(f"  ✅ parsed. question_preview={q_preview}...")

        for j, item in enumerate(arr):
            qas_list.append({
                "id": cur_id + j,
                "question": item["question"],
                "answer": item["answer"],
                "retrieval_needed": 0,   # 문서 기반이 아니므로
            })

        cur_id += len(arr)

    return qas_list



# ===== QA Prompt ===== : short answer question
saqgen_prompt = ChatPromptTemplate.from_template(
"""
다음 context에서 **RAG 평가용 Ground Truth 질문/답변 1쌍**을 JSON 배열로 생성하라.

# 0) 절대 원칙
- 질문/답변은 **context의 명시 정보만** 사용한다.
- 질문은 **정확히 1개**, **단답형**으로만 생성한다.
- 질문 끝에 반드시 **“단답형으로 답하라.”** 를 포함한다.
- 답변은 수치/명칭/값 등 **짧게** 작성한다.

# 1) 모호성(ambiguity) 금지 규칙 (가장 중요)
질문은 반드시 아래 조건을 모두 만족해야 한다.

(1) 질문 대상의 "정체"가 문장 안에 포함되어야 한다.
- 대상은 **고유 식별자(정식 명칭, 모델명, 기관명, 품목명, 시스템명, 사업명, 정책명, 프로젝트명, 공정명, 약어의 풀네임 등)** 중 하나로 특정한다.
- context에 고유 식별자가 없으면, 해당 정보로는 질문을 만들지 말고 **다른 사실(수치/값/명칭)** 로 질문을 다시 만든다.

(2) 다음 표현은 질문에 절대 사용하지 않는다.
- 지시어/대명사: “이것, 그것, 해당, 이런, 저런, 위의, 아래의, 본 문서, 이 문서, 여기, 거기”
- 과도한 일반명사: “기자재, 장비, 시스템, 이미지, 표, 자료, 현황, 내용, 데이터”
  - 단, **바로 뒤에 고유 식별자 또는 정확한 명칭**이 붙어 구체화되는 경우만 예외로 허용
    (예: “관수 제어기 모델명 ABC-123”, “시설하우스 A동” 등)

(3) 질문은 context 밖에서도 단독으로 완전히 명확해야 한다.
- “무엇의/누구의/어느/어떤”이 생략되어 의미가 흔들리면 실패다.
- 질문은 항상 “무엇(고유명칭)의 어떤 속성(수치/값/명칭)?” 구조로 작성한다.

# 2) 질문 생성 절차 (필수)
- Step A: context에서 **고유 식별자(명칭/모델/기관/품목/사업/정책/프로젝트/공정 등)** 를 1개 이상 찾는다.
- Step B: 그 고유 식별자와 **직접 연결된 단일 사실(수치/값/명칭)** 을 1개 선택한다.
- Step C: 그 사실만을 묻는 **단답형 질문 1개**를 만든다.
- Step D: 아래 “자체 검증 체크”에서 하나라도 실패하면 질문을 폐기하고 다시 생성한다.

# 3) 자체 검증 체크 (통과 못 하면 재작성)
- [ ] 질문에 고유 식별자(명칭/모델/기관/품목 등)가 포함되어 있는가?
- [ ] “기자재/장비/이미지/현황/자료/내용” 같은 일반명사가 단독으로 쓰이지 않았는가?
- [ ] “해당/이것/그것/본 문서” 같은 지시어가 없는가?
- [ ] 답변이 context에 그대로 존재하는 단일 값(수치/명칭)인가?

# 4) 출력 형식 (매우 중요)
- 답변 값 하나만 출력한다.
  예: 12월 15일
- 어떤 경우에도 다음을 출력하지 마라: JSON, 괄호, "답변", "answer", 콜론, 줄바꿈, 추가 설명.

# context
{context}
"""
)