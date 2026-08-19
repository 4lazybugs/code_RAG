from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
from enum import Enum
import asyncio

from .ai.qwen_rag import QwenRAG
from .auth.dependencies import get_user_identifier

import httpx  # 외부 API 호출용


# --- App Initialization ---
app = FastAPI()

# --- AI Configuration ---
rag = QwenRAG()

MAX_INPUT_LENGTH = 200  # 질의 문구 최대 글자수 (문서에 정의한 항목크기 기준)
class RAGRequest(BaseModel):
    query: str
    farm_cd: Optional[str] = None
    house_cd: Optional[str] = None
    basetime: Optional[str] = None

# --- 사업단 API 엔드포인트 정의 ---
class SajubdanEndpoint(str, Enum):
    ENV_SENSOR = "https://itconv.co.kr:16706/genAi/pilot/api/sensorData"        # 환경데이터 조회
    NUTRIENT = "https://itconv.co.kr:16706/genAi/pilot/api/nutrientData"        # 양액데이터 조회 (실제 경로명 확인 필요)
    SPECTRAL_DETECTION = "https://itconv.co.kr:16706/genAi/pilot/api/spectralDetection"  # 생성요소AI 조회


class SajubdanAPIError(Exception):
    """사업단 API 응답이 실패(E) 상태이거나 형식이 어긋날 때"""
    pass


# --- 공통 호출 함수 (핵심 로직 단 하나) ---
async def call_sajubdan_api(
    endpoint: SajubdanEndpoint,
    farm_cd: str,
    house_cd: str,
    basetime: str,
    timeout: float = 30.0
) -> list[dict[str, Any]]:
    """
    사업단 측 API(환경/양액/생성요소AI) 공통 호출기.
    세 API 모두 요청/응답 포맷이 동일하므로 하나의 함수로 처리.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            endpoint.value,
            json={"farm_cd": farm_cd, "house_cd": house_cd, "basetime": basetime},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()

    rs_msg = data.get("rsMsg", {})
    if rs_msg.get("statusCode") != "S":
        raise SajubdanAPIError(
            f"[{endpoint.name}] 조회 실패: {rs_msg.get('statusMsg', '알 수 없는 오류')}"
        )

    return data.get("dataList1", [])

async def flow_itc(request: RAGRequest):
    try:
        env_data, nutrient_data, spectral_data = await asyncio.gather(
            call_sajubdan_api(SajubdanEndpoint.ENV_SENSOR, request.farm_cd, request.house_cd, request.basetime),
            call_sajubdan_api(SajubdanEndpoint.NUTRIENT, request.farm_cd, request.house_cd, request.basetime),
            call_sajubdan_api(SajubdanEndpoint.SPECTRAL_DETECTION, request.farm_cd, request.house_cd, request.basetime),
        )
    except SajubdanAPIError as e:
        return {"message": str(e), "status": "E"}
    except httpx.HTTPError as e:
        return {"message": f"사업단 API 통신 오류: {str(e)}", "status": "E"}

    # 2. VDB + RDB(성대측 로직) — 3개 데이터를 컨텍스트로 최종 답변 생성
    final_answer = await rag.chat_with_context(
        query=request.query,
        env_context=env_data,
        nutrient_context=nutrient_data,
        spectral_context=spectral_data
    )

    return {
        "message": "Answer successful",
        "status": "S",
        "farm_cd": request.farm_cd,
        "house_cd": request.house_cd,
        "basetime": request.basetime,
        "llmAnswer": final_answer
    }

# --- OpenAI Compatible Endpoints ---
@app.get("/v1/models")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen-rag",
                "object": "model"
            }
        ]
    }


# --- Flow ② : 사업단 오케스트레이션 (조건 미충족 → query만 존재) ---
async def flow_orchestration(request: RAGRequest):
    # 1. VDB 검색
    # 2. 1차 답변 생성
    answer = rag.chat(request.query)

    return {
        "message": "Answer successful",
        "status": "S",
        "llmAnswer": answer
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: RAGRequest,
    user_id: str = Depends(get_user_identifier)
):
    if not request.query.strip():
        return {"message": "문구 작성 부탁드립니다.", "status": "E"}

    if len(request.query) > MAX_INPUT_LENGTH:
        return {
            "message": f"질의 문구는 {MAX_INPUT_LENGTH}자 이내로 작성해주세요.",
            "status": "E"
        }

    is_itc = all([request.farm_cd, request.house_cd, request.basetime])

    try:
        if is_itc:
            return await flow_itc(request)
        else:
            return await flow_orchestration(request)
    except Exception as e:
        import traceback
        traceback.print_exc()  # 서버 터미널에 상세 에러 출력
        return {"message": f"서버 문제 발생했습니다: {str(e)}", "status": "E"}  # 임시로 에러 내용도 응답에 포함


@app.get("/")
async def root():
    return {"message": "FastAPI is running"}



async def flow_itc(request: RAGRequest):
    try:
        env_data, nutrient_data, spectral_data = await asyncio.gather(
            call_sajubdan_api(SajubdanEndpoint.ENV_SENSOR, request.farm_cd, request.house_cd, request.basetime),
            call_sajubdan_api(SajubdanEndpoint.NUTRIENT, request.farm_cd, request.house_cd, request.basetime),
            call_sajubdan_api(SajubdanEndpoint.SPECTRAL_DETECTION, request.farm_cd, request.house_cd, request.basetime),
        )
    except SajubdanAPIError as e:
        return {"message": str(e), "status": "E"}
    except httpx.HTTPError as e:
        return {"message": f"사업단 API 통신 오류: {str(e)}", "status": "E"}

    # TODO: chat_with_context 구현 전까지 임시로 원본 데이터만 반환해서 API 연동 확인
    return {
        "message": "Answer successful (TEST MODE - LLM 미적용)",
        "status": "S",
        "farm_cd": request.farm_cd,
        "house_cd": request.house_cd,
        "basetime": request.basetime,
        "env_data": env_data,
        "nutrient_data": nutrient_data,
        "spectral_data": spectral_data
    }