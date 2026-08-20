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
    query_rag: str
    farm_cd: Optional[str] = None
    house_cd: Optional[str] = None
    basetime: Optional[str] = None

# --- 사업단 API 엔드포인트 정의 ---
class Endpoint_ui(str, Enum):
    ENV_SENSOR = "https://itconv.co.kr:16706/genAi/pilot/api/sensorData"        # 환경데이터 조회
    NUTRIENT = "https://itconv.co.kr:16706/genAi/pilot/api/nutrientData"        # 양액데이터 조회 (실제 경로명 확인 필요)
    SPECTRAL_DETECTION = "https://itconv.co.kr:16706/genAi/pilot/api/spectralDetection"  # 생성요소AI 조회


class SajubdanAPIError(Exception):
    """사업단 API 응답이 실패(E) 상태이거나 형식이 어긋날 때"""
    pass


# --- 공통 호출 함수 (핵심 로직 단 하나) ---
async def call_rdb_api(
    endpoint: Endpoint_ui,
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


def latest_row(data: list[dict], keys: list[str] | None = None) -> dict:
    """RDB 시계열 데이터에서 최신 1건만 추출 (마지막 row = 최신, timestamp 정렬 확인됨)"""
    if not data:
        return {}
    row = data[-1]
    if keys:
        return {k: row[k] for k in keys if k in row}
    return row

# --- 디버깅 전용: RDB 크기만 확인 (flow_itc와 완전히 별개) ---
@app.post("/v1/debug/rdb-size")
async def debug_rdb_size(request: RAGRequest):
    if not (request.farm_cd and request.house_cd and request.basetime):
        return {"message": "farm_cd/house_cd/basetime 필요", "status": "E"}

    try:
        env_data, nutrient_data, spectral_data = await asyncio.gather(
            call_rdb_api(Endpoint_ui.ENV_SENSOR, request.farm_cd, request.house_cd, request.basetime),
            call_rdb_api(Endpoint_ui.NUTRIENT, request.farm_cd, request.house_cd, request.basetime),
            call_rdb_api(Endpoint_ui.SPECTRAL_DETECTION, request.farm_cd, request.house_cd, request.basetime),
        )
    except SajubdanAPIError as e:
        return {"message": str(e), "status": "E"}
    except httpx.HTTPError as e:
        return {"message": f"사업단 API 통신 오류: {str(e)}", "status": "E"}

    return {
        "status": "S",
        "env_rows": len(env_data),
        "env_chars": len(str(env_data)),
        "env_first_row": env_data[0] if env_data else None,   # 추가
        "env_last_row": env_data[-1] if env_data else None,   # 추가
        "nutrient_rows": len(nutrient_data),
        "nutrient_chars": len(str(nutrient_data)),
        "spectral_rows": len(spectral_data),
        "spectral_chars": len(str(spectral_data)),
    }

async def flow_itc(request: RAGRequest):
    if request.farm_cd and request.house_cd and request.basetime:
        try:
            env_data, nutrient_data, spectral_data = await asyncio.gather(
                call_rdb_api(Endpoint_ui.ENV_SENSOR, request.farm_cd, request.house_cd, request.basetime),
                call_rdb_api(Endpoint_ui.NUTRIENT, request.farm_cd, request.house_cd, request.basetime),
                call_rdb_api(Endpoint_ui.SPECTRAL_DETECTION, request.farm_cd, request.house_cd, request.basetime),
            )
        except SajubdanAPIError as e:
            return {"message": str(e), "status": "E"}
        except httpx.HTTPError as e:
            return {"message": f"사업단 API 통신 오류: {str(e)}", "status": "E"}

        env_latest = latest_row(env_data, keys=["outdoor_temperature"])
        nutrient_latest = latest_row(nutrient_data, keys=["zone01_set"])

        final_answer = rag.chat(
            prompt=request.query_rag,
            flag_paid=True,
            env_context=env_latest,
            nutrient_context=nutrient_latest,
            spectral_context=None,
        )

        return {
            "message": "Answer successful",
            "status": "S",
            "farm_cd": request.farm_cd,
            "house_cd": request.house_cd,
            "basetime": request.basetime,
            "llmAnswer": final_answer
        }

    final_answer = rag.chat(prompt=request.query_rag, flag_paid=False)

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
    answer = rag.chat(prompt=request.query_rag)
    interpretation = rag.interpret(request.query_rag)

    return {
        "answer": answer,
        "interpretation": interpretation
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: RAGRequest,
    user_id: str = Depends(get_user_identifier)
):
    if not request.query_rag.strip():
        return {"message": "문구 작성 부탁드립니다.", "status": "E"}

    if len(request.query_rag) > MAX_INPUT_LENGTH:
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


class RetrieveRequest(BaseModel):
    query: str


@app.post("/v1/debug/retrieve")
async def debug_retrieve(request: RetrieveRequest):
    if not request.query_rag.strip():
        return {"message": "문구 작성 부탁드립니다.", "status": "E"}

    try:
        results = rag.retrieve_only(request.query_rag)
        return {
            "status": "S",
            "query": request.query_rag,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"서버 문제 발생했습니다: {str(e)}", "status": "E"}

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}

