from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import List

from .ai.qwen_rag import QwenRAG
from .auth.dependencies import get_user_identifier
from .auth.throttling import apply_rate_limit

# --- App Initialization ---
app = FastAPI()

# --- AI Configuration ---
rag = QwenRAG()

# --- OpenAI Models ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]

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

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    user_id: str = Depends(get_user_identifier) # user id 식별
):
    # 식별된 user_id 기준으로 요청 빈도를 검사
    # 일정 시간 내 요청 수가 threshold를 초과하면 요청 차단
    user_prompt = request.messages[-1].content

    response_text = rag.chat(user_prompt)

    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ]
    }

MAX_INPUT_LENGTH = 200  # 질의 문구 최대 글자수 (문서에 정의한 항목크기 기준)

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    user_id: str = Depends(get_user_identifier)
):
    if not request.messages or not request.messages[-1].content.strip():
        return {
            "message": "문구 작성 부탁드립니다.",
            "status": "E"
        }

    user_prompt = request.messages[-1].content

    if len(user_prompt) > MAX_INPUT_LENGTH:
        return {
            "message": f"질의 문구는 {MAX_INPUT_LENGTH}자 이내로 작성해주세요.",
            "status": "E"
        }

    try:
        response_text = rag.chat(user_prompt)
    except Exception:
        return {
            "message": "서버 문제 발생했습니다.",
            "status": "E"
        }

    return {
        "message": "Answer successful",
        "status": "S",
        "llmAnswer": response_text
    }

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}