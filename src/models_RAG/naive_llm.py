from .base import BaseExpert
from langchain_openai import ChatOpenAI
from utils import get_config
# --------------------------------------------------

class RawLlmExpert(BaseExpert):
    """Expert that uses raw LLM without retrieval"""
    CFG = get_config()
    model_name = CFG.model_name
    def setup(self, retriever_mode: str) -> None:
        self.model = ChatOpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            model=self.model_name,
            temperature=0.0,
        )

    def handle(self, question: str):
        try:
            return self.model.invoke(question)   # ✅ .strip() 제거
        except Exception as e:
            return f"[오류]: {str(e)}"

