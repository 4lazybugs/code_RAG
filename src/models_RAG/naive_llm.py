from .base import BaseExpert
from langchain_ollama.llms import OllamaLLM
# --------------------------------------------------

class RawLlmExpert(BaseExpert):
    """Expert that uses raw LLM without retrieval"""
    
    def setup(self, retriever_mode: str) -> None:
        self.model = OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)

    def handle(self, question: str) -> str:
        try:
            return self.model.invoke(question).strip()
        except Exception as e:
            return f"[오류]: {str(e)}"
