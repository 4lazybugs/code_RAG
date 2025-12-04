from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple
from utils import get_config

class BaseExpert(ABC):
    """
    추상 베이스 클래스:
      - setup(): 리트리버·LLM·체인 초기화
      - handle(): 질문에 대해 응답 생성
    """
    def __init__(self, retriever_map: Dict[str, Any], retriever_mode: str):
        self.args = get_config()
        self.retriever_map = retriever_map
        self.setup(retriever_mode)
        
    @abstractmethod
    def setup(self, retriever_mode: str) -> None:
        """Initialize retriever, LLM, and chains"""
        pass

    @abstractmethod
    def handle(self, question: str) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Handle a question and return response"""
        pass