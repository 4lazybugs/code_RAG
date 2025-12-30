from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple
from load_params import get_config

class BaseExpert(ABC):
    def __init__(self, retriever_map: Dict[str, Any], retriever_mode: str):
        self.args = get_config()
        self.retriever_map = retriever_map
        self.setup(retriever_mode)

    @abstractmethod
    def setup(self, retriever_mode: str) -> None:
        pass

    @abstractmethod
    def handle(self, question: str) -> Union[str, Tuple[str, Dict[str, Any]]]:
        pass
