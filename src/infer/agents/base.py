from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List

class BaseModel(ABC):
    
    @abstractmethod
    def answer_once(self, gen_input: Dict[str, Any]) -> Tuple[str, str, str, str]:
        pass
    
    @abstractmethod
    def answer_all(self, gen_inputs: List[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
        pass