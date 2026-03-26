from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List

class BaseModel(ABC):
    
    @abstractmethod
    def answer_once(self, gen_input: Dict[str, Any]) -> Tuple[str, str, str, str]:
        pass
    
    def answer_all(self, input_dics: list[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
        return [self.answer_once(p) for p in input_dics]