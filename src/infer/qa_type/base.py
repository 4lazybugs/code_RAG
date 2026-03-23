# base.py
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

@dataclass
class QAtype:
    prompt: Any = None
    load_input: Optional[Callable] = None
    load_output: Optional[Callable] = None

    def set_prompt(self, prompt):
        self.prompt = prompt
        return self

    def set_inputs(self, load_input):
        self.load_input = load_input
        return self

    def set_outputs(self, load_output):
        self.load_output = load_output
        return self

    def build(self):
        return self

