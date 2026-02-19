from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

@dataclass(frozen=True)
class Params:
    global_: Mapping[str, Any] = field(default_factory=dict)
    per: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def get_llm(self, key: str) -> Any:
        if key not in self.global_:
            raise KeyError(f"params.global_ missing key: {key}")
        return self.global_[key]

    def get_params(self, strategy_name: str, key: str, default: Any = None) -> Any:
        return self.per.get(strategy_name, {}).get(key, default)