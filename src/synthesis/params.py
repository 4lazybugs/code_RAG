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

    def get_params(self, key: str, default=None):
        return self.per.get(key, default)  # 중간 네임스페이스 제거