from typing import Dict, List, Any, Callable
from pathlib import Path
import json

fetch_dict: Dict[str, Callable] = {}

def add_fetch_key(key: str):
    def deco(fn: Callable) -> Callable:
        if key in fetch_dict:
            raise KeyError(f"Duplicate fetch key detected: {key}")
        fetch_dict[key] = fn
        return fn
    return deco


class FetchData:
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.data = self._load_data()  # List[dict]

    def _load_data(self) -> List[dict]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ########### fetching functions for each metric ###########
    @add_fetch_key("rouge1")
    def fetch_rouge1(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
        }
    
    @add_fetch_key("rougel")
    def fetch_rougel(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
        }
    
    @add_fetch_key("em")
    def fetch_em(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
        }
    
    @add_fetch_key("bert")
    def fetch_bert(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
        }

    @add_fetch_key("sbert")
    def fetch_sbert(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
        }
    
    @add_fetch_key("mrr")
    def fetch_mrr(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
            "gen_docs":   [s.get("gen_docs") for s in self.data],
            "ref_docs":   [s.get("ref_docs") for s in self.data],
        }
    
    @add_fetch_key("ground")
    def fetch_ground(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
            "gen_docs":   [s.get("gen_docs") for s in self.data],
            "ref_docs":   [s.get("ref_docs") for s in self.data],
        }
    
    @add_fetch_key("correct")
    def fetch_correctness(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
        }
    
    @add_fetch_key("recall")
    def fetch_recall(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
            "gen_docs":   [s.get("gen_docs") for s in self.data],
            "ref_docs":   [s.get("ref_docs") for s in self.data],
        }

    @add_fetch_key("hotpot")
    def fetch_recall(self) -> Dict[str, List[Any]]:
        return {
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
            "gen_docs":   [s.get("gen_docs") for s in self.data],
            "ref_docs":   [s.get("ref_docs") for s in self.data],
        }

    def _to_rows(self, cols):
        if not cols:
            return []

        n = len(next(iter(cols.values())))
        if any(len(v) != n for v in cols.values()):
            raise ValueError("Length mismatch")

        return [{k: v[i] for k, v in cols.items()} for i in range(n)]

    def fetch(self, metric: str) -> List[Dict[str, Any]]:
        try:
            fn = fetch_dict[metric]
        except KeyError:
            raise KeyError(f"Unknown metric: {metric}") from None

        cols = fn(self)                # dict of lists (필요한 키만)
        batch_data = self._to_rows(cols)  # list of dict (score_all 호환)
        return batch_data
