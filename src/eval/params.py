from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

class Params:
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.data = self._load_data()

    def _load_data(self) -> list[dict]:
        return json.loads(self.data_path.read_text(encoding="utf-8"))

    def get_param(self, key: str) -> list[Any]:
        return [record.get(key) for record in self.data]
    

'''
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
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "reference": [s.get("reference") or s.get("answer") for s in self.data],
            "generated":  [s.get("generated") or s.get("gen_answer") for s in self.data],
        }
    
    @add_fetch_key("rougel")
    def fetch_rougel(self) -> Dict[str, List[Any]]:
        return {
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "reference": [s.get("reference") or s.get("answer") for s in self.data],
            "generated":  [s.get("generated") or s.get("gen_answer") for s in self.data],
        }
    
    @add_fetch_key("em")
    def fetch_em(self) -> Dict[str, List[Any]]:
        return {
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "reference": [s.get("reference") or s.get("answer") for s in self.data],
            "generated":  [s.get("generated") or s.get("gen_answer") for s in self.data],
        }
    
    @add_fetch_key("bert")
    def fetch_bert(self) -> Dict[str, List[Any]]:
        return {
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "reference": [s.get("reference") or s.get("answer") for s in self.data],
            "generated":  [s.get("generated") or s.get("gen_answer") for s in self.data],
        }

    @add_fetch_key("sbert")
    def fetch_sbert(self) -> Dict[str, List[Any]]:
        return {
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "reference": [s.get("reference") or s.get("answer") for s in self.data],
            "generated":  [s.get("generated") or s.get("gen_answer") for s in self.data],
        }
    
    @add_fetch_key("recall")
    def fetch_recall(self) -> Dict[str, List[Any]]:
        return {
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "ref_docs": [s.get("reference_docs") for s in self.data],
            "gen_docs": [s.get("retrieved") for s in self.data],
        }

    @add_fetch_key("mrr")
    def fetch_mrr(self) -> Dict[str, List[Any]]:
        return {
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "ref_docs": [s.get("reference_docs") for s in self.data],
            "gen_docs": [s.get("retrieved") for s in self.data],
        }
    
    @add_fetch_key("ground")
    def fetch_ground(self) -> Dict[str, List[Any]]:
        return {
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "reference": [s.get("reference") or s.get("answer") for s in self.data],
            "generated":  [s.get("generated") or s.get("gen_answer") for s in self.data],
            "gen_docs":   [s.get("gen_docs") or s.get("retrieved") for s in self.data],
            "ref_docs":   [s.get("ref_docs") or s.get("reference_docs") for s in self.data],
        }
    
    @add_fetch_key("correct")
    def fetch_correctness(self) -> Dict[str, List[Any]]:
        return {
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "reference": [s.get("reference") or s.get("answer") for s in self.data],
            "generated":  [s.get("generated") or s.get("gen_answer") for s in self.data],
        }

    @add_fetch_key("supfact")
    def fetch_supfact(self) -> Dict[str, List[Any]]:
        """
        SupportingEMEvaluator가 요구하는 supporting_fact_* 키를 만들어서 공급.
        - 데이터에 info가 있으면 info에서 꺼내고
        - 없으면 최상단 키에서 직접 꺼내본다(둘 중 하나라도 지원)
        """
        def pick(s: dict, key: str) -> Any:
            info = s.get("info") or {}
            return info.get(key, s.get(key, ""))

        return {
            "id": [s.get("id") for s in self.data],
            "question": [s.get("question") for s in self.data],
            "supporting_fact_gen":  [pick(s, "supporting_fact_gen") for s in self.data],
            "supporting_fact_ref":  [pick(s, "supporting_fact_ref") for s in self.data],
            "supporting_fact_comp": [pick(s, "supporting_fact_comp") for s in self.data],

            # (선택) 저장 json에 reference/generated도 같이 남기고 싶으면 포함
            "reference": [s.get("reference") for s in self.data],
            "generated":  [s.get("generated") for s in self.data],
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
'''

if __name__ == "__main__":
    base_path = Path("results/inferenced/test/qag_raw_llm.json")
    params = Params(base_path)
    ref = params.get_param("reference")
    ans = params.get_param("generated")
    print(ref)
    print(ans)

