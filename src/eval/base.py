from abc import ABC, abstractmethod
from typing import Type, Dict, List, Any, Callable
from tqdm.auto import tqdm
import argparse
import yaml
import os
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


def load_yaml(path='configs/config_eval.yaml'):
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # 환경 변수 치환 처리
    config = {}
    for k, v in raw_config.items():
        if isinstance(v, str):
            config[k] = os.path.expandvars(v)
        else:
            config[k] = v

    return config

def get_config():
    default_cfg = load_yaml()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sbert_model_name", type=str, default=default_cfg.get('sbert_model_name'))
    parser.add_argument("--bert_model_name", type=str, default=default_cfg.get('bert_model_name'))

    args = parser.parse_args()
    return args


metric_dict: Dict[str, Type["Evaluator"]] = {}

def add_metric_key(key: str):
    """
    사용:
      @add_metric_key("bert")
      class BertEvaluator(Evaluator): ...
    """
    def deco(cls: Type["Evaluator"]) -> Type["Evaluator"]:
        if key in metric_dict:
            raise KeyError(f"Duplicate metric_key detected: {key}")

        # 선택: 클래스에도 metric_key 박아두고 싶으면(추천)
        cls.metric_key = key  # type: ignore[attr-defined]

        metric_dict[key] = cls
        return cls

    return deco


class Evaluator(ABC):
    """
    오프라인 평가용 베이스:
    - Runner/Repository가 references/generate/gen_docs/ref_docs를 이미 제공한다는 가정
    - 여기서는 compute_scores()만 강제한다
    """

    @staticmethod # 객체(self)도 클래스(cls)도 필요 없는 함수
    def _normalize(text) -> str:
        """하위 evaluator에서 공통으로 쓸 정규화 유틸(선택)"""
        if text is None:
            return ""
        if hasattr(text, "content"):
            text = text.content
        if isinstance(text, dict):
            for k in ("content", "text", "answer", "output"):
                if k in text:
                    text = text[k]
                    break
        if isinstance(text, list):
            # content 배열을 문자열로 변환
            text = "\n".join(str(item) for item in text if item)
        return str(text).strip()
    
    @staticmethod
    def _get_field(data: dict, *keys) -> Any:
        """여러 가능한 키 중 첫 번째로 존재하는 값을 반환"""
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return None

    @abstractmethod # 상속받은 자식 클래스에서 반드시 구현해야 함
    def score_once(self, data: dict) -> float:
        """
        단일 샘플 평가
        """
        pass

    def score_all(self, batch_data, *, show_progress=True, desc=None):
        it = batch_data
        if show_progress:
            it = tqdm(
                batch_data,
                desc=getattr(self, "metric_key", "scoring"),
                bar_format="{desc:<10} | {bar:40} | {n}/{total} | {elapsed}s",
                ascii=True,
                colour="cyan",
                leave=True,   # 줄 남김
            )
        return [self.score_once(x) for x in it]
