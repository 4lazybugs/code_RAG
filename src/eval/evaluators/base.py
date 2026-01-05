from abc import ABC, abstractmethod
from typing import Type, Dict, List
from tqdm.auto import tqdm
import argparse
import yaml
import os

def load_yaml(path='src/eval/config.yaml'):
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
        return str(text).strip()

    @abstractmethod # 상속받은 자식 클래스에서 반드시 구현해야 함
    def score_once(self, data: dict) -> list:
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
