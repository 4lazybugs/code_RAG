import re
from .base import Evaluator

def _normalize(text) -> str:
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

class SupportingEMEvaluator(Evaluator):
    """
    supporting_fact_gen이 supporting_fact_ref 또는 supporting_fact_comp에
    포함되는지(부분문자열) 체크하는 metric.

    - exact_match의 normalize/remove_spaces 컨셉을 동일하게 사용
    - 점수: 포함되면 1.0, 아니면 0.0
    """
    metric_key = "supporting_gen_contain"

    def __init__(self, remove_spaces: bool = True, case_insensitive: bool = True):
        super().__init__()
        self.remove_spaces = remove_spaces
        self.case_insensitive = case_insensitive

    def _prep(self, s: str) -> str:
        s = _normalize(s)
        if self.case_insensitive:
            s = s.lower()
        if self.remove_spaces:
            s = s.replace(" ", "")
        return s

    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs: list) -> list:
        """
        references/generated는 원칙상 answer 비교용이지만,
        여기서는 generated에 dict(info) 또는 sample dict가 들어오도록 설계해야 합니다.

        권장 방식:
        - EvalRunner에서 이 evaluator를 호출할 때 generated에 info dict를 넣어준다.
          또는
        - QAGRepository가 generated_list와 별개로 info_list를 로딩해서,
            EvalRunner가 compute_scores에 info를 넘겨준다.

        여기서는 generated 원소가 dict/info라고 가정:
        generated[i] = {"supporting_fact_gen":..., "supporting_fact_ref":..., "supporting_fact_comp":...}
        """
        scores = []

        for gen in generated:
            # gen이 str이면 정보가 없으니 0 처리
            if not isinstance(gen, dict):
                scores.append(0.0)
                continue

            sf_gen = self._prep(gen.get("supporting_fact_gen", ""))
            sf_ref = self._prep(gen.get("supporting_fact_ref", ""))
            sf_comp = self._prep(gen.get("supporting_fact_comp", ""))

            # supporting_fact_gen이 비어있으면 실패로 처리(원하면 둘 다 비면 1로 바꿀 수 있음)
            if not sf_gen:
                scores.append(0.0)
                continue

            ok = (sf_gen in sf_ref) or (sf_gen in sf_comp)
            scores.append(1.0 if ok else 0.0)

        return scores
