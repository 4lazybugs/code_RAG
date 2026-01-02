import re
from .base import Evaluator

def _normalize(text) -> str:
    if text is None:
        return ""

    # LangChain 메시지(AIMessage 등) 대응
    if hasattr(text, "content"):
        text = text.content

    # dict 형태 대응
    if isinstance(text, dict):
        for k in ("content", "text", "answer", "output"):
            if k in text:
                text = text[k]
                break

    return str(text).strip()


_LEADING_CHOICE_RE = re.compile(r"^\s*(\d+)\s*(?:[).]|:)?")  # "1)", "1.", "1 :", "1" 등

def _leading_number(text: str):
    """
    문자열 선두에서 숫자(객관식 선택지 번호)를 추출합니다.
    예) "1) 20~28℃" -> "1"
        "  2. 정답"  -> "2"
        "3"         -> "3"
        "정답: 1"    -> None (선두가 아니므로)
    """
    if not text:
        return None
    m = _LEADING_CHOICE_RE.match(text)
    return m.group(1) if m else None


class ExactMatchEvaluator(Evaluator):
    metric_key = "exact_match"

    def __init__(self, remove_spaces=True, mcq=True):
        super().__init__()
        self.remove_spaces = remove_spaces
        self.mcq = mcq  # True면 "선두 숫자 있으면 객관식 모드" 자동 적용

    def _prep(self, s: str) -> str:
        s = _normalize(s)
        return s.replace(" ", "") if self.remove_spaces else s

    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs: list) -> list:
        scores = []

        for ref, gen in zip(references, generated):
            ref_norm = self._prep(ref)
            gen_norm = self._prep(gen)

            # 둘 다 비어있으면 정답 처리
            if not ref_norm and not gen_norm:
                scores.append(1.0)
                continue

            # 1) 객관식 모드(선두 숫자) 자동 판정
            if self.mcq:
                ref_choice = _leading_number(ref_norm)
                gen_choice = _leading_number(gen_norm)

                # 둘 중 하나라도 "선두 숫자"가 잡히면 객관식으로 판단(원하시면 둘 다 잡힐 때만으로 바꿔도 됩니다)
                if ref_choice is not None or gen_choice is not None:
                    # 숫자 추출 실패한 쪽은 오답 처리
                    if ref_choice is None or gen_choice is None:
                        scores.append(0.0)
                    else:
                        scores.append(1.0 if ref_choice == gen_choice else 0.0)
                    continue

            # 2) 그 외: 전체 문자열 exact match
            scores.append(1.0 if ref_norm == gen_norm else 0.0)

        return scores
