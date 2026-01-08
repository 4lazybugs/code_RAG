# qa_mode/specs/hotpot.py
from typing import Dict, Any, Optional, Sequence, List
from langchain_core.prompts import ChatPromptTemplate
from .base import register_qa_mode
import re

def _format_options(options: Optional[Sequence[str]]) -> str:
    if not options:
        return ""
    return "[선택지]\n" + "\n".join(options)

hotpot_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 아래 [참고 자료]에 근거하여 질문에 답하라.

[절대 규칙]
- 근거가 없으면 추측하지 말고 answer는 "모름".
- supporting_fact_gen은 반드시 [참고 자료]의 원문에서 그대로 복사하여 1개만 제시한다.
- link_word는 문단 A/B(참고자료 내 서로 다른 문서들)에 공통으로 등장하는 연결 단어/구/고유명칭이면 최선이다.
  없으면 ""로 둔다.

[출력 형식 규칙]
- 반드시 JSON 객체 1개만 출력한다. (설명/주석/추가 텍스트 금지)
- 키는 아래 3개만 사용:
  "answer", "supporting_fact_gen", "link_word"

[정답 규칙]
- 선택지가 있는 문제(MCQ)이면 answer는 정답 번호 하나만 출력한다. (예: "1")
- 선택지가 없는 문제이면 answer는 한 문장 이내의 짧은 답으로 출력한다.

---
[참고 자료]
{reviews}

---
[질문]
{question}

[선택지(없을 수도 있음)]
{options}
""")


def _make_pretty(text: str, max_block_chars: int = 1200) -> List[str]:
    """
    infer_hpqa.py의 make_pretty를 경량화하여 재사용:
    - HTML <tr> 단위가 있으면 tr 기준으로,
    - 없으면 빈 줄 단위 문단으로 나눈 뒤,
    - 너무 긴 블록은 잘라서 요약 리스트로 만든다.
    """
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return []

    _RE_TR = re.compile(r"<tr.*?>.*?</tr>", flags=re.DOTALL | re.IGNORECASE)
    trs = _RE_TR.findall(s)
    blocks = trs if trs else re.split(r"\n\s*\n+", s)

    out: List[str] = []
    for b in blocks:
        b = "\n".join(ln.strip() for ln in b.splitlines() if ln.strip()).strip()
        if not b:
            continue
        if len(b) > max_block_chars:
            b = b[:max_block_chars] + "\n...(truncated)"
        out.append(b)
    return out

@register_qa_mode("hotpot")
class HotpotMode:
    """
    HotpotQA 스타일 멀티홉 QA 모드.
    - build_inputs: retrieved 문서 리스트나 reviews 문자열을 받아 프롬프트 인자로 변환
    - build_output: retrieved 문서 리스트를 평가/저장용 구조로 변환
    """

    prompt: ChatPromptTemplate = hotpot_prompt

    def build_inputs(self, payload: Dict[str, Any]) -> Dict[str, str]:
        # NaiveRag 등에서 payload["retrieved"]에 List[Document]가 들어온다고 가정한다.
        retrieved = payload.get("retrieved", [])

        if retrieved:
            # infer_hpqa.py에서 사용한 것과 유사하게 [DOCi] + filename 헤더를 붙여 하나의 큰 텍스트로 만든다.
            blocks = []
            for i, d in enumerate(retrieved, start=1):
                fname = d.metadata.get("filename", "")
                blocks.append(f"[DOC{i}] filename={fname}\n{d.page_content}")
            reviews = "\n\n".join(blocks)
        else:
            reviews = payload.get("reviews", "")

        return {
            "reviews": reviews,
            "question": payload["question"],
            "options": _format_options(payload.get("options")),
        }

    def build_output(self, retrieved) -> Dict[str, Any]:
        """
        retrieved: List[Document]
        infer_hpqa.py에서 평가용으로 쓰던 구조와 최대한 비슷하게,
        각 문서를 pretty string 리스트로 변환하여 저장한다.
        """
        return {
            "retrieved": [
                {
                    "rel_path": doc.metadata.get("rel_path"),
                    "rank": doc.metadata.get("__rank__"),
                    "filename": doc.metadata.get("filename"),
                    "content": _make_pretty(doc.page_content),
                }
                for doc in retrieved
            ]
        }
