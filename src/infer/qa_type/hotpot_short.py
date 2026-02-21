# qa_mode/specs/hotpot.py
from typing import Dict, Any, Optional, Sequence, List
from langchain_core.prompts import ChatPromptTemplate
from .base import register_qa_mode
import re

def _format_options(options: Optional[Sequence[str]]) -> str:
    if not options:
        return ""
    return "[선택지]\n" + "\n".join(options)

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

hotpot_prompt = ChatPromptTemplate.from_template(
"""
당신은 multi-hop 질의응답 시스템이다.

입력:
- 문단 A
- 문단 B
- 질문

규칙:
1. 반드시 문단 A와 문단 B를 BOTH 모두 읽어야 한다.
2. 두 문단을 함께 사용해야만 답을 도출할 수 있어야 한다.
3. 답은 반드시 context 안에 명시된 "단일 값"이어야 한다.
4. 외부 지식 사용 금지.

추가 작업:
- 문단 A에서 답에 필요한 핵심 근거 문장을 원문 그대로 길게 발췌하라.
- 문단 B에서도 동일하게 원문 그대로 길게 발췌하라.
- supporting_fact는 각각 독립적으로 이해 가능할 만큼 충분히 길게 선택하라.

출력 형식:
설명 없이 JSON 배열만 출력

[
  {{
    "answer": "...",
    "supporting_fact_a": "...",
    "supporting_fact_b": "..."
  }}
]

--------------------

문단 A:
{md_a}

문단 B:
{md_b}

질문:
{question}
"""
)

@register_qa_mode("hotpot")
class Hotpot_short:
    def __init__(self, llm: Any = None):
        self.prompt = hotpot_prompt
        self.llm = llm
        self.chain = self.prompt | self.llm if self.llm else None

    def load_inputs(self, ans_input: Dict[str, Any]) -> Dict[str, str]:
        retrieved = ans_input.get("retrieved", [])
        #breakpoint()
        md_a = retrieved[0].page_content # retrieved 한 것 중 top-1
        md_b = retrieved[1].page_content # retrieved 한 것 중 top-2
        
        blocks = []
        for i, d in enumerate(retrieved, start=1):
            fname = d.metadata.get("filename", "")
            blocks.append(f"[DOC{i}] filename={fname}\n{d.page_content}")
        reviews = "\n\n".join(blocks)

        return {
            "reviews": reviews,
            "question": ans_input["question"],
            "md_a": md_a,
            "md_b": md_b
        }

    def return_result(self, agent_input: Dict[str, str]) -> str:
        result = self.chain.invoke(agent_input)
        return str(result.content) # LangChain의 AIMessage 타입에서 content 필드만 추출

    def build_output(self, retrieved) -> Dict[str, Any]:

        return {
            "retrieved": [
                {
                    #"rel_path": doc.metadata.get("rel_path"),
                    "rank": doc.metadata.get("__rank__"),
                    "filename": doc.metadata.get("filename"),
                    "content": _make_pretty(doc.page_content),
                }
                for doc in retrieved
            ]
        }
