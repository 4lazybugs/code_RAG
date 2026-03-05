# qa_mode/specs/hotpot.py
from typing import Dict, Any, Optional, Sequence, List
from .base import register_qa_mode
from src.prompts.qa_type import hotpot_short_prompt
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

@register_qa_mode("hotpot")
class Hotpot_short:
    def __init__(self, llm: Any = None, prompt= hotpot_short_prompt):
        self.prompt = prompt
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
