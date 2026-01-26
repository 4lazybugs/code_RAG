from .base import BaseExpert
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from load_params import get_config
from typing import Optional, Sequence
from typing import Optional, Sequence, Tuple, Dict, Any, List
import json
# --------------------------------------------------

def _format_options(options: Optional[Sequence[str]]) -> str:
    """
    options가 있으면 프롬프트에 들어갈 블록 문자열로 변환.
    없으면 빈 문자열 반환.
    """
    if not options:
        return ""  # 프롬프트의 {options} 자리에 아무것도 안 나오게
    # 프롬프트에서 {options} 자리에 그대로 들어갈 텍스트
    return "[선택지]\n" + "\n".join(options)

# RAG prompts for all/partial modes
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

import re

def _pick_supporting_fact(text: str, min_chars: int = 40) -> str:
    """
    문맥이 깨지지 않도록 '너무 짧지 않은' 근거 1개를 텍스트에서 뽑는다.
    우선순위: (1) 줄 단위 (2) 문장 단위 (3) 앞부분 slice
    """
    if not text:
        return ""

    s = text.strip()

    # 1) 줄 단위
    for line in s.splitlines():
        line = line.strip()
        if len(line) >= min_chars:
            return line

    # 2) 문장 단위(간단 분리)
    sents = re.split(r"(?<=[.!?。])\s+", s)
    for sent in sents:
        sent = sent.strip()
        if len(sent) >= min_chars:
            return sent

    # 3) fallback
    return s[:max(min_chars, 120)].strip()

class HotpotExpert(BaseExpert):
    CFG = get_config()
    model_name = CFG.model_name

    def setup(self, retriever_mode: str) -> None:
        self.model = ChatOpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            model=self.model_name,
            temperature=0.0,
        )
        if retriever_mode not in self.retriever_map:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")
        self.retriever = self.retriever_map[retriever_mode]

        # ✅ Hotpot용 구조화 프롬프트 사용
        self.rag_chain = hotpot_prompt | self.model

    def handle(self, question: str, options: Optional[Sequence[str]] = None) -> Tuple[str, Dict[str, Any]]:
        try:
            snippets = self.retriever.invoke(question)
            self.retrieved_snippets = snippets  # ✅ 기존 유지

            # ✅ LLM이 "발췌"하기 쉽도록 문서 라벨링
            blocks = []
            for i, d in enumerate(snippets, start=1):
                fname = d.metadata.get("filename", "")
                blocks.append(f"[DOC{i}] filename={fname}\n{d.page_content}")
            reviews = "\n\n".join(blocks)

            options_block = _format_options(options)
            resp = self.rag_chain.invoke({
                "reviews": reviews,
                "question": question,
                "options": options_block
            })
            content = resp.content if hasattr(resp, "content") else str(resp)

            # --- (1) ref/comp supporting fact는 retrieved에서 자동 생성 ---
            ref_text = snippets[0].page_content if len(snippets) >= 1 else ""
            comp_text = snippets[1].page_content if len(snippets) >= 2 else ""
            supporting_fact_ref = _pick_supporting_fact(ref_text, min_chars=40)
            supporting_fact_comp = _pick_supporting_fact(comp_text, min_chars=40)

            # --- (2) LLM이 낸 JSON에서 answer/supporting_fact_gen/link_word 파싱 ---
            answer = ""
            supporting_fact_gen = ""
            link_word = ""

            try:
                obj = json.loads(content)
                answer = str(obj.get("answer", "")).strip()
                supporting_fact_gen = str(obj.get("supporting_fact_gen", "")).strip()
                link_word = str(obj.get("link_word", "")).strip()
            except Exception:
                # JSON 깨졌으면: content를 답으로라도 사용
                answer = content.strip()

            info = {
                "question": question,
                "answer": answer,
                "link_word": link_word,
                "supporting_fact_ref": supporting_fact_ref,
                "supporting_fact_comp": supporting_fact_comp,
                "supporting_fact_gen": supporting_fact_gen,
            }

            return answer, info

        except Exception as e:
            ans = "모름"
            info = {
                "question": question,
                "answer": ans,
                "link_word": "",
                "supporting_fact_ref": "",
                "supporting_fact_comp": "",
                "supporting_fact_gen": "",
                "error": str(e),
            }
            return ans, info
