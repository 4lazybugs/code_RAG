from typing import Dict, Any, Optional, Sequence
from langchain_core.prompts import ChatPromptTemplate
from .base import register_qa_mode

def _format_options(options: Optional[Sequence[str]]) -> str:
    if not options:
        return ""
    return "[선택지]\n" + "\n".join(options)

mcq_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 자료를 근거로 질문에 답하라.

중요: 출력 형식을 반드시 지켜라.

[출력 형식 규칙]
- 정답 번호 하나만 출력한다.
  예: 1
- 선택지가 없는 문제이면: 짧은 단답(한 문장 이내)만 출력한다.
- 어떤 경우에도 다음을 출력하지 마라: 선택지 내용, 괄호, "정답", "정답 번호", 콜론, 줄바꿈, 추가 설명.

[판단 규칙]
- 선택지는 {options}에 있을 수도 있다.

---
[참고 자료]
{reviews}

---
[질문]
{question}

[선택지]
{options}
""")

@register_qa_mode("mcq")
class MCQ:
    prompt: ChatPromptTemplate = mcq_prompt

    def build_inputs(self, payload: Dict[str, Any]) -> Dict[str, str]:
        # retrieved는 NaiveRag에서 normalize_docs로 표준화된 List[Document]
        retrieved = payload.get("retrieved", [])
        reviews = "\n\n".join(
            f"[{doc.metadata.get('__rank__')}] {doc.metadata.get('rel_path')}\n{doc.page_content}"
            for doc in retrieved
        ) if retrieved else payload.get("reviews", "")

        return {
            "question": payload["question"],
            "reviews": reviews,
            "options": _format_options(payload.get("options")),
        }

    def build_output(self, retrieved) -> Dict[str, Any]:
        # retrieved는 표준화된 List[Document]
        return {
            "retrieved": [
                {
                    "rel_path": doc.metadata.get("rel_path"),
                    "rank": doc.metadata.get("__rank__"),
                    "content": [line.strip() for line in doc.page_content.split("\n") if line.strip()],
                }
                for doc in retrieved
            ]
        }
