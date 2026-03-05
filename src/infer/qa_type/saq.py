from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from .base import register_qa_mode


saq_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 자료를 근거로 질문에 답하라.

중요: 출력 형식을 반드시 지켜라.

[출력 형식 규칙]
- 한 문장 이내의 짧은 단답만 출력한다.
- 어떤 경우에도 다음을 출력하지 마라:
  줄바꿈, 추가 설명, 근거 인용, "정답", "정답:", 콜론.

---
[참고 자료]
{reviews}

---
[질문]
{question}
""")
    

@register_qa_mode("saq")
class SAQ:
    """
    Short Answer Question (주관식, 선택지 없음)
    """
    def __init__(self, llm: Any = None, prompt= saq_prompt):
        self.prompt = prompt
        self.llm = llm
        self.chain = self.prompt | self.llm if self.llm else None

    def load_inputs(self, payload: Dict[str, Any]) -> Dict[str, str]:
        # retrieved는 NaiveRag에서 normalize_docs로 표준화된 List[Document]
        retrieved = payload.get("retrieved", [])
        reviews = "\n\n".join(
            f"[{doc.metadata.get('__rank__')}] {doc.metadata.get('rel_path')}\n{doc.page_content}"
            for doc in retrieved
        ) if retrieved else payload.get("reviews", "")

        return {
            "question": payload["question"],
            "reviews": reviews,
        }

    def return_result(self, agent_input: Dict[str, str]) -> str:
        result = self.chain.invoke(agent_input)
        return str(result.content) # LangChain의 AIMessage 타입에서 content 필드만 추출

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
