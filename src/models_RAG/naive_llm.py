from .base import BaseExpert
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from load_params import get_config
from typing import Optional, Sequence

def _format_options(options: Optional[Sequence[str]]) -> str:
    if not options:
        return ""
    return "[선택지]\n" + "\n".join(options)

raw_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 질문에 답하라.

중요: 출력 형식을 반드시 지켜라.

[출력 형식 규칙]
- 선택지가 주어진 문제(MCQ)이면: 정답 번호 하나만 출력한다. 예: 1
- 선택지가 없는 문제이면: 한 문장 이내 단답만 출력한다.
- 어떤 경우에도 다음을 출력하지 마라: 선택지 내용, 괄호, "정답", "정답 번호", 콜론, 줄바꿈, 추가 설명.

[판단 규칙]
- 선택지는 아래 {options}에 있을 수도 있고, 주관식이면 없을 수도 있다.

[질문]
{question}

{options}
""")

class RawLlmExpert(BaseExpert):
    """Expert that uses raw LLM without retrieval"""
    CFG = get_config()
    model_name = CFG.model_name

    def setup(self, retriever_mode: str) -> None:
        self.model = ChatOpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            model=self.model_name,
            temperature=0.0,
        )
        # ✅ 기존 invoke 흐름 최대한 유지: prompt | model 체인만 추가
        self.chain = raw_prompt | self.model

    def handle(self, question: str, options: Optional[Sequence[str]] = None):
        try:
            options_block = _format_options(options)

            # ✅ options가 없으면 기존처럼 question만 던지는 것도 가능하지만,
            # 형식 강제를 위해 항상 prompt를 타는 게 더 안정적임
            result = self.chain.invoke({"question": question, "options": options_block})

            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            return f"[오류]: {str(e)}"
