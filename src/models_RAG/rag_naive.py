from .base import BaseExpert
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from load_params import get_config
from typing import Optional, Sequence
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
rag_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문가다. 자료를 근거로 질문에 답하라.

중요: 출력 형식을 반드시 지켜라.

[출력 형식 규칙]
- 선택지가 주어진 문제(MCQ)이면: 정답 번호 하나만 출력한다.
  예: 1
- 선택지가 없는 문제이면: 짧은 단답(한 문장 이내)만 출력한다.
- 어떤 경우에도 다음을 출력하지 마라: 선택지 내용, 괄호, "정답", "정답 번호", 콜론, 줄바꿈, 추가 설명.

[판단 규칙]
- 선택지는 {options}에 있을 수도 있고, 주관식 문제이면 없을 수도 있다.
- 참고 자료에 근거가 없으면 추측하지 말고 "모름"이라고만 답하라.

---
[참고 자료]
{reviews}

---
[질문]
{question}

[선택지(없을 수도 있음)]
{options}
""")

class PartialExpert(BaseExpert):
    """Expert that uses top-K snippets for RAG (speed/partial answers)"""
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
        self.rag_chain = rag_prompt | self.model
        

    def handle(self, question: str, options: Optional[Sequence[str]] = None) -> str:
        try:
            '''
            BaseRetriever
            ├─ invoke(query)                ← 사용자가 호출
            │    └─ _get_relevant_documents ← 내부에서 호출됨 (오버라이딩 대상)
            └─ _aget_relevant_documents     ← async 버전
            '''
            # ✅ invoke(rag_naive.py)호출 -> _get_relevant_documents(retriever.py;오버라이딩)호출
            snippets = self.retriever.invoke(question)
            self.retrieved_snippets = snippets # ✅ 이번 질문에서 참조한 top-k를 저장
            review = "\n\n".join(d.page_content for d in snippets)

            options_block = _format_options(options)
            result = self.rag_chain.invoke({"reviews": review, "question": question, "options": options_block})
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            return f"[오류]: {str(e)}"