from .base import BaseExpert
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils import get_config
# --------------------------------------------------

# RAG prompts for all/partial modes
rag_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문 상담사입니다. 아래 참고 자료를 바탕으로 질문에 답변하세요.

**답변 작성 규칙:**
1. 참고 자료의 내용을 **이해하고 재구성**하여 자연스럽고 명확하게 설명하세요
2. 답변은 **핵심 정보만 간결하게**, 불필요한 배경설명·문장 반복을 피하세요  
3. 문장은 **짧고 직관적**으로 작성하고, 장문·장황한 서술을 하지 마세요
4. 원문을 그대로 복사하거나 표/목록 형식을 그대로 옮기지 마세요
5. 문서 출처, 페이지 번호, 파일명 등은 언급하지 마세요

---
[참고 자료]
{reviews}

---
[질문]
{question}

[답변]
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
        

    def handle(self, question: str) -> str:
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
            result = self.rag_chain.invoke({"reviews": review, "question": question})
            return result
        except Exception as e:
            return f"[오류]: {str(e)}"