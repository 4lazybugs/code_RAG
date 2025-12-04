from .base import BaseExpert
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

# --------------------------------------------------

class AllExpert(BaseExpert):
    """Expert that uses all documents for RAG (accuracy-focused)"""
    
    def setup(self, retriever_mode: str) -> None:
        model = OllamaLLM(model=self.args.model_name, temperature=0.0, top_p=1.0, top_k=40)
        if retriever_mode not in self.retriever_map:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=self.retriever_map[retriever_mode],
            return_source_documents=False
        )

    def handle(self, question: str) -> str:
        try:
            # 리트리버 사용 (load_all_docs() 대신)
            res = self.qa_chain.invoke({"query": question})
            return res["result"].strip()
        except Exception as e:
            return f"[오류]: {str(e)}"

# --------------------------------------------------