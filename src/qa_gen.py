# qa_gen.py: LLM을 이용하여 RAG 평가에 쓸 GT QA 데이터셋 생성
import json
import time
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama as OllamaLLM
from utils import get_config

def batch_md_files(md_files, batch_size=3):
    for i in range(0, len(md_files), batch_size):
        yield md_files[i:i+batch_size]

CFG = get_config()
embedor_model_name = CFG.embedor_model_name

# ===== Embedding Wrapper =====
class Emb(Embeddings):
    def __init__(self):
        self.m = SentenceTransformer(embedor_model_name)
    def embed_documents(self, texts):
        return self.m.encode(texts).tolist()
    def embed_query(self, text):
        return self.m.encode([text])[0].tolist()

# ===== QA Prompt =====
qa_prompt = ChatPromptTemplate.from_template(
"""
다음 context에서 **RAG 평가용 Ground Truth**로 사용할 질문/답변 2~3쌍을 JSON 배열로 생성하라.

# 매우 중요한 생성 규칙 (Specificity 필수)
- context의 도메인은 매번 달라질 수 있다(농업, 장비, 보고서, 매뉴얼, 정책, 시스템 로그 등).
- 질문은 반드시 context에 등장하는 **구체적이고 식별 가능한 고유 정보(entity)** 를 포함해야 하며, 
  context에서 벗어나도 **독립적으로 의미가 완전히 명확해야 한다.**

- 질문에는 다음 중 최소 1개 이상이 반드시 포함되어야 한다:
  - 문서 또는 보고서 이름  
  - 특정 장비/시스템/작물/제품의 이름  
  - 특정 기관명 또는 작성자/소유주  
  - 특정 사건·절차·작업의 명칭  
  - 특정 날짜, 특정 장소, 특정 조건, 특정 수치  

- “이 장비”, “해당 문서”, “이 농장”, “누가”, “언제” 같은 모호한 참조 표현을 절대 사용하지 마라.  
  질문 한 문장 안에서 **무엇에 대해 말하고 있는지 정확히 특정되도록** 작성해야 한다.

- 질문은 context의 정보 일부를 **명확히 재진술하여 문맥이 연결되어야 한다.**  
  예:  
    - “2021년 3월 15일에 ○○ 기관이 수행한 △△ 장비 점검 기록에서 고장 원인은 무엇인가요?”  
    - “□□ 시스템 매뉴얼 2.0 버전에서 ‘자체 진단 절차’의 첫 단계는 무엇인가요?”  
    - “○○ 농장에서 운영 중인 ‘△△ 스마트 관비 장치’의 허용 EC 범위는 얼마인가요?”  

- 답변도 대명사 없이 **context의 고유 정보를 그대로 포함하여 명확하게** 작성한다.  
  (예: “해당 장비” X → “△△ 장비” O)

- context에 없는 정보, 일반 지식, 추론 기반 내용은 절대 생성하지 않는다.

- 질문들이 서로 같은 템플릿 구조를 반복하지 않도록, 표현 방식과 구조를 다양하게 한다.

# 예시는 형식 설명용일 뿐이며, 실제 질문은 반드시 context의 고유 엔터티와 내용만을 기반으로 생성해야 한다.
예시(형식 참고용):
- “○○ 기관이 작성한 ‘□□ 장비 안전 점검 보고서’에서 최대 허용 온도는 얼마인가요?”
- “△△ 시스템 운영 매뉴얼에서 ‘비상 정지 절차’는 어떤 순서로 시작되나요?”
- “2022년 7월 3일에 기록된 ‘○○ 재배기록지’에서 관수 시작 시간은 언제인가요?”

# 출력 형식 (중요)
아래 JSON 배열 형식만 출력하라.  
추가 문장, 설명, 주석, 자연어 텍스트는 절대 포함하지 않는다.

[
  {{"question": "질문1", "answer": "답변1"}},
  {{"question": "질문2", "answer": "답변2"}}
]

출력 시 key 이름은 반드시 "question", "answer"만 사용한다.

# context
{context}
"""
)

def gen_qas(text: str, chain) -> List[dict]:
    resp = chain.invoke({"context": text})
    try:
        arr = json.loads(resp.content)
        return arr
    except:
        return []


# ===== Main =====
if __name__ == "__main__":
    start = time.time()

    CFG = get_config()
    model_name = CFG.model_name
    llm = OllamaLLM(model=model_name, temperature=0.0)
    chain = qa_prompt | llm

    rawdb_root = Path("db/raw_db")
    qa_root = Path("qa_data/GT")
    pdf_files = list(rawdb_root.rglob("*.pdf"))

    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem        # "c"
        out_dir = qa_root / pdf_name    # qa_data/GT/c

        out_dir.mkdir(parents=True, exist_ok=True)

    # 이제 QA 생성 (md 파일 기반)
    md_root = Path("db/cleaned_md")
    md_files = list(md_root.rglob("*.md"))
    print("[INFO] md 파일:", len(md_files))

    qa_id = 1

    # PDF 경로와 MD 경로를 1:1 매핑시키고 싶으면 아래에서 batch 처리
    # 여기서는 일단 전체 md를 3개씩 묶어서 하나의 QA 파일로 만듦
    qa_file = out_dir / "GT_QA_DATA.jsonl"   # ← 저장될 파일 경로 명확히 지정
    with qa_file.open("w", encoding="utf-8") as f:
        for batch in batch_md_files(md_files, batch_size=3):
            print("[INFO] 처리 batch:", batch)

            texts = []
            for md in batch:
                texts.append(md.read_text(encoding="utf-8"))
            merged_context = "\n\n".join(texts)

            qas = gen_qas(merged_context, chain)

            for qa in qas:
                f.write(json.dumps({
                    "id": qa_id,
                    "question": qa["question"],
                    "answer": qa["answer"],
                }, ensure_ascii=False) + "\n")
                qa_id += 1

    print("[DONE] 총 소요:", time.time() - start)
    print("[DONE] 저장:", qa_file)

