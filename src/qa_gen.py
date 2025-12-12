# qa_gen.py: LLM을 이용하여 RAG 평가에 쓸 GT QA 데이터셋 생성
import json
import time, os
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ 추가
from utils import get_config
from dotenv import load_dotenv

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
다음 context에서 **RAG 평가용 Ground Truth 질문/답변 2쌍**을 JSON 배열로 생성하라.

# 핵심 출력 규칙 (반드시 준수)
- 질문은 정확히 **2개만 생성**한다.
  - **질문 1:** 단답형 질문  
    → 질문 문장 끝에 반드시 **“단답형으로 답하라.”** 를 명시한다.  
    → 답변은 수치, 명칭, 값 등 **짧은 형태로만** 작성한다.
  - **질문 2:** 서술형 질문  
    → 절차, 이유, 과정, 결과 등을 **문장 형태로 설명하도록** 질문한다.

# 질문 생성 방향성 (중요)
- 질문은 형식적 메타정보(파일 경로, URL, 페이지 번호, 사이트 위치 등)를 묻지 않는다.
- 질문은 반드시 **context의 실제 내용 자체(수치, 명칭, 절차, 조건, 결과, 판단 근거 등)** 를 직접적으로 묻는다.
- 질문은 context를 벗어나더라도 **단독으로 의미가 완전히 명확해야 한다.**
- “이것”, “해당 내용”, “그 장비”, “이 문서” 같은 **모호한 지시어는 절대 사용하지 않는다.**
- 질문 문장 안에서 대상이 되는 **문서, 시스템, 장비, 작물, 기관, 사건, 절차 등은 반드시 고유 명칭으로 특정**한다.

# 질문 작성 원칙 (육하원칙 기반)
- 질문은 누가·언제·어디서·무엇을·어떻게·왜 요소를 가능한 한 많이 포함하되, 문맥상 자연스럽게 구성한다.
- 질문은 context의 핵심 정보를 **재진술하는 방식**으로 작성하여 문맥 연결이 명확해야 한다.
- 두 질문은 **서로 다른 정보 유형**을 다룬다.
  - 예: 단답형은 수치·값·명칭
  - 서술형은 절차·원인·결과·의사결정 근거

# 답변 작성 규칙
- 답변은 반드시 **context에 존재하는 정보만 사용**한다.
- 추론, 일반 지식, 상식 보완, 해석 확장은 절대 금지한다.
- 답변에도 대명사를 사용하지 말고, **고유 명칭과 수치를 그대로 포함**하여 명확하게 작성한다.

# 출력 형식 (매우 중요)
- 아래 JSON 배열 형식만 출력한다.
- 추가 설명, 주석, 자연어 문장은 절대 포함하지 않는다.
- key 이름은 반드시 "question", "answer"만 사용한다.

[
  {{"question": "단답형 질문 (반드시 '단답형으로 답하라.' 포함)", "answer": "단답형 답변"}},
  {{"question": "서술형 질문", "answer": "서술형 답변"}}
]

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
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    CFG = get_config()

    CFG = get_config()
    model_name = CFG.model_name
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)
    chain = qa_prompt | llm

    # md 파일들 읽기
    md_root = Path("db/post_processed_md/")
    md_files = list(md_root.rglob("*.md"))
    print("[INFO] md 파일:", len(md_files))

    qa_root = Path("qa_data/GT/")
    qa_root.mkdir(parents=True, exist_ok=True)

    qa_id = 1  # 전체 QA에 대해 유니크 id 부여 (원하면 pdf별로 리셋해도 됨)

    for md_path in md_files:
        # 1) pdf 이름: 상위 폴더 이름을 pdf 폴더로 사용한다고 가정
        #    예: db/raw_db_extracted/dfdf.pdf/a.md -> pdf_name = "dfdf.pdf"
        pdf_name = md_path.parent.name

        # 2) chunk 이름: 파일 이름에서 .md 제거
        #    예: a.md -> a
        chunk_name = md_path.stem

        # 3) GT 저장 위치: qa_data/GT/dfdf.pdf/a.json
        pdf_out_dir = qa_root / pdf_name
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        qa_file = pdf_out_dir / f"{chunk_name}.json"

        # 이 md 파일만 context로 사용
        text = md_path.read_text(encoding="utf-8")
        raw_qas = gen_qas(text, chain)

        # id 포함해서 JSON 배열 형태로 저장
        qas_with_id = []
        for qa in raw_qas:
            qas_with_id.append({
                "id": qa_id,
                "question": qa["question"],
                "answer": qa["answer"],
            })
            qa_id += 1

        # a.json, b.json 안에는 하나의 JSON 배열로 저장
        with qa_file.open("w", encoding="utf-8") as f:
            json.dump(qas_with_id, f, ensure_ascii=False, indent=2)

        print(f"[DONE] 저장: {qa_file}")

    end = time.time()
    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n총 걸린 시간: {int(minutes)}분 {seconds:.2f}초")