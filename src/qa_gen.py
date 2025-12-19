# qa_gen.py: LLM을 이용하여 RAG 평가에 쓸 GT QA 데이터셋 생성
import json
import time, os
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils import get_config
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama  # 가장 호환성 높음

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
다음 context에서 **RAG 평가용 Ground Truth 질문/답변 1쌍**을 JSON 배열로 생성하라.

# 0) 절대 원칙
- 질문/답변은 **context의 명시 정보만** 사용한다.
- 질문은 **정확히 1개**, **단답형**으로만 생성한다.
- 질문 끝에 반드시 **“단답형으로 답하라.”** 를 포함한다.
- 답변은 수치/명칭/값 등 **짧게** 작성한다.

# 1) 모호성(ambiguity) 금지 규칙 (가장 중요)
질문은 반드시 아래 조건을 모두 만족해야 한다.

(1) 질문 대상의 "정체"가 문장 안에 포함되어야 한다.
- 대상은 **고유 식별자(정식 명칭, 모델명, 기관명, 품목명, 시스템명, 사업명, 정책명, 프로젝트명, 공정명, 약어의 풀네임 등)** 중 하나로 특정한다.
- context에 고유 식별자가 없으면, 해당 정보로는 질문을 만들지 말고 **다른 사실(수치/값/명칭)** 로 질문을 다시 만든다.

(2) 다음 표현은 질문에 절대 사용하지 않는다.
- 지시어/대명사: “이것, 그것, 해당, 이런, 저런, 위의, 아래의, 본 문서, 이 문서, 여기, 거기”
- 과도한 일반명사: “기자재, 장비, 시스템, 이미지, 표, 자료, 현황, 내용, 데이터”
  - 단, **바로 뒤에 고유 식별자 또는 정확한 명칭**이 붙어 구체화되는 경우만 예외로 허용
    (예: “관수 제어기 모델명 ABC-123”, “시설하우스 A동” 등)

(3) 질문은 context 밖에서도 단독으로 완전히 명확해야 한다.
- “무엇의/누구의/어느/어떤”이 생략되어 의미가 흔들리면 실패다.
- 질문은 항상 “무엇(고유명칭)의 어떤 속성(수치/값/명칭)?” 구조로 작성한다.

# 2) 질문 생성 절차 (필수)
- Step A: context에서 **고유 식별자(명칭/모델/기관/품목/사업/정책/프로젝트/공정 등)** 를 1개 이상 찾는다.
- Step B: 그 고유 식별자와 **직접 연결된 단일 사실(수치/값/명칭)** 을 1개 선택한다.
- Step C: 그 사실만을 묻는 **단답형 질문 1개**를 만든다.
- Step D: 아래 “자체 검증 체크”에서 하나라도 실패하면 질문을 폐기하고 다시 생성한다.

# 3) 자체 검증 체크 (통과 못 하면 재작성)
- [ ] 질문에 고유 식별자(명칭/모델/기관/품목 등)가 포함되어 있는가?
- [ ] “기자재/장비/이미지/현황/자료/내용” 같은 일반명사가 단독으로 쓰이지 않았는가?
- [ ] “해당/이것/그것/본 문서” 같은 지시어가 없는가?
- [ ] 답변이 context에 그대로 존재하는 단일 값(수치/명칭)인가?

# 4) 출력 형식 (매우 중요)
- 아래 JSON 배열 형식만 출력한다.
- 추가 설명, 주석, 자연어 문장 절대 금지.
- key는 반드시 "question", "answer"만 사용.

[
  {{"question": "단답형 질문(고유 식별자 포함, 문장 끝에 '단답형으로 답하라.' 포함)", "answer": "단답형 답변(단일 값)"}}
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
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = qa_prompt | llm
    
    # md 파일들 읽기
    md_root = Path("db/cleaned_md/manual_book/")
    md_files = list(md_root.rglob("*.md"))
    print("[INFO] md 파일:", len(md_files))

    qa_root = Path("qa_data/GT/manual_book/")
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
            # question / answer 키 없으면 skip
            if "question" not in qa or "answer" not in qa:
                continue
                    
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

        # md 파일들 읽기
    md_root = Path("db/cleaned_md/farm_consulting/")
    md_files = list(md_root.rglob("*.md"))
    print("[INFO] md 파일:", len(md_files))

    qa_root = Path("qa_data/GT/farm_consulting/")
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
            # question / answer 키 없으면 skip
            if "question" not in qa or "answer" not in qa:
                continue
                    
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