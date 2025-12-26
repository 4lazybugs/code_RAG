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

# ===== QA Prompt ===== : short answer question
saq_prompt = ChatPromptTemplate.from_template(
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

# ===== QA Prompt ===== : multiple choice question
mcq_prompt = ChatPromptTemplate.from_template(
"""
다음 context에서 **RAG 평가용 Ground Truth 객관식(MCQ) 질문/선택지/정답 1쌍**을 JSON 배열로 생성하라.

# 0) 절대 원칙
- 질문/선택지/정답은 **context의 명시 정보만** 사용한다.
- 질문은 **정확히 1개**만 생성한다.
- 질문은 **context 밖에서도 단독으로 완전히 명확**해야 한다.
- 정답은 **선택지(1~5) 중 정확히 하나**여야 한다.
- 출력은 **반드시 JSON 배열 1개 원소**만 포함한다.

# 1) 모호성(ambiguity) 금지 규칙 (가장 중요)
질문은 반드시 아래 조건을 모두 만족해야 한다.

(1) 질문 대상의 정체가 문장 안에 포함되어야 한다.
- 대상은 **고유 식별자(정식 명칭, 모델명, 기관명, 품목명, 시스템명, 사업명, 정책명, 프로젝트명, 공정명, 약어의 풀네임 등)** 로 특정한다.
- context에 고유 식별자가 없으면 해당 정보로는 질문을 만들지 말고 **다른 사실(수치/값/명칭)** 로 질문을 다시 만든다.

(2) 아래 표현은 질문에 절대 사용하지 않는다.
- 지시어/대명사: “이것, 그것, 해당, 이런, 저런, 위의, 아래의, 본 문서, 이 문서, 여기, 거기”
- 과도한 일반명사: “기자재, 장비, 시스템, 이미지, 표, 자료, 현황, 내용, 데이터”
  - 단, **바로 뒤에 고유 식별자/정확한 명칭이 즉시 붙어 구체화되는 경우만 예외로 허용**

(3) 질문 형식
- 질문은 항상 다음 구조로 작성한다:
  **“[고유명칭]의 [속성(수치/값/명칭)]은 무엇인가?”**

# 2) 선택지(options) 생성 규칙 (형식 강제)
- options는 **문자열 배열**로 생성한다.
- options 길이는 **반드시 5**여야 한다.
- 각 선택지는 **반드시 아래 형식**을 따른다(띄어쓰기 포함):
  - "1) <텍스트>"
  - "2) <텍스트>"
  - "3) <텍스트>"
  - "4) <텍스트>"
  - "5) <텍스트>"
- `<텍스트>` 부분에는 **앞번호(예: 1), ①, (1), 1. 등)를 다시 포함하지 않는다.**  
  (즉, "1) 1) ..." 같은 중복 금지)
- 정답 1개 + 오답 4개로 구성한다.
- 오답은 **context에 등장하지 않지만, 형태상 그럴듯한 값**이어야 한다.
- 선택지에는 정답 힌트(“정답”, “맞음”, “오답” 등)를 포함하지 않는다.

# 3) 정답(answer) 규칙 (중요)
- answer는 **문자열(string)** 로 출력한다.
- answer 값은 반드시 아래 중 하나여야 한다:
  - "1" 또는 "2" 또는 "3" 또는 "4" 또는 "5"
- answer는 **정답 선택지의 번호**를 의미한다.
- answer가 "k"이면 정답 선택지는 options[k-1]이다.

# 4) 생성 절차 (필수)
- Step A: context에서 고유 식별자 1개 이상을 찾는다.
- Step B: 그 고유 식별자와 직접 연결된 단일 사실(수치/값/명칭) 1개를 고른다.
- Step C: 그 사실만을 묻는 질문 1개를 만든다.
- Step D: 정답 1개와 오답 4개로 options(총 5개)를 만든다.
- Step E: 정답에 해당하는 번호를 answer("1"~"5")로 지정한다.

# 5) 자체 검증 체크 (하나라도 실패하면 폐기하고 다시 생성)
- [ ] 질문에 고유 식별자가 포함되어 있는가?
- [ ] 금지 지시어(이것/해당/본 문서 등)가 없는가?
- [ ] options가 정확히 5개인가?
- [ ] options의 각 항목이 정확히 "n) <텍스트>" 형식인가? (n=1..5)
- [ ] options 텍스트에 추가 번호(예: "1) ① (1) 1.")가 중복 포함되어 있지 않은가?
- [ ] options 중 **정확히 1개만** context와 일치하는가?
- [ ] answer가 "1"~"5" 중 하나인가?
- [ ] answer 번호가 가리키는 선택지 텍스트가 context의 명시 정보와 일치하는가?

# 6) 출력 형식 (매우 중요)
- 아래 JSON 배열 형식만 출력한다.
- 추가 설명/주석/자연어 문장 절대 금지.
- key는 반드시 "question", "options", "answer"만 사용한다.

[
  {{
    "question": "질문 문장 (고유 식별자 포함)",
    "options": ["1) <선택지>", "2) <선택지>", "3) <선택지>", "4) <선택지>", "5) <선택지>"],
    "answer": "정답 번호(문자열, 1~5)"
  }}
]

# context
{context}
"""
)

def gen_GT(md_root: Path, qa_root: Path, chain):
    md_root = Path(md_root)
    md_files = list(md_root.rglob("*.md"))
    print(f"[INFO] md 파일: {len(md_files)} ({md_root})")

    qa_root = Path(qa_root)
    qa_root.mkdir(parents=True, exist_ok=True)

    qa_id = 1  # 전체 QA에 대해 유니크 id

    for md_path in md_files:
        # 1) chunk 이름
        pdf_name = md_path.parent.name
        
        # 2) chunk 이름
        chunk_name = md_path.stem

        # 3) 저장 위치
        pdf_out_dir = qa_root / pdf_name
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        qa_file = pdf_out_dir / f"{chunk_name}.json"

        # md 내용 → QA 생성
        text = md_path.read_text(encoding="utf-8")
        raw_qas = gen_qas(text, chain)

        qas_with_id = []
        for qa in raw_qas:
            if "question" not in qa or "answer" not in qa:
                continue

            qas_with_id.append({
                "id": qa_id,
                "question": qa["question"],
                "answer": qa["answer"],
                "reference_docs": [md_path.name],
                "options": qa["options"] if "options" in qa else None,
            })
            qa_id += 1

        with qa_file.open("w", encoding="utf-8") as f:
            json.dump(qas_with_id, f, ensure_ascii=False, indent=2)

        print(f"[DONE] 저장: {qa_file}")


def gen_qas(text: str, chain) -> List[dict]:
    resp = chain.invoke({"context": text})
    try:
        arr = json.loads(resp.content)
        return arr
    except:
        return []


def merge_gt_by_id(gt_root: Path, out_path: Path):
    gt_root = Path(gt_root)
    out_path = Path(out_path)

    merged = []
    for fp in sorted(gt_root.rglob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            merged.extend(json.load(f))   # gen_GT가 list로 저장한다는 전제

    merged.sort(key=lambda x: x["id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 저장됨: {out_path}")



# ===== Main =====
if __name__ == "__main__":
    start = time.time()
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = mcq_prompt | llm # multiple choice question
    #chain = saq_prompt | llm # short answer question
    
    
    # manual_book
    gen_GT(
        md_root=Path("db/cleaned_md/manual_book/"),
        qa_root=Path("qa_data/GT/manual_book/mcq"),
        chain=chain
    )

    merge_gt_by_id(
        gt_root=Path("qa_data/GT/manual_book"),
        out_path=Path("qa_data/GT/manual_book/mcq/gt_merged_manual_book.json")
    )

    '''
    # farm_consulting
    gen_GT(
        md_root=Path("db/cleaned_md/farm_consulting/"),
        qa_root=Path("qa_data/GT/mcq/farm_consulting"),
        chain=chain
    )
    
    merge_gt_by_id(
        gt_root=Path("qa_data/GT/farm_consulting"),
        out_path=Path("qa_data/GT/farm_consulting/mcq/gt_merged_farm_consulting.json")
    )

    
    # test
    gen_GT(
        md_root=Path("db/cleaned_md/test/"),
        qa_root=Path("qa_data/test"),
        chain=chain
    )

    merge_gt_by_id(
        gt_root=Path("qa_data/test"),
        out_path=Path("qa_data/test/gt_merged_test.json")
    )
    '''

    end = time.time()
    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n총 걸린 시간: {int(minutes)}분 {seconds:.2f}초")