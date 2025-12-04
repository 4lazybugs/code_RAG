from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pathlib import Path
from langchain_ollama.chat_models import ChatOllama as OllamaLLM
import os, re, time


########## <embeddor 객체> ##################### 
''' LangChain에서 임베딩 모델을 쓰기 위해,
   "임베딩 모델"을 내부에 들고 있는 객체(embeddor) 가 필요 '''
class embeddor(Embeddings):
    def __init__(self):
        self.model =  SentenceTransformer("upskyy/bge-m3-korean") # embedding 모델 초기화
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist() # numpy array -> list
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist() # numpy array(user query) -> list
        # [0]은 user query는 하나인데 encoded vector는 2D이기 때문
################################################


########### < (1) semantic chunking> ###################################
text_splitter = SemanticChunker(
    embeddings=embeddor(),
    # percentile - 모든 문장간 차이 계산 by 상위 5%(default)
    breakpoint_threshold_type="percentile" # "standard_deviation", "interquartile_range"
)
#########################################################################


########### < (2) agentic chunking> #####################################
model_name = "qwen2.5:32b-instruct" #'exaone3.5:32b' <- 한국어 최적화 국산 모델인데 해봤는데 "쓰레기"
llm = OllamaLLM(model=model_name,temperature=0.0, top_p=1.0, top_k=40)
###### 프롬프트 ########
prompt = ChatPromptTemplate.from_template(
"""너는 OCR/스크랩 원문을 정리하고 **제목/불릿을 줄글 문단으로 변환**하는 편집자다.

# 입력
- raw: 원문 전체 텍스트

# 목표
1) 원문 클린 
2) 의미 변화 기준 문단 분할
3) 자연스러운 한국어 줄글로 다듬기
4) **오직 JSON 리스트(배열)** 만 출력 (코드블록·추가설명 금지)

# 클린 규칙
- 개행 통일(\\r\\n, \\r → \\n), HTML 코멘트 <!-- ... --> 제거
- 반복 캡션("In this image ...", "Cultivated Strawberry ...") 제거
- 수식/LaTeX 단독 행 제거(역슬래시 포함), 한글 1~2자 단독 행 제거
- 표/목차 구분선(|, -, ., ·, 공백, 숫자만 있는 줄) 제거
- ISBN 라인 제거, 공백 정리(줄끝 공백 제거, 3개↑ 개행 → 2개)

# 문단 분할 규칙
- **주제/의미 변환** 지점에서 나눈다.
- 동일 주제의 짧은 문장들은 한 문단으로 합친다.
- 표/목차/주석 등 **구조화 요소는 반드시 줄글로 변환**한다(불릿 금지).

# 줄글 변환 지침(매우 중요)
- **헤더/제목만 출력 금지**
- 반드시 **2~5문장**의 **완전한 서술형 문단**으로 작성한다.
- 각 문단은 **최소 {min_chars}자** 이상, 가능하면 **{max_chars}자 내외**로 쓴다.
- 불릿/목록은 “먼저…, 다음…, 또한…, 마지막으로 …” 같은 **연결어**로 이어서 한 문단으로 합친다.
- 표는 **핵심 헤더/범주/추세**만 요약해 문장화한다(숫자 나열 금지, 비교/대조 포함).
- **굵게/기울임/마크다운 기호, 콜론으로 끝나는 제목형 문장, 한 단어짜리 항목**을 쓰지 말 것.
- 사실 추가/창작 금지. 원문 정보만 자연스럽게 엮는다.

# 출력 형식(반드시 엄수)
- 정확히 다음 형태로만 출력: ["문단1...", "문단2...", "..."]
- 코드블록( ``` ) 금지, 추가 설명 금지. 비어 있으면 [].

# 지금 처리할 입력
raw: {input}
"""
)

min_chars, max_chars = 500, 5000 # llm이 생성할 문단별 최소/최대 문자수
chain = prompt | llm
# llm이 생성한 response에서 답변(content)만 추출하는 함수
def get_propositions(text: str) -> List[str]:
    resp = chain.invoke({"input": text, "min_chars": min_chars, "max_chars": max_chars})
    content = (getattr(resp, "content", "") or "").strip()
    fallback = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    return fallback
######################################################################


if __name__ == "__main__":
    """ 
    전체 코드 구성:
        for (각 md 파일에 대해):
            semantic chunking -> agentic chunking -> save
        end
    """

    start = time.time()

    # 1) 입력 md 파일들 재귀적으로 모두 찾기
    md_root = Path("db/raw_db_extracted/markdown")
    md_files = list(md_root.rglob("*.md"))
    print(f"[INFO] 찾은 md 파일 개수: {len(md_files)}")

    # 2) 출력 루트 디렉토리(파일별 하위 폴더가 생김)
    base_output_docs = Path("db/docs_semantic_md")
    base_output_chunks = Path("db/chunks_agentic_md")
    base_output_docs.mkdir(parents=True, exist_ok=True)
    base_output_chunks.mkdir(parents=True, exist_ok=True)

    for md_file in md_files:
        print(f"\n[INFO] 처리 시작 → {md_file}")

        # 2-1) md 내용 읽기
        with md_file.open("r", encoding="utf-8") as f:
            md_text = f.read()

        # 2-2) semantic chunking : md_text(raw_pdf) -> documents=[doc1, doc2, ...]
        documents = text_splitter.create_documents([md_text])

        # 2-3) 이 md 파일 전용 출력 디렉토리 설정
        stem = md_file.stem  # 예: manual_strawberry_high_quality
        output_dir_docs = base_output_docs / f"[md]{stem}"
        output_dir_chunks = base_output_chunks / f"[md]{stem}"
        os.makedirs(output_dir_docs, exist_ok=True)
        os.makedirs(output_dir_chunks, exist_ok=True)

        for i, doc in enumerate(documents, 1):
            doc = doc.page_content  # 실제 텍스트만 추출
            clean_doc = doc.strip()
            '''
            if len(clean_doc) < min_chars:
                # 글자 수가 너무 적으면 agentic chunking 안 하고 doc 그대로 저장
                print(f"doc_{i}.md (글자수 {len(clean_doc)} < {min_chars})는 agentic chunking 대상이 아님")
                chunk_path = os.path.join(
                    output_dir_chunks, f"chunk_same_as_doc{i}.md"
                )
                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(clean_doc)
            else:
                # get_propositions : 각 doc -> doc_split(llm이 분할한 문단)
                doc_split = get_propositions(clean_doc)

                for j, chunk in enumerate(doc_split, 1):
                    if len(chunk.strip()) < 20:
                        print(f"[skip] chunk_{j}.md (글자수 {len(chunk.strip())} < 20)")
                        continue

                    chunk_path = os.path.join(
                        output_dir_chunks, f"chunk{j}_from_doc{i}.md"
                    )
                    with open(chunk_path, "w", encoding="utf-8") as f:
                        f.write(chunk)
            '''
            # semantic chunk 결과(doc 자체) 저장
            doc_path = os.path.join(output_dir_docs, f"doc{i}.md")
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(clean_doc)

        print(f"[DONE] {md_file} 처리 완료")

    end = time.time()
    elapsed = end - start
    print(f"\n총 걸린 시간: {elapsed:.2f}초")
