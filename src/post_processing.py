from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pathlib import Path
from langchain_ollama.chat_models import ChatOllama as OllamaLLM
import os, re, time
from utils import get_config

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


########### < (2) agentic chunking 설정 > #####################################
prompt = ChatPromptTemplate.from_template(
"""
너는 OCR 또는 자동 추출로 생성된 Markdown 문서를 정제하고,
필요할 경우 의미 단위(semantic unit) 기준으로 재구성하는 전문 편집자다.

# 입력
- raw: 전체 Markdown 텍스트

# 작업 목표
1) **원본 Markdown 구조를 최대한 보존하며**, 잘못 추출된 부분만 선택적으로 정제한다.
   - 헤더(### 등), 리스트, 코드블록, 표, 인라인 HTML(<div>, <img> 등),
     이미지 경로, 링크 형식 등 **모든 Markdown/HTML 문법 요소는 유지**한다.
   - 문장 중간 끊김, 잘못된 줄바꿈, 명백한 OCR 오타, 잘못된 띄어쓰기 등
     **명확하게 문제가 있는 부분만 최소 범위로 고친다.**
   - 의미나 표현이 자연스러우며 문제가 없는 부분은 절대 수정하지 않는다.
     (불필요한 문장 재작성·재표현·스타일 변경 금지)
   - 의미 없는 빈 줄이나 중복 문장은 제거하되,
     텍스트의 구조적 순서와 원래 섹션 흐름은 유지한다.

2) **semantic unit(의미 단위)**을 기준으로 문단 또는 섹션을 적절히 **병합하거나 분할**한다.
   - 하나의 주제 흐름을 설명하는 여러 문단이 불필요하게 분리되어 있으면 자연스럽게 **하나의 의미 단위로 병합**한다.
   - 반대로 주제가 전환되면 그 지점에서 **새로운 문단 또는 섹션으로 분리**한다.
   - 이때 Markdown 구조는 유지하되, 텍스트 단위의 재구성은 허용된다.
   - 단, **임의로 새로운 큰 제목(H1~H2)을 생성하지 않는다.**

3) **검색·RAG·임베딩 용도에 적합하도록 문서의 의미 단위(chunk)를 정돈**한다.
   - 각 의미 단위는 **독립적으로 이해 가능한 chunk**가 되도록 조정한다.
   - chunk 간 **context leakage가 최소화**되도록 자연스러운 경계를 형성한다.
   - 가능하다면 한 chunk는 **대략 300~1500 tokens(또는 그에 상응하는 길이)** 내에서 유지되도록 조정하되,
     Markdown 구조를 무리하게 깨뜨리지 않는다.

# 스타일 및 주의사항
- 텍스트 정제는 “텍스트 문단”에만 적용하고,
  이미지, 표, 코드블록, 리스트 등은 원래 위치와 구조를 유지한다.
- 사실 보완/창작 금지.
  **필요한 경우에만 정제하고, 불필요한 부분은 절대로 수정하지 않는다.**
- Markdown/HTML 블록은 절대 삭제하거나 변형하지 않는다.
  예:
    - `### 03. 흙토람에서 관비처방서 확인하기`는 그대로 유지.
    - `<div>...</div>` 이미지 박스도 그대로 유지.

# 출력 형식
- 전체 문서를 **정제된 Markdown 형태 그대로 출력**한다.
- 코드블록, JSON, 배열 형태로 감싸지 말고,
  **정상적인 Markdown 문자열만 출력한다.**

# 지금 처리할 입력
raw: {input}
"""
)

def get_propositions(text: str) -> List[str]:
    resp = chain.invoke({"input": text})
    content = (getattr(resp, "content", "") or "").strip()
    # 여기서는 JSON 파싱 대신 fallback으로 라인 스플릿(원하시면 json.loads로 바꿔도 됨)
    try:
        import json
        return json.loads(content)
    except Exception:
        fallback = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
        return fallback
######################################################################


if __name__ == "__main__":
    start = time.time()
    CFG = get_config()
    
    model_name = CFG.model_name
    llm = OllamaLLM(model=model_name, temperature=0.0, top_p=1.0, top_k=40)
    chain = prompt | llm

    md_root = Path("db/raw_db_extracted")
    md_files = list(md_root.rglob("*.md"))
    print(f"[INFO] 찾은 md 파일 개수: {len(md_files)}")

    base_output_chunks = Path("db/chunks_agentic_md_only")
    base_output_chunks.mkdir(parents=True, exist_ok=True)

    for md_file in md_files:
        print(f"\n[INFO] 처리 시작 → {md_file}")

        md_text = md_file.read_text(encoding="utf-8").strip()
        stem = md_file.stem
        out_dir = base_output_chunks / f"[md]{stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1회 agentic chunking
        chunks = get_propositions(md_text)

        for j, chunk in enumerate(chunks, 1):
            clean_chunk = chunk.strip()
            if len(clean_chunk) < 20:
                print(f"[skip] chunk_{j} (len={len(clean_chunk)})")
                continue
            out_path = out_dir / f"chunk{j}.md"
            out_path.write_text(clean_chunk, encoding="utf-8")
            print(f"[SAVE] {out_path}")

        print(f"[DONE] {md_file} 처리 완료")

    elapsed = time.time() - start
    print(f"\n총 걸린 시간: {elapsed:.2f}초")