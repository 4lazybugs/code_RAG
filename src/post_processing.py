from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pathlib import Path
from langchain_ollama.chat_models import ChatOllama as OllamaLLM
import re, time
from utils import get_config

########### < agentic chunking 설정 > #####################################
prompt = ChatPromptTemplate.from_template(
"""
※ 주의: 입력이 짧더라도 절대 '정제 불필요'와 같은 안내 메시지를 출력하지 않는다.
입력이 무엇이든 항상 정제된 Markdown만 출력한다.
추가적인 문장, 설명, 변명, 요약, 시스템 안내 메시지를 절대 출력하지 않는다.

너는 OCR 또는 자동 추출로 생성된 Markdown 문서를 정제하고,
필요할 경우 의미 단위(semantic unit) 기준으로 재구성하는 전문 편집자다.

# 입력
- raw: 전체 Markdown 텍스트

# 작업 목표
1) **Markdown 구조를 최대한 보존한 채 최소한의 정제만 수행**한다.  
   - 헤더(### 등), 리스트, 코드블록, 표, 인라인 HTML(<div>, <img> 등),  
     이미지·링크 등 **모든 Markdown/HTML 문법 요소는 변경 없이 유지**한다.
   - 문장 끊김, 잘못된 줄바꿈, 명백한 OCR 오타·띄어쓰기만 **필요 최소 범위에서 수정**한다.
   - 의미가 자연스럽고 문제가 없는 문장은 절대 재작성하거나 표현을 바꾸지 않는다.
   - 불필요한 빈 줄·중복 문장은 제거하되, 문서의 **원래 흐름과 구조는 그대로 유지**한다.

2) 텍스트 문단에 한해 **의미 단위 기준 병합·분할**을 수행한다.  
   - 같은 주제를 설명하는 문단은 자연스럽게 병합하고,  
     주제가 바뀌는 지점에서는 문단 또는 섹션을 분리한다.
   - 구조를 해치는 과도한 재구성은 금지하며, 새로운 상위 제목(H1~H2) 생성은 하지 않는다.

3) 결과 문서는 **RAG·임베딩용으로 독립적 의미 단위(chunk)**가 되도록 정돈한다.  
   - 각 chunk는 독립적으로 이해 가능해야 하며,  
     길이는 대략 300~1500 tokens 범위를 권장하되 Markdown 구조를 무리하게 변경하지 않는다.

# 출력 형식
- 전체 문서를 **정제된 Markdown 그대로 출력**한다.
- 코드블록, JSON, 배열 등으로 감싸지 않는다.

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
    llm = OllamaLLM(
        model=model_name,
        temperature=0.0,
        top_p=1.0,
        top_k=40,
    )
    # 전역 chain 필요하면 global 로 빼도 되고, 여기서만 써도 됨
    global chain
    chain = prompt | llm

    # 원본 md들이 들어있는 루트
    md_root = Path("db/raw_db_extracted/")
    md_files = list(md_root.rglob("*.md"))
    print(f"[INFO] 찾은 md 파일 개수: {len(md_files)}")

    # 정제된 md를 저장할 루트
    output_dir = Path("db/cleaned_md")  # 폴더 이름은 취향껏
    output_dir.mkdir(parents=True, exist_ok=True)

    for md_file in md_files:
        print(f"\n[INFO] 처리 시작 → {md_file}")

        # 원본 Markdown 읽기
        md_text = md_file.read_text(encoding="utf-8").strip()

        # LLM에게 그대로 넘겨서 '전처리된 Markdown 전체' 받기
        resp = chain.invoke({"input": md_text})
        cleaned = (getattr(resp, "content", "") or "").strip()

        # 출력 경로: 원래 구조 유지하고 싶으면 상대 경로 그대로 써도 됨
        rel_path = md_file.relative_to(md_root)        # raw_db_extracted 이하 경로
        out_path = output_dir / rel_path              # 동일 구조로 저장
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(cleaned, encoding="utf-8")
        print(f"[SAVE] 정제된 md 저장 → {out_path}")

    elapsed = time.time() - start
    print(f"\n총 걸린 시간: {elapsed:.2f}초")