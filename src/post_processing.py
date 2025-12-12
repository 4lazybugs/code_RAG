from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ 추가
import re, time, os
from utils import get_config
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 


########### < agentic chunking 설정 > #####################################
prompt = ChatPromptTemplate.from_template(
"""
※ 주의: 입력이 짧더라도 절대 '정제 불필요'와 같은 안내 메시지를 출력하지 않는다.
입력이 무엇이든 항상 정제된 Markdown만 출력한다.
추가적인 문장, 설명, 변명, 요약, 시스템 안내 메시지를 절대 출력하지 않는다.

너는 OCR 또는 자동 추출로 생성된 Markdown 문서를 **최소한의 정제만 수행하는 전문 편집자**이다.

# 절대 수정하지 말아야 하는 요소
- Markdown 구조(헤더, 리스트, 구분선, 인덴트 등)
- 표(Table)의 모든 구조 요소: 행/열/셀 구성, 구분선(|, ---)
- 코드블록(````), 인라인 코드(``)
- 이미지, 링크, HTML 태그(<div>, <img> 등)
- 문서의 제목 계층 구조와 전체 레이아웃

# 반드시 수행해야 하는 작업
1. 명백한 OCR 오타만 최소 범위에서 수정한다.
2. 잘못된 띄어쓰기를 자연스러운 수준에서만 고친다.
3. 중국어로 표기된 단어나 문장은 모두 대응하는 자연스러운 한국어로 교정한다.
4. 의미가 정상적인 문장은 절대 재작성하거나 스타일을 변경하지 않는다.
5. 표(Table) 내부도 텍스트 교정만 가능하며, **셀 구조/행/열/구분선은 절대 변경하지 않는다.**

# 출력 형식
- 정제된 Markdown만 그대로 출력한다.
- 코드블록, JSON, 배열, 주석 등으로 감싸지 않는다.
- 어떤 추가 문장도 출력하지 않는다.

# 입력
{input}
"""
)
######################################################################


if __name__ == "__main__":
    start = time.time()
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    CFG = get_config()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)

    # 원본 md들이 들어있는 루트
    md_root = Path("db/test/")
    md_files = list(md_root.glob("**/**/*.md"))
    print(f"[INFO] 찾은 md 파일 개수: {len(md_files)}")

    # 정제된 md를 저장할 루트
    output_dir = Path("db/post_processed_md")  # 폴더 이름은 취향껏
    output_dir.mkdir(parents=True, exist_ok=True)

    for md_file in md_files:
        print(f"\n[INFO] 처리 시작 → {md_file}")

        # 원본 Markdown 읽기
        md_text = md_file.read_text(encoding="utf-8").strip()
        message = prompt.format_messages(input=md_text)
        processed_md = llm.invoke(message).content

        # 출력 경로: 원래 구조 유지하고 싶으면 상대 경로 그대로 써도 됨
        rel_path = md_file.relative_to(md_root)        # raw_db_extracted 이하 경로
        out_path = output_dir / rel_path              # 동일 구조로 저장
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(processed_md, encoding="utf-8")
        print(f"[SAVE] 정제된 md 저장 → {out_path}")

    end = time.time()
    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n총 걸린 시간: {int(minutes)}분 {seconds:.2f}초")