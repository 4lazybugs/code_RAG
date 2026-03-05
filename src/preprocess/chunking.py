from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ 추가
import time, os
from load_params import get_config
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 

from langchain_core.prompts import ChatPromptTemplate
from src.prompts.chunking import prompt
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