from pathlib import Path
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
import json

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    
    rows = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all("td")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    
    return "\n".join(rows)


def llm_chunking(md_file: Path) -> list[dict]:
    content = md_file.read_text(encoding="utf-8")
    lines = content.split("\n")

    # 헤더 추출
    header = next((l.lstrip("#").strip() for l in lines if l.startswith("#")), "")
    if not header:
        soup = BeautifulSoup(content, "html.parser")
        div = soup.find("div", style=lambda s: s and "text-align" in s)
        if div:
            header = div.get_text(strip=True)

    raw_text = extract_text_from_html(content)

    response = llm.invoke(f"""
    아래는 농업 컨설팅 보고서입니다. 문서 내용을 읽고 의미 단위로 자유롭게 청크를 나눠주세요.
    카테고리와 세부항목은 문서 내용에 맞게 스스로 판단하여 이름을 붙여주세요.

    [context 작성 규칙]
    - 각 청크를 독립적으로 읽어도 이해할 수 있도록 문서 전체의 핵심 배경 정보를 요약

    문서 제목: {header}

    내용:
    {raw_text}

    반드시 아래 JSON 스키마를 지켜서 답하세요. 다른 말은 하지 마세요:
    {{
        "context": "핵심 배경 정보 요약",
        "chunks": [
            {{
                "category": "직접 판단한 카테고리명",
                "sub_category": "직접 판단한 세부항목명",
                "text": "해당 청크의 내용"
            }}
        ]
    }}
    """)

    try:
        result = json.loads(response.content)
        context = result["context"]
        chunks = []
        for chunk in result["chunks"]:
            chunks.append({
                "source_file": md_file.name,
                "title": header,
                "context": context,
                "category": chunk["category"],
                "sub_category": chunk["sub_category"],
                "text": chunk["text"],
            })
        return chunks

    except json.JSONDecodeError:
        print(f"[WARN] JSON 파싱 실패: {md_file.name}")
        return []

