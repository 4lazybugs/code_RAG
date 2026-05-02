from bs4 import BeautifulSoup
from typing import Any
import json
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def html2text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # 이미지 태그 제거
    for img in soup.find_all("img"):
        img.decompose()

    result = []

    # 마크다운 텍스트 라인 처리 (## 헤더, 일반 텍스트)
    for line in html.split("\n"):
        stripped = line.strip()
        # 순수 마크다운 라인 (HTML 태그 없는 것)
        if stripped and "<" not in stripped:
            result.append(stripped)

    # 테이블 처리
    for tr in soup.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            result.append(" | ".join(cells))

    return "\n".join(result)

def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


def decision(raw_text: str, decision_chain) -> bool:
    try:
        response = decision_chain.invoke({"raw_text": raw_text})
        result = parse_json(response.content)
        return result.get("is_useful", False)
    except Exception as e:
        print(f"[WARN] decision 실패: {e}")
        return True


def lumber_chunking(prose_pages: list[str], boundary_chain, accumulate_pages: int = 3) -> list[tuple[str, int, int]]:
    if not prose_pages:
        return []

    chunks = []
    current_chunk = [prose_pages[0]]
    accumulated = prose_pages[0]
    start_idx = 0

    for i in range(1, len(prose_pages)):
        new_text = prose_pages[i]

        if len(current_chunk) >= accumulate_pages:
            try:
                response = boundary_chain.invoke({
                    "accumulated": accumulated[-2000:],
                    "new_text": new_text,
                })
                result = parse_json(response.content)
                is_boundary = result.get("is_boundary", False)
                print(f"  페이지 {i+1}: {'[경계]' if is_boundary else '[연속]'} {result.get('reason', '')}")
            except Exception as e:
                print(f"[WARN] 단절 판단 실패: {e}")
                is_boundary = False

            if is_boundary:
                chunks.append(("\n\n".join(current_chunk), start_idx, i - 1))
                current_chunk = [new_text]
                accumulated = new_text
                start_idx = i
                continue

        current_chunk.append(new_text)
        accumulated += "\n\n" + new_text

    if current_chunk:
        chunks.append(("\n\n".join(current_chunk), start_idx, len(prose_pages) - 1))

    return chunks


def agentic_chunking(raw_chunk: str, meta_chain, agentic_chain) -> tuple[str, list[str]]:
    try:
        # 1) 요약 추출
        meta_response = meta_chain.invoke({"raw_text": raw_chunk})
        md_summary = parse_json(meta_response.content).get("md_summary", "")

        # 2) 요약 기반 청크 분리
        agentic_response = agentic_chain.invoke({
            "md_summary": md_summary,
            "raw_text":   raw_chunk,
        })
        chunks = parse_json(agentic_response.content)
        chunk_texts = [c["chunk"] for c in chunks if c.get("chunk")]

        return md_summary, chunk_texts

    except Exception as e:
        print(f"[WARN] agentic_chunking 실패: {e}")
        return "", [raw_chunk]