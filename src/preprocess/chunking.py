from bs4 import BeautifulSoup
from typing import Any
import json
import numpy as np
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# def html2text(html: str) -> str:
#     soup = BeautifulSoup(html, "html.parser")

#     # 이미지 태그 제거
#     for img in soup.find_all("img"):
#         img.decompose()

#     result = []

#     # 마크다운 텍스트 라인 처리 (## 헤더, 일반 텍스트)
#     for line in html.split("\n"):
#         stripped = line.strip()
#         # 순수 마크다운 라인 (HTML 태그 없는 것)
#         if stripped and "<" not in stripped:
#             result.append(stripped)

#     # 테이블 처리
#     for tr in soup.find_all("tr"):
#         cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
#         cells = [c for c in cells if c]
#         if cells:
#             result.append(" | ".join(cells))

#     return "\n".join(result)

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


def decision_batch(raw_texts: list[str], decision_chain) -> list[bool]:
    inputs = [{"raw_text": t} for t in raw_texts]
    responses = decision_chain.batch(inputs)

    results = []
    for response in responses:
        try:
            result = parse_json(response.content)
            results.append(result.get("is_useful", False))
        except Exception as e:
            print(f"[WARN] decision batch parse 실패: {e}")
            results.append(True)

    return results


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


def agentic_chunking_batch(raw_chunks: list[str], meta_chain, agentic_chain):
    meta_inputs = [{"raw_text": raw_chunk} for raw_chunk in raw_chunks]
    meta_responses = meta_chain.batch(meta_inputs)

    md_summaries = []
    for response in meta_responses:
        try:
            md_summary = parse_json(response.content).get("md_summary", "")
        except Exception as e:
            print(f"[WARN] meta parse 실패: {e}")
            md_summary = ""
        md_summaries.append(md_summary)

    agentic_inputs = [
        {
            "md_summary": md_summary,
            "raw_text": raw_chunk,
        }
        for md_summary, raw_chunk in zip(md_summaries, raw_chunks)
    ]

    agentic_responses = agentic_chain.batch(agentic_inputs)

    results = []
    for raw_chunk, md_summary, response in zip(raw_chunks, md_summaries, agentic_responses):
        try:
            chunks = parse_json(response.content)
            chunk_texts = [c["chunk"] for c in chunks if c.get("chunk")]
            results.append((md_summary, chunk_texts))
        except Exception as e:
            print(f"[WARN] agentic parse 실패: {e}")
            results.append((md_summary, [raw_chunk]))

    return results


def recursive_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """RecursiveCharacterTextSplitter 기반 baseline.
 
    문단 → 줄 → 문장 → 단어 순으로 separator를 우선 고려해 자르는,
    LangChain에서 가장 널리 쓰이는 표준 청킹 방식.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
 
 
def fixed_size_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """순수 고정 길이 baseline.
 
    separator를 전혀 고려하지 않고 문자 수 기준으로 기계적으로 자른다.
    (recursive_chunking과 구분되는 가장 단순한 baseline)
    """
    if not text:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks
 
 
def semantic_chunking(
    sentences: list[str],
    embed_model,
    breakpoint_percentile: float = 90.0,
) -> list[str]:
    """임베딩 기반 semantic chunking (LLM 미사용 baseline)
 
    인접 문장 간 임베딩 cosine similarity를 계산해,
    유사도가 급격히 떨어지는 지점(= 의미적 경계)에서 청크를 분리한다.
    """
 
    if len(sentences) <= 1:
        return sentences
 
    embeddings = np.array(embed_model.embed_documents(sentences))
 
    sims = [
        float(np.dot(embeddings[i], embeddings[i + 1]) /
              (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])))
        for i in range(len(embeddings) - 1)
    ]
    distances = [1 - s for s in sims]
    threshold = np.percentile(distances, breakpoint_percentile)
 
    chunks, current = [], [sentences[0]]
    for i, d in enumerate(distances):
        if d > threshold:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))
 
    return chunks
 
 
def split_sentences(text: str) -> list[str]:
    """간단한 문장 분리 (한국어/영어 혼용 대응)"""
    sentences = re.split(r'(?<=[.!?。])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]
 