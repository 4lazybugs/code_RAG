from bs4 import BeautifulSoup
from typing import Any
import json
import numpy as np
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


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


def page_merging(prose_pages: list[str], boundary_chain, accumulate_pages: int = 3) -> list[tuple[str, int, int]]:
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


def recursive_chunking(text: str, chunk_size: int = 128, overlap: int = 50) -> list[str]:
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
 


def fixed_size_chunking(text: str, max_tokens: int = 128, overlap: int = 10) -> list[str]:
    """순수 고정 길이 baseline (토큰=단어 단위).
 
    separator를 전혀 고려하지 않고 단어 수 기준으로 기계적으로 자른다.
    (recursive_chunking과 구분되는 가장 단순한 baseline)
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    step = max(max_tokens - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i:i + max_tokens]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + max_tokens >= len(words):
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


def lumber_chunking(
    paragraphs: list[str],
    lumber_chain,
    max_tokens: int = 64,
) -> list[tuple[str, int, int]]:
    """LumberChunker (Duarte et al., 2024, EMNLP Findings) 변형 청킹.
 
    원 논문은 토큰 수 θ를 "최소 누적치"로 써서, θ를 넘긴 그룹을 만든 뒤
    그 안에서 LLM이 단절 지점을 찾는다. 이 방식은 LLM이 단절을 못 찾으면
    그룹 전체(= θ를 이미 초과한 크기)가 그대로 청크가 되어버려서,
    "모든 청크가 max_tokens 이하"라는 하드 캡을 보장하지 못한다.
 
    여기서는 그 문제를 없애기 위해 순서를 바꿨다:
      1) 다음 문단을 더하면 max_tokens를 넘기게 되는 바로 그 시점에
         누적을 멈춘다. -> 이 시점의 그룹은 이미 max_tokens 이하로 보장됨.
      2) 그 그룹(이미 cap 이하)을 lumber_chain(LLM)에 "ID: 텍스트" 형태로
         넘겨, 의미 단절이 있으면 그 지점의 ID를 반환받는다.
      3) 반환된 ID까지만 하나의 청크로 확정한다. LLM이 단절을 못 찾으면
         boundary_id는 그룹의 마지막 ID를 그대로 가리키므로, 이 경우에도
         청크 크기는 "1)에서 이미 cap 이하로 보장된 그룹"과 같아진다.
         즉 LLM 응답이 무엇이든 결과 청크는 절대 max_tokens를 넘지 않는다
         (단, 문단 하나 자체가 max_tokens보다 큰 예외 케이스는 아래 참고).
 
    Args:
        paragraphs: 문단/페이지 단위로 이미 분리된 텍스트 리스트.
        lumber_chain: {"numbered_paragraphs": str} -> JSON
            {"boundary_id": int, "reason": str} 를 반환하는 체인.
            (prompts.py의 lumber_prompt | llm 형태로 만들면 됨)
            boundary_id는 그룹 내 상대 인덱스(0-based) 기준.
        max_tokens: 청크 하나가 절대 넘을 수 없는 최대 토큰 수 (하드 캡).
 
    Returns:
        (chunk_text, start_idx, end_idx) 튜플 리스트.
        start_idx/end_idx는 원본 paragraphs 리스트 기준 절대 인덱스.
 
    Note:
        문단 하나만으로도 max_tokens를 넘는 경우(예: 표가 통째로 한 문단인
        경우)는 더 쪼갤 문단이 없으므로 그 문단 단독으로 청크가 되고,
        이때는 cap을 넘을 수 있다. 이런 케이스는 함수 밖에서
        recursive_chunking 등으로 후처리하는 것을 권장한다.
    """
    if not paragraphs:
        return []
 
    def _count_tokens(text: str) -> int:
        # 간단한 근사치: 공백 기준 단어 수.
        # 정확도가 필요하면 tiktoken 등으로 교체 가능.
        return len(text.split())
 
    chunks: list[tuple[str, int, int]] = []
    start_idx = 0
    n = len(paragraphs)
 
    while start_idx < n:
        group_ids = [start_idx]
        group_token_count = _count_tokens(paragraphs[start_idx])
        i = start_idx + 1
 
        # max_tokens를 "넘기기 전"까지만 문단을 누적 (넘기게 될 문단은 포함 X)
        while i < n:
            next_tokens = _count_tokens(paragraphs[i])
            if group_token_count + next_tokens > max_tokens:
                break
            group_ids.append(i)
            group_token_count += next_tokens
            i += 1
 
        # 문단이 하나뿐이면 더 쪼갤 게 없으므로 그대로 확정
        if len(group_ids) == 1:
            chunks.append((paragraphs[start_idx], start_idx, start_idx))
            start_idx += 1
            continue
 
        # 그룹을 "ID {상대인덱스}: {텍스트}" 형태로 LLM에 전달
        numbered_group = "\n\n".join(
            f"ID {pid - start_idx}: {paragraphs[pid]}" for pid in group_ids
        )
 
        try:
            response = lumber_chain.invoke({"numbered_paragraphs": numbered_group})
            result = parse_json(response.content)
            boundary_offset = int(result.get("boundary_id", len(group_ids) - 1))
            boundary_offset = max(0, min(boundary_offset, len(group_ids) - 1))
            print(
                f"  그룹 [{start_idx}:{start_idx + len(group_ids) - 1}] "
                f"({group_token_count} tok) -> 분할 ID {boundary_offset} "
                f"({result.get('reason', '')})"
            )
        except Exception as e:
            print(f"[WARN] lumber_chunking 경계 판단 실패: {e}")
            boundary_offset = len(group_ids) - 1  # 실패 시 그룹 끝까지를 청크로
 
        end_idx = start_idx + boundary_offset
        chunks.append(("\n\n".join(paragraphs[start_idx:end_idx + 1]), start_idx, end_idx))
        start_idx = end_idx + 1
 
    return chunks
 

def lumber_chunking_from_text(
    text: str,
    lumber_chain,
    max_tokens: int = 128,
    paragraph_sep: str = "\n\n",
) -> list[str]:
    """lumber_chunking을 baseline 인터페이스(text -> list[str])에 맞춘 버전.

    recursive_chunking / fixed_size_chunking과 동일하게 원본 텍스트 하나를
    받아서, 내부에서 문단 분리 -> lumber_chunking -> 텍스트만 추출까지
    한 번에 처리한다. start_idx/end_idx가 필요 없는 baseline 비교용.
    """
    paragraphs = [p.strip() for p in text.split(paragraph_sep) if p.strip()]
    results = lumber_chunking(paragraphs, lumber_chain, max_tokens=max_tokens)
    return [chunk_text for chunk_text, _, _ in results]


def lumber_chunking_batch(
    paragraph_groups: list[list[str]],
    lumber_chain,
    max_tokens: int = 550,
) -> list[list[tuple[str, int, int]]]:
    """여러 문서(paragraphs 리스트)에 대해 lumber_chunking을 순차 적용.
 
    그룹 시작점이 이전 그룹의 분할 결과에 따라 달라지는 순차 의존적
    알고리즘이라, agentic_chunking_batch처럼 한 번에 batch invoke로
    병렬화할 수 없다. 문서 간에는 독립적이므로 문서 단위로만 순회한다.
    """
    return [
        lumber_chunking(paragraphs, lumber_chain, max_tokens)
        for paragraphs in paragraph_groups
    ]
 


 
def split_sentences(text: str) -> list[str]:
    """간단한 문장 분리 (한국어/영어 혼용 대응)"""
    sentences = re.split(r'(?<=[.!?。])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]
 