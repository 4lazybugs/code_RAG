from pathlib import Path
from bs4 import BeautifulSoup
from typing import Any
import json

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


def llm_chunking(md_file: Path, chain: Any, meta_chain: Any, decision_chain: Any) -> list[dict]:
    content = md_file.read_text(encoding="utf-8")
    lines = content.split("\n")

    header = next((l.lstrip("#").strip() for l in lines if l.startswith("#")), "")
    if not header:
        soup = BeautifulSoup(content, "html.parser")
        div = soup.find("div", style=lambda s: s and "text-align" in s)
        if div:
            header = div.get_text(strip=True)

    raw_text = html2text(content)

    # 0단계: 유효성 판단
    try:
        decision_response = decision_chain.invoke({"raw_text": raw_text})
        decision = json.loads(decision_response.content)
        if not decision["is_useful"]:
            print(f"  → [SKIP] {decision['reason']}")
            return []
        print(f"  → [PASS] {decision['reason']}")
    except Exception as e:
        print(f"[WARN] 유효성 판단 실패: {md_file} / {e}")

    # 1단계: 메타 추출
    try:
        meta_response = meta_chain.invoke({"raw_text": raw_text})
        meta = json.loads(meta_response.content)
        context_prefix = meta["context_prefix"]
        print(f"  → context_prefix: {context_prefix}")
    except Exception as e:
        print(f"[WARN] 메타 추출 실패: {md_file} / {e}")
        context_prefix = header

    # 2단계: 청킹
    response = chain.invoke({
        "header": header,
        "raw_text": raw_text,
        "context_prefix": context_prefix,
    })

    try:
        result = json.loads(response.content)
        md_summary = result["md_summary"]

        chunks = []
        for i, chunk in enumerate(result["chunks"]):
            chunks.append({
                "id": i,
                "source_file": md_file.name,
                "source_path": str(md_file),
                "md_summary": md_summary,
                "raw_chunk": chunk["text"],
                "agentic_chunk": chunk["agentic_chunk"],
            })
        return chunks

    except Exception as e:
        print(f"[WARN] JSON 파싱 실패: {md_file} / {e}")
        print(response.content[:500])
        return []


def collect_md_files(path: Path) -> list[Path]:
    path = Path(path)

    if not path.exists():
        print(f"[WARN] 경로가 존재하지 않습니다: {path}")
        return []

    if path.is_file():
        if path.suffix.lower() != ".md":
            print(f"[WARN] md 파일이 아닙니다: {path}")
            return []
        return [path]

    return sorted(path.rglob("*.md"))


def chunking_and_save(
    chain: Any,
    meta_chain: Any,
    decision_chain: Any,
    md_dirs: list[Path],
    output_dir: Path,
    max_files: int | None = None
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    processed_files = 0

    for md_dir in md_dirs:
        md_dir = Path(md_dir)
        md_files = collect_md_files(md_dir)

        print(f"[INFO] {md_dir} 에서 마크다운 파일 {len(md_files)}개 발견")

        for md_file in md_files:
            print(f"  처리중: {md_file}")

            chunks = llm_chunking(md_file, chain, meta_chain, decision_chain)

            if not chunks:  # SKIP된 파일은 카운트도 저장도 안 함
                continue
            
            all_chunks.extend(chunks)
            processed_files += 1

            print(f"  → {len(chunks)}개 청크 생성")

            if chunks:  # SKIP된 파일은 저장 안 함
                rel_parent = Path(md_file.stem)

                save_path = output_dir / rel_parent
                save_path.mkdir(parents=True, exist_ok=True)

                for chunk in chunks:
                    chunk_id = chunk["id"]
                    filename = f"{md_file.stem}_chunk{chunk_id:02d}.json"
                    with open(save_path / filename, "w", encoding="utf-8") as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)

            if max_files is not None and processed_files >= max_files:
                print(f"⚠️ max_files={max_files} 도달, 조기 종료")
                return all_chunks

    print(f"총 청크 수: {len(all_chunks)}")
    return all_chunks