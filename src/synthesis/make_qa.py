import json
from pathlib import Path
from tqdm import tqdm

def gen_from_chunks(chunk_root: Path, params, max_qa: int | None = None):
    llm = params.get_llm("llm")
    prompt = params.get_params("prompt_qa")
    chain = prompt | llm

    results = []
    chunk_files = sorted(chunk_root.rglob("*_chunk*.json"))
    print(f"found chunk files: {len(chunk_files)}")

    for chunk_file in tqdm(chunk_files, desc="chunk_qa"):
        chunk = json.loads(chunk_file.read_text(encoding="utf-8"))

        try:
            resp = chain.invoke({
                "id": len(results),
                "source_file": chunk.get("source_file", ""),
                "raw_chunk": chunk.get("raw_chunk", ""),
                "md_summary": chunk.get("md_summary", ""),
            })
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            print(f"❌ invoke failed: {e}")
            continue

        try:
            arr = json.loads(content)
            if not isinstance(arr, list):
                raise ValueError
        except Exception:
            print(f"❌ JSON parse failed: {content[:200]}")
            continue

        for item in arr:
            results.append({
                "id": len(results),
                "chunk_name": str(chunk_file),  # ← 이거 추가
                "source_files": chunk.get("source_files", ""),
                "raw_chunk": chunk.get("raw_chunk", ""),
                "md_summary": chunk.get("md_summary", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            })

            if max_qa is not None and len(results) >= max_qa:
                return results

    return results