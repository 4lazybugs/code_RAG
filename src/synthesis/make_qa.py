import json
from pathlib import Path
from tqdm import tqdm

def gen_from_chunks(chunk_root: Path, params, max_qa: int | None = None):
    llm = params.get_llm("llm")
    prompt = params.get_params("prompt_qa")
    chain = prompt | llm

    chunk_files = sorted(chunk_root.rglob("*_chunk*.json"))
    print(f"found chunk files: {len(chunk_files)}")

    chunks = [json.loads(f.read_text(encoding="utf-8")) for f in chunk_files]
    inputs = [
        {
            "id": i,
            "source_file": c.get("source_file", ""),
            "raw_chunk": c.get("raw_chunk", ""),
            "md_summary": c.get("md_summary", ""),
        }
        for i, c in enumerate(chunks)
    ]

    responses = chain.batch(inputs, config={"max_concurrency": 10})

    results = []
    for chunk_file, chunk, resp in tqdm(zip(chunk_files, chunks, responses), total=len(chunks), desc="QA 생성"):
        content = resp.content if hasattr(resp, "content") else str(resp)
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
                "chunk_name": str(chunk_file),
                "source_files": chunk.get("source_files", ""),
                "raw_chunk": chunk.get("raw_chunk", ""),
                "md_summary": chunk.get("md_summary", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            })
            if max_qa is not None and len(results) >= max_qa:
                return results

    return results