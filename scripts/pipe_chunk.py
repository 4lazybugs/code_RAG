import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_config
from src.prompts.chunking_prompt import lumber_prompt
from src.preprocess.chunking import (
    recursive_chunking, fixed_size_chunking, semantic_chunking, split_sentences,
    lumber_chunking_from_text,
)
from src.preprocess.md_utils import collect_all_md_files


#########  파라미터  #######################
CFG = get_config("configs/config_gen.yaml")

MD_DIRS   = [Path(d) for d in CFG.md_in_dirs]
CHUNK_DIR = Path(CFG.chunk_out_dir)

load_dotenv()

llm          = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
embed_model  = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},   # GPU 있으면 "cuda"
    encode_kwargs={"normalize_embeddings": True},
)
lumber_chain = lumber_prompt | llm
###############################################

# 필터링/변환은 이미 끝난 상태 -> 이미 처리된 결과 폴더를 입력으로 사용
INPUT_DIR = CHUNK_DIR / "converted"


def run_chunking(
    input_files: list[Path],
    output_dir: Path,
    chunk_fn,
    tag: str,
    **kwargs,
) -> list[dict]:
    """recursive / fixed / semantic / lumber 공통 실행기.
    필터링 없이 입력 파일을 그대로 읽어서 청킹만 수행한다.
    chunk_fn은 text: str -> list[str] 형태로 통일해서 넘긴다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    idx = 1
    for f in input_files:
        text = f.read_text(encoding="utf-8")
        if not text.strip():
            continue

        chunks = chunk_fn(text, **kwargs)
        for chunk in chunks:
            record = {
                "id": idx,
                "source_files": [f.name],
                "md_summary": None,
                "raw_chunk": chunk,
            }
            all_chunks.append(record)
            with open(output_dir / f"{tag}_{idx:03d}.json", "w", encoding="utf-8") as out:
                json.dump(record, out, ensure_ascii=False, indent=2)
            idx += 1

    print(f"[INFO] [{tag}] {len(all_chunks)}개 청크 저장 완료 → {output_dir}")
    return all_chunks


def run_chunking_parallel(
    input_files: list[Path],
    output_dir: Path,
    chunk_fn,
    tag: str,
    max_workers: int = 8,
    **kwargs,
) -> list[dict]:
    """lumber처럼 파일 하나당 LLM 호출이 여러 번 순차로 발생하는 chunk_fn을
    파일 단위로 병렬 실행한다. (문서 간에는 서로 독립적이므로 안전)

    LLM 호출은 네트워크 I/O 대기가 대부분이라 스레드로도 충분히 빨라짐.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _process(f: Path):
        text = f.read_text(encoding="utf-8")
        if not text.strip():
            return f, []
        return f, chunk_fn(text, **kwargs)

    file_results: dict[Path, list[str]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, f): f for f in input_files}
        for future in as_completed(futures):
            f, chunks = future.result()
            file_results[f] = chunks
            done += 1
            if done % 20 == 0 or done == len(input_files):
                print(f"[INFO] [{tag}] 진행 {done}/{len(input_files)}")

    # 저장은 입력 순서대로 (id 부여 순서를 안정적으로 유지하기 위함)
    all_chunks = []
    idx = 1
    for f in input_files:
        for chunk in file_results.get(f, []):
            record = {
                "id": idx,
                "source_files": [f.name],
                "md_summary": None,
                "raw_chunk": chunk,
            }
            all_chunks.append(record)
            with open(output_dir / f"{tag}_{idx:03d}.json", "w", encoding="utf-8") as out:
                json.dump(record, out, ensure_ascii=False, indent=2)
            idx += 1

    print(f"[INFO] [{tag}] {len(all_chunks)}개 청크 저장 완료 → {output_dir}")
    return all_chunks


if __name__ == "__main__":
    input_files = collect_all_md_files([INPUT_DIR])
    print(f"[INFO] 청킹 대상: {len(input_files)}개 파일 (필터링 skip, {INPUT_DIR})")

    run_chunking_parallel(
        input_files, CHUNK_DIR / "lumber",
        lambda t: lumber_chunking_from_text(t, lumber_chain, max_tokens=64), "lumber",
        max_workers=8,
    )

    run_chunking(
        input_files, CHUNK_DIR / "recursive", recursive_chunking, "recursive",
        chunk_size=64, overlap=50,
    )
    run_chunking(
        input_files, CHUNK_DIR / "fixed", fixed_size_chunking, "fixed",
        max_tokens=64, overlap=50,
    )
    run_chunking(
        input_files, CHUNK_DIR / "semantic",
        lambda t: semantic_chunking(split_sentences(t), embed_model), "semantic",
    )