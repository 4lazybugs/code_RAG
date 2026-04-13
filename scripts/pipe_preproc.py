import json
from src.config import get_config
from pathlib import Path

from src.preprocess.md_filter import trim_and_save
from src.preprocess.chunking import llm_chunking

CFG = get_config("configs/config_preproc.yaml")

###########################################################
MD_DIRS = [Path(d) for d in CFG.md_dirs]
CHUNK_DIR = Path(CFG.chunk_dir)
##########################################################


def chunking_and_save(md_dirs: list[Path], output_dir: Path, max_files: int | None = None) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for md_dir in md_dirs:
        md_dir = Path(md_dir)
        if not md_dir.exists():
            print(f"[WARN] 디렉토리가 존재하지 않습니다: {md_dir}")
            continue

        md_files = sorted(md_dir.glob("*.md"))
        print(f"[INFO] {md_dir} 에서 마크다운 파일 {len(md_files)}개 발견")

        for md_file in md_files:
            print(f"  처리중: {md_file.name}")
            chunks = llm_chunking(md_file)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)}개 청크 생성")

            save_path = output_dir / md_dir.name / md_file.stem
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / "chunks.json", "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            if max_files is not None and len(all_chunks) >= max_files:
                print(f"⚠️ max_files={max_files} 도달, 조기 종료")
                return all_chunks

    return all_chunks

if __name__ == "__main__":

    # trim_and_save(
    #     search_dirs=SEARCH_DIRS,
    #     output_dir=OUTPUT_DIR,
    #     headers=HEADERS,
    # )

    all_chunks = chunking_and_save(
        md_dirs=MD_DIRS,
        output_dir=CHUNK_DIR,
        max_files=CFG.max_fnum
    )
    print(f"총 청크 수: {len(all_chunks)}")
    # print(all_chunks[0])  # 첫 번째 청크 확인
