import re
from pathlib import Path

from src.preprocess.chunking import decision_batch

def collect_md_files(md_dir: Path) -> list[Path]:
    def natural_key(path: Path):
        parts = re.split(r'(\d+)', path.stem)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    files = sorted(md_dir.rglob("*.md"), key=natural_key)
    print(f"[INFO] 탐색 경로: {md_dir}")
    for f in files:
        print(f"  발견: {f.relative_to(md_dir)}")
    return files


def collect_all_md_files(md_dirs: list[Path]) -> list[Path]:
    """md_dirs 전체(파일/폴더 혼합)를 순회하며 md 파일 목록을 모은다."""
    all_files = []
    for d in md_dirs:
        d = Path(d)
        if d.is_file() and d.suffix == ".md":
            all_files.append(d)
        elif d.is_dir():
            all_files.extend(collect_md_files(d))
        else:
            print(f"[WARN] 건너뜀 (파일/폴더 아님): {d}")
    return all_files


# 같은 디렉토리에 대해 Decision 필터링을 여러 번 반복하지 않도록 캐싱한다.
# 동일한 md_dirs를 공유하므로, 필터링 결과를 재사용해 LLM 호출을 줄인다.
_FILTER_CACHE: dict[Path, list[tuple[Path, str]]] = {}


def get_filtered_pages(
    cache_key: Path,
    page_items: list[tuple[Path, str]],
    decision_chain,
) -> list[tuple[Path, str]]:
    if cache_key in _FILTER_CACHE:
        cached = _FILTER_CACHE[cache_key]
        print(f"[INFO] 필터링 캐시 재사용: {cache_key} ({len(cached)}개 페이지)")
        return cached

    raw_texts = [t for _, t in page_items]
    useful_flags = decision_batch(raw_texts, decision_chain)
    useful_items = [(f, t) for (f, t), ok in zip(page_items, useful_flags) if ok]
    _FILTER_CACHE[cache_key] = useful_items
    return useful_items