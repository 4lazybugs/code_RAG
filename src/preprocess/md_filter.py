import re
from pathlib import Path


def filt_and_save(
    search_dirs: list[Path],
    output_dir: Path,
    headers: list[str] | None = None,
) -> None:
    """
    Args:
        search_dirs: 검색할 디렉토리 목록 (원본 경로)
        output_dir: 필터링된 파일을 저장할 새 경로
        headers: 기준이 되는 헤더 문자열 목록 (하나라도 매칭되면 OK)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"saved": 0, "skipped": 0, "error": 0}

    for search_dir in search_dirs:
        search_dir = Path(search_dir)
        if not search_dir.exists():
            print(f"[WARN] 디렉토리가 존재하지 않습니다: {search_dir}")
            continue

        pattern = "*.md"
        md_files = sorted(search_dir.rglob(pattern))
        print(f"[INFO] {search_dir} 에서 마크다운 파일 {len(md_files)}개 발견\n")

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                lines = content.splitlines(keepends=True)

                # 헤더 목록 중 하나라도 매칭되는 첫 번째 줄 찾기
                header_idx = None
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if any(stripped == h for h in headers):
                        header_idx = i
                        break

                if header_idx is None:
                    print(f"  [SKIP] 헤더 없음, 저장 안 함 → {md_file.name}")
                    stats["skipped"] += 1
                    continue

                trimmed = "".join(lines[header_idx:])
                
                # 마지막 _숫자 제거해서 그룹 폴더명 생성
                group_name = re.sub(r'_\d+$', '', md_file.stem)

                dest_path = output_dir / group_name / md_file.name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_text(trimmed, encoding="utf-8")
                
                print(f"  [OK] 저장 완료 → {dest_path.name} "
                      f"({header_idx}줄 제거, {len(lines) - header_idx}줄 저장)")
                stats["saved"] += 1

            except Exception as e:
                print(f"  [ERROR] 처리 실패 ({md_file}): {e}")
                stats["error"] += 1

    print(f"\n=== 완료 ===")
    print(f"  저장: {stats['saved']}개  →  {output_dir}")
    print(f"  헤더 없어서 스킵: {stats['skipped']}개")
    print(f"  오류: {stats['error']}개")