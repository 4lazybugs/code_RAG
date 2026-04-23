"""
AI 추출 md vs OCR 추출 md를 TEDS로 비교하는 스크립트

입력:
- reference_root: AI가 추출한 markdown 파일 또는 디렉터리
- pred_root: OCR이 추출한 markdown 파일 또는 디렉터리

동작:
- 파일이면 1:1 비교
- 디렉터리면 재귀적으로 *.md 탐색 후 relative path 기준 매칭
- markdown table만 추출해서 HTML table로 변환 후 TEDS 계산
"""

from __future__ import annotations

import csv
import html
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup, Tag, NavigableString
from apted import APTED, Config


# =========================================================
# 1. Markdown table parsing
# =========================================================

TABLE_BORDER_RE = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_CELL_RE = re.compile(r"^\s*:?-{3,}:?\s*$")


def normalize_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def split_md_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [normalize_text(cell) for cell in line.split("|")]


def is_separator_row(cells: List[str]) -> bool:
    if not cells:
        return False
    return all(SEPARATOR_CELL_RE.match(cell or "") for cell in cells)


def extract_markdown_tables(md_text: str) -> List[List[List[str]]]:
    """
    markdown에서 table block만 추출
    반환:
      [
        [header_cells, row1_cells, row2_cells, ...],
        ...
      ]
    """
    lines = md_text.splitlines()
    tables = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if not TABLE_BORDER_RE.match(line):
            i += 1
            continue

        if i + 1 >= n:
            i += 1
            continue

        header_cells = split_md_row(lines[i])
        sep_cells = split_md_row(lines[i + 1])

        if not is_separator_row(sep_cells):
            i += 1
            continue

        rows = [header_cells]
        i += 2

        while i < n and TABLE_BORDER_RE.match(lines[i]):
            rows.append(split_md_row(lines[i]))
            i += 1

        tables.append(rows)

    return tables


def pad_rows(table: List[List[str]]) -> List[List[str]]:
    max_cols = max(len(row) for row in table) if table else 0
    return [row + [""] * (max_cols - len(row)) for row in table]


def markdown_table_to_html(table: List[List[str]]) -> str:
    if not table:
        return "<table></table>"

    table = pad_rows(table)
    header = table[0]
    body = table[1:]

    parts = ["<table>", "<thead>", "<tr>"]
    for cell in header:
        parts.append(f"<th>{html.escape(cell)}</th>")
    parts.extend(["</tr>", "</thead>", "<tbody>"])

    for row in body:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{html.escape(cell)}</td>")
        parts.append("</tr>")

    parts.extend(["</tbody>", "</table>"])
    return "".join(parts)


# =========================================================
# 2. TEDS
# =========================================================

class TableTree:
    def __init__(self, tag: str, text: str = "", children: Optional[List["TableTree"]] = None):
        self.tag = tag
        self.text = normalize_text(text)
        self.children = children or []


def html_to_tree(node) -> Optional[TableTree]:
    if isinstance(node, NavigableString):
        text = normalize_text(str(node))
        if text:
            return TableTree(tag="text", text=text)
        return None

    if not isinstance(node, Tag):
        return None

    children = []
    for child in node.children:
        tree_child = html_to_tree(child)
        if tree_child is not None:
            children.append(tree_child)

    if not children:
        return TableTree(tag=node.name, text=node.get_text(" ", strip=True), children=[])

    return TableTree(tag=node.name, text="", children=children)


def count_nodes(node: Optional[TableTree]) -> int:
    if node is None:
        return 0
    return 1 + sum(count_nodes(child) for child in node.children)


def normalized_edit_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0

    m, n = len(a), len(b)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )
            prev = temp

    dist = dp[n]
    return dist / max(m, n)


class TEDSConfig(Config):
    def rename(self, node1: TableTree, node2: TableTree) -> float:
        if node1.tag != node2.tag:
            return 1.0

        if not node1.children and not node2.children:
            if node1.text == node2.text:
                return 0.0
            return normalized_edit_distance(node1.text, node2.text)

        return 0.0

    def children(self, node: TableTree):
        return node.children


def compute_teds_html(pred_html: str, true_html: str) -> float:
    pred_soup = BeautifulSoup(pred_html, "lxml")
    true_soup = BeautifulSoup(true_html, "lxml")

    pred_table = pred_soup.find("table")
    true_table = true_soup.find("table")

    if pred_table is None and true_table is None:
        return 1.0
    if pred_table is None or true_table is None:
        return 0.0

    pred_tree = html_to_tree(pred_table)
    true_tree = html_to_tree(true_table)

    if pred_tree is None and true_tree is None:
        return 1.0
    if pred_tree is None or true_tree is None:
        return 0.0

    apted = APTED(pred_tree, true_tree, TEDSConfig())
    dist = apted.compute_edit_distance()

    pred_nodes = count_nodes(pred_tree)
    true_nodes = count_nodes(true_tree)
    denom = max(pred_nodes, true_nodes, 1)

    return max(0.0, 1.0 - (dist / denom))


# =========================================================
# 3. File matching / evaluation
# =========================================================

@dataclass
class EvalResult:
    file: str
    rel_path: str
    ref_file: str
    pred_file: str
    num_ref_tables: int
    num_pred_tables: int
    teds_mean: float
    teds_scores: List[float]
    status: str
    note: str


def collect_md_files(path: Path) -> List[Path]:
    """
    path가:
    - 파일이면 [that file]
    - 디렉터리면 재귀적으로 *.md 수집
    """
    if path.is_file():
        if path.suffix.lower() != ".md":
            raise ValueError(f"Not a markdown file: {path}")
        return [path]

    if path.is_dir():
        return sorted(path.rglob("*.md"))

    raise ValueError(f"Invalid path: {path}")


def build_rel_key(file_path: Path, root_path: Path) -> str:
    """
    매칭용 relative key 생성
    파일이면 파일명만
    디렉터리면 relative path
    """
    if root_path.is_file():
        return file_path.name
    return str(file_path.relative_to(root_path))


def build_pred_map(pred_root: Path) -> dict[str, Path]:
    pred_files = collect_md_files(pred_root)
    pred_map = {}

    for pred_file in pred_files:
        key = build_rel_key(pred_file, pred_root)
        pred_map[key] = pred_file

    return pred_map


def compare_md_pair(ref_file: Path, pred_file: Path, ref_root: Path, pred_root: Path) -> EvalResult:
    ref_text = ref_file.read_text(encoding="utf-8")
    pred_text = pred_file.read_text(encoding="utf-8")

    ref_tables = extract_markdown_tables(ref_text)
    pred_tables = extract_markdown_tables(pred_text)

    rel_path = build_rel_key(ref_file, ref_root)

    if not ref_tables and not pred_tables:
        return EvalResult(
            file=ref_file.name,
            rel_path=rel_path,
            ref_file=str(ref_file),
            pred_file=str(pred_file),
            num_ref_tables=0,
            num_pred_tables=0,
            teds_mean=1.0,
            teds_scores=[],
            status="no_table",
            note="양쪽 모두 markdown table 없음",
        )

    if not ref_tables or not pred_tables:
        return EvalResult(
            file=ref_file.name,
            rel_path=rel_path,
            ref_file=str(ref_file),
            pred_file=str(pred_file),
            num_ref_tables=len(ref_tables),
            num_pred_tables=len(pred_tables),
            teds_mean=0.0,
            teds_scores=[],
            status="mismatch",
            note="한쪽에만 table 존재",
        )

    pair_count = min(len(ref_tables), len(pred_tables))
    scores = []

    for i in range(pair_count):
        ref_html = markdown_table_to_html(ref_tables[i])
        pred_html = markdown_table_to_html(pred_tables[i])
        score = compute_teds_html(pred_html=pred_html, true_html=ref_html)
        scores.append(score)

    count_penalty = abs(len(ref_tables) - len(pred_tables)) * 0.05
    mean_score = sum(scores) / len(scores) if scores else 0.0
    mean_score = max(0.0, mean_score - count_penalty)

    if mean_score >= 0.95:
        status = "excellent"
    elif mean_score >= 0.85:
        status = "good"
    elif mean_score >= 0.70:
        status = "warning"
    else:
        status = "bad"

    note = ""
    if len(ref_tables) != len(pred_tables):
        note = f"table count mismatch: ref={len(ref_tables)}, pred={len(pred_tables)}"

    return EvalResult(
        file=ref_file.name,
        rel_path=rel_path,
        ref_file=str(ref_file),
        pred_file=str(pred_file),
        num_ref_tables=len(ref_tables),
        num_pred_tables=len(pred_tables),
        teds_mean=round(mean_score, 4),
        teds_scores=[round(s, 4) for s in scores],
        status=status,
        note=note,
    )


def evaluate_all(reference_root: Path, pred_root: Path, max_workers: int = 8) -> List[EvalResult]:
    ref_files = collect_md_files(reference_root)
    pred_map = build_pred_map(pred_root)

    print(f"총 reference md 파일 {len(ref_files)}개 비교 시작 (workers={max_workers})\n")

    results: List[EvalResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for ref_file in ref_files:
            rel_key = build_rel_key(ref_file, reference_root)
            pred_file = pred_map.get(rel_key)

            if pred_file is None:
                results.append(
                    EvalResult(
                        file=ref_file.name,
                        rel_path=rel_key,
                        ref_file=str(ref_file),
                        pred_file="",
                        num_ref_tables=0,
                        num_pred_tables=0,
                        teds_mean=0.0,
                        teds_scores=[],
                        status="missing_pred",
                        note="matching OCR md 없음",
                    )
                )
                continue

            fut = executor.submit(compare_md_pair, ref_file, pred_file, reference_root, pred_root)
            futures[fut] = ref_file

        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print(
                f"[{r.status.upper():11s}] {r.rel_path} | "
                f"TEDS={r.teds_mean:.4f} | "
                f"ref_tables={r.num_ref_tables}, pred_tables={r.num_pred_tables}"
            )
            if r.note:
                print(f"    └─ {r.note}")

    return sorted(results, key=lambda x: x.rel_path)


# =========================================================
# 4. Save report
# =========================================================

def json_like_list(xs: List[float]) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in xs) + "]"


def save_report(results: List[EvalResult], out_path: Path = Path("teds_report.csv")) -> None:
    fieldnames = [
        "file",
        "rel_path",
        "ref_file",
        "pred_file",
        "num_ref_tables",
        "num_pred_tables",
        "teds_mean",
        "teds_scores",
        "status",
        "note",
    ]

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "file": r.file,
                "rel_path": r.rel_path,
                "ref_file": r.ref_file,
                "pred_file": r.pred_file,
                "num_ref_tables": r.num_ref_tables,
                "num_pred_tables": r.num_pred_tables,
                "teds_mean": r.teds_mean,
                "teds_scores": json_like_list(r.teds_scores),
                "status": r.status,
                "note": r.note,
            })

    print(f"\n→ {out_path} 저장 완료 ({len(results)}행)")


# =========================================================
# 5. Main
# =========================================================

if __name__ == "__main__":
    # 예시 1: 파일 vs 파일
    # reference_root = Path("db/raw_db_extracted_anthorpic/consulting_2022_01/mds/consulting_2022_01_1.md")
    # pred_root = Path("db/raw_db_extracted_resolution/consulting_2022_01/mds/consulting_2022_01_1.md")

    # 예시 2: 디렉터리 vs 디렉터리
    reference_root = Path("db/raw_db_extracted_anthorpic/consulting_2022_01/mds")
    pred_root = Path("db/raw_db_extracted_resolution/consulting_2022_01/mds")

    results = evaluate_all(reference_root=reference_root, pred_root=pred_root, max_workers=8)

    status_counts = Counter(r.status for r in results)
    mean_teds = sum(r.teds_mean for r in results) / len(results) if results else 0.0

    print("\n=== 요약 ===")
    print(f"전체 파일: {len(results)}개")
    print(f"평균 TEDS: {mean_teds:.4f}")
    for status, count in status_counts.most_common():
        print(f"  {status:11s}: {count}개")

    save_report(results, Path("teds_report.csv"))