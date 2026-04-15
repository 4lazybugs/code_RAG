"""
OCR 품질 평가기 - LLM-as-a-Judge (Anthropic Claude VLM)
원본 페이지 이미지 + 추출된 텍스트를 같이 평가합니다.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()


# ── PDF → base64 이미지 ───────────────────────────────────────────────────────
def pdf_to_base64(pdf_path: Path, dpi: int = 150) -> str | None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "page")
            subprocess.run(
                ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), out_prefix],
                capture_output=True, timeout=30
            )
            out_file = Path(f"{out_prefix}.png")
            if out_file.exists():
                return base64.b64encode(out_file.read_bytes()).decode("utf-8")
            print(f"  [이미지 변환 실패] 출력 파일 없음: {pdf_path}")
            return None
    except FileNotFoundError:
        print("  [오류] pdftoppm 없음. sudo apt install poppler-utils 실행하세요.")
        return None
    except Exception as e:
        print(f"  [이미지 변환 실패] {pdf_path}: {e}")
        return None


# ── MD에 대응하는 PDF 경로 찾기 ───────────────────────────────────────────────
def find_pdf(md_path: Path) -> Path | None:
    pdf_path = md_path.parent / "pdfs" / md_path.with_suffix(".pdf").name
    return pdf_path if pdf_path.exists() else None


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────
JUDGE_SYSTEM = """\
당신은 OCR 추출 결과를 평가합니다.
원본 이미지를 보고 추출된 텍스트와 비교하세요.

반드시 아래 JSON 형식으로만 응답하세요.
{
  "grade": "<good|bad>",
  "reason": "한 문장 요약"
}

bad: 오직 이 경우만 bad입니다.
     원본 이미지에 있는 텍스트의 대부분이 추출 결과에 누락된 경우.

그 외 모든 경우는 good입니다.
"""

JUDGE_USER_TMPL = """\
아래는 OCR로 추출된 마크다운 텍스트입니다.
원본 이미지와 비교하여 품질을 평가하고 JSON으로만 응답하세요.

--- 텍스트 시작 ---
{text}
--- 텍스트 끝 ---
"""


# ── LLM 평가 ─────────────────────────────────────────────────────────────────
def llm_judge(text: str, pdf_path: Path | None = None, max_chars: int = 1500, retries: int = 3) -> dict:
    snippet = text[:max_chars] if text else "(텍스트 없음)"
    prompt = JUDGE_USER_TMPL.format(text=snippet)

    img_b64 = pdf_to_base64(pdf_path) if pdf_path else None
    if img_b64:
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = prompt

    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = msg.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            return {
                "llm_grade":  result.get("grade", "bad"),
                "llm_reason": result.get("reason", ""),
                "used_image": img_b64 is not None,
            }
        except json.JSONDecodeError:
            time.sleep(1)
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"  [RateLimit] {wait}초 대기...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [LLM 오류] {e}")
            break

    return {"llm_grade": "bad", "llm_reason": "LLM 평가 실패", "used_image": False}


# ── 파일 단위 평가 ────────────────────────────────────────────────────────────
def evaluate_file(md_path: Path) -> dict:
    text     = md_path.read_text(encoding="utf-8")
    pdf_path = find_pdf(md_path)

    base = {
        "file": md_path.name,
        "doc":  md_path.parent.name,
        "char_count": len(text),
        "used_image": False,
        "llm_grade": None,
        "llm_reason": "",
        "status": "bad",
    }

    llm_result = llm_judge(text, pdf_path=pdf_path)
    base.update(llm_result)
    base["status"] = base.get("llm_grade", "bad")
    return base


# ── 전체 평가 (병렬) ──────────────────────────────────────────────────────────
def evaluate_all(extracted_root: Path, max_workers: int = 5) -> list[dict]:
    md_paths = sorted(extracted_root.rglob("*.md"))
    print(f"총 {len(md_paths)}개 파일 평가 시작 (workers={max_workers})\n")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_file, md_path): md_path for md_path in md_paths}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            img_flag = "🖼" if r.get("used_image") else "📄"
            print(f"{img_flag} [{r['status'].upper():4s}] {r['doc']}/{r['file']}")
            if r.get("llm_reason"):
                print(f"          └─ {r['llm_reason']}")

    return sorted(results, key=lambda x: x["file"])


# ── 리포트 저장 ───────────────────────────────────────────────────────────────
def save_report(results: list[dict], out_path: Path = Path("ocr_quality_report.csv")) -> None:
    if not results:
        print("저장할 bad 파일 없음")
        return

    fieldnames = ["doc", "file", "char_count", "used_image", "status", "llm_grade", "llm_reason"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n→ {out_path} 저장 완료 ({len(results)}행)")


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = Path("db/raw_db_extracted/consulting_2022_01")

    results = evaluate_all(extracted_root=root, max_workers=5)

    status_counts = Counter(r["status"] for r in results)
    print(f"\n=== 요약 ===")
    print(f"전체 파일: {len(results)}개")
    for status, count in status_counts.most_common():
        print(f"  {status:6s}: {count}개")

    bad = [r for r in results if r["status"] == "bad"]
    save_report(bad)