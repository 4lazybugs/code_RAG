from pathlib import Path
from paddleocr import PaddleOCRVL
import re, time

# ---------------- PII Regex Masking (Local, Fast) ----------------
KR_MOBILE_RE = re.compile(r"(?<!\d)(01[016789])[-\s]?\d{3,4}[-\s]?\d{4}(?!\d)")

GENERIC_PHONE_RE = re.compile(
    r"""
    (?<!\d)
    (?:\+?\d{1,3}[\s\-.]?)?
    (?:\(?\d{1,4}\)?[\s\-/\.]*){2,5}
    \d{1,4}
    (?!\d)
    """,
    re.VERBOSE,
)

EMAIL_RE = re.compile(r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RRN_RE   = re.compile(r"(?<!\d)\d{6}[-\s]?\d{7}(?!\d)")

KOR_SURNAMES = "김|이|박|최|정|강|조|주|선|윤|장|임|한|오|서|신|권|황|안|송|류|전|홍|고|문|양|손|배|백|허|유|남|심|노|도|예|변|방|단|추|사|엄|어|주|장|임|전|홍|고|문|양|손|배|백|허|유|남|심|노"
NAME_RE = re.compile(
    rf"(^|[^가-힣])((?:{KOR_SURNAMES})[가-힣]{{1,2}})(?=[^가-힣]|$)"
)

def _mask_generic_phone(text: str, min_digits: int = 6) -> str:
    def repl(m: re.Match) -> str:
        s = m.group(0)
        digits = re.sub(r"\D", "", s)
        return "[PHONE]" if len(digits) >= min_digits else s
    return GENERIC_PHONE_RE.sub(repl, text)

def mask_pii_md(text: str) -> str:
    text = KR_MOBILE_RE.sub("[PHONE]", text)
    text = _mask_generic_phone(text, min_digits=6)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = RRN_RE.sub("[RRN]", text)
    text = NAME_RE.sub(r"\1[NAME]", text)
    return text
# ------------------------------------------------------------------


def extract_n_save(
    pipeline: PaddleOCRVL,
    input_dir: Path,
    save_dir: Path,
    clean_dir: Path,
    *,
    do_filter: bool = True,
    max_chars: int = 70,
    keep_structure: bool = True,
) -> None:
    """
    input_dir 하위 PDF들을 재귀 탐색하여 OCR -> 페이지별 md 저장.
    do_filter=True이면 md_text를 mask_pii_md로 정제 후 clean_dir에 저장.
    do_filter=False이면 clean_dir에는 원본 md를 그대로 복사 저장(=res.save_to_markdown)만 수행.
    """
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)
    clean_dir = Path(clean_dir)

    pdf_files = list(input_dir.rglob("*.pdf"))
    print(f"총 {len(pdf_files)}개의 PDF를 찾았습니다. ({input_dir})")

    save_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path}")

        output = pipeline.predict(str(pdf_path))

        for page_idx, res in enumerate(output):
            # 원본 코드 유지: rel_path / out_path 구성
            rel_path = pdf_path.relative_to(input_dir) if keep_structure else Path(pdf_path.name)

            out_path = save_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # PaddleOCRVL 결과를 md로 저장(페이지 파일 생성)
            res.save_to_markdown(save_path=out_path)
            print(f"Saved page {page_idx} → {out_path}")

            # 생성된 페이지 md 파일 읽기
            md_path = out_path / f"{rel_path.stem}_{page_idx}.md"
            if not md_path.exists():
                # 예상 파일이 없으면 스킵(모델 버전에 따라 저장 규칙이 다를 수 있음)
                print(f"[WARN] md not found: {md_path}")
                continue

            md_text = md_path.read_text(encoding="utf-8")
            if len(md_text) < max_chars:
                print(f"[SKIP] {rel_path}_{page_idx} ({len(md_text)} chars)")
                continue

            # clean 저장 경로
            clean_path = clean_dir / rel_path
            clean_path.parent.mkdir(parents=True, exist_ok=True)

            # do_filter=False면 원본 md를 clean_dir에 그대로 저장 (원 코드 흐름 유지)
            res.save_to_markdown(save_path=clean_path)

            if do_filter:
                cleaned_text = mask_pii_md(md_text)
                cleaned_md_path = clean_path / f"{rel_path.stem}_{page_idx}.md"
                cleaned_md_path.write_text(cleaned_text, encoding="utf-8")
                print(f"Saved cleaned page {page_idx} → {cleaned_md_path}")
            else:
                # 필터링 안 하면, 그냥 저장된 md 경로를 로그로 출력
                saved_md_path = clean_path / f"{rel_path.stem}_{page_idx}.md"
                print(f"Saved (no filter) page {page_idx} → {saved_md_path}")


if __name__ == "__main__":
    start = time.time()

    # OCR pipeline 초기화
    pipeline = PaddleOCRVL()

    # -------- farm_consulting --------
    extract_n_save(
        pipeline=pipeline,
        input_dir=Path("db/raw_db/farm_consulting/"),
        save_dir=Path("db/raw_db_extracted/farm_consulting/"),
        clean_dir=Path("db/cleaned_md/farm_consulting/not_filtered/"),
        do_filter=False,     # 여기서 옵션으로 제어
        max_chars=70,
    )

    '''
    # -------- manual_book --------
    extract_n_save(
        pipeline=pipeline,
        input_dir=Path("db/raw_db/manual_book/"),
        save_dir=Path("db/raw_db_extracted/manual_book/"),
        clean_dir=Path("db/cleaned_md/manual_book/"),
        do_filter=False,    # 예: manual_book은 필터링 안 함
        max_chars=70,
    )
    '''

    end = time.time()
    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n총 걸린 시간: {int(minutes)}분 {seconds:.2f}초")
