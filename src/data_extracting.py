from pathlib import Path
from paddleocr import PaddleOCRVL
import re, time

# ---------------- PII Regex Masking (Local, Fast) ----------------
# 1) 정형(한국 휴대폰)
KR_MOBILE_RE = re.compile(r"(?<!\d)(01[016789])[-\s]?\d{3,4}[-\s]?\d{4}(?!\d)")

# 2) 범용(국제/비정형/깨진 OCR 대응) : (0) (222) 758/ 같은 케이스 포함
GENERIC_PHONE_RE = re.compile(
    r"""
    (?<!\d)                       # 앞이 숫자가 아니고
    (?:\+?\d{1,3}[\s\-.]?)?       # 국가번호(optional)
    (?:\(?\d{1,4}\)?[\s\-/\.]*){2,5}  # 괄호/공백/슬래시/점/하이픈 포함 블록 2~5회
    \d{1,4}                       # 마지막 숫자 블록
    (?!\d)                        # 뒤가 숫자가 아님
    """,
    re.VERBOSE,
)

EMAIL_RE = re.compile(r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RRN_RE   = re.compile(r"(?<!\d)\d{6}[-\s]?\d{7}(?!\d)")  # 주민번호(단순 패턴)

# 3) 이름(문서 전체에서 모두 마스킹) - 표(|), HTML 태그 등에서도 안정적으로 동작하도록 경계 재정의
KOR_SURNAMES = "김|이|박|최|정|강|조|윤|장|임|한|오|서|신|권|황|안|송|류|전|홍|고|문|양|손|배|백|허|유|남|심|노"
NAME_RE = re.compile(
    rf"(^|[^가-힣])((?:{KOR_SURNAMES})[가-힣]{{1,2}})(?=[^가-힣]|$)"
)

def _mask_generic_phone(text: str, min_digits: int = 6) -> str:
    """
    범용 전화번호 후보 중 '숫자 개수'가 일정 이상인 것만 [PHONE]으로 치환해 오탐을 줄인다.
    min_digits=6: (0) (222) 758/ 같은 깨진 케이스도 대부분 잡히며,
    너무 공격적이면 7~8로 올리면 된다.
    """
    def repl(m: re.Match) -> str:
        s = m.group(0)
        digits = re.sub(r"\D", "", s)
        return "[PHONE]" if len(digits) >= min_digits else s

    return GENERIC_PHONE_RE.sub(repl, text)

def mask_pii_md(text: str) -> str:
    # 전화번호: 정형 -> 범용(오탐 방지 포함) 순서 권장
    text = KR_MOBILE_RE.sub("[PHONE]", text)
    text = _mask_generic_phone(text, min_digits=6)

    # 이메일 / 주민번호
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = RRN_RE.sub("[RRN]", text)

    # 이름: 전체 마스킹 (앞 경계는 보존하고 이름만 치환)
    text = NAME_RE.sub(r"\1[NAME]", text)

    return text
# ------------------------------------------------------------------

if __name__ == "__main__":
    start = time.time()

    # OCR pipeline 초기화 (모델 로딩)
    pipeline = PaddleOCRVL()

    ####### farm_consulting #########################################################
    # 입력 폴더 (재귀 탐색)
    input_dir = Path("db/raw_db/farm_consulting/")
    pdf_files = list(input_dir.rglob("*.pdf"))
    print(f"총 {len(pdf_files)}개의 PDF를 찾았습니다.")

    # 출력 폴더
    save_dir = Path("db/raw_db_extracted/farm_consulting/")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 정제된 md를 저장할 루트
    clean_dir = Path("db/cleaned_md/farm_consulting/")
    clean_dir.mkdir(parents=True, exist_ok=True)

    MAX_CHARS = 300  # 스킵 기준(문자 수)
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path}")

        # PDF 한 개 OCR 수행
        output = pipeline.predict(str(pdf_path))
        
        # 각 페이지 결과 저장
        for page_idx, res in enumerate(output):
            # Markdown 파일 이름 생성: test2_0.md처럼
            # 출력 경로: 원래 구조 유지하고 싶으면 상대 경로 그대로 써도 됨
            rel_path = pdf_path.relative_to(input_dir)        # raw_db_extracted 이하 경로
            out_path = save_dir / rel_path              # 동일 구조로 저장
            out_path.parent.mkdir(parents=True, exist_ok=True)
            res.save_to_markdown(save_path=out_path)
            print(f"Saved page {page_idx} → {out_path}")

            if len(res) < MAX_CHARS:
                print(f"[SKIP] 길이 초과({len(res)} chars) → 저장 안함: {out_path}")
                continue

            cleaned = mask_pii_md(res)   # ✅ 저장 전에 PII 마스킹
            
            clean_path = clean_dir / rel_path              # 동일 구조로 저장
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            # md로 저장
            cleaned.save_to_markdown(save_path=clean_path)
            print(f"Saved page {page_idx} → {clean_path}")
    ##################################################################################

    ####### ~farm_consulting #########################################################
    # 입력 폴더 (재귀 탐색)
    input_dir = Path("db/raw_db/manual_book/")
    pdf_files = list(input_dir.rglob("*.pdf"))
    print(f"총 {len(pdf_files)}개의 PDF를 찾았습니다.")

    # 출력 폴더
    save_dir = Path("db/raw_db_extracted/manual_book/")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 정제된 md를 저장할 루트
    clean_dir = Path("db/cleaned_md/manual_book/")
    clean_dir.mkdir(parents=True, exist_ok=True)

    MAX_CHARS = 300  # 스킵 기준(문자 수)
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path}")

        # PDF 한 개 OCR 수행
        output = pipeline.predict(str(pdf_path))
        
        # 각 페이지 결과 저장
        for page_idx, res in enumerate(output):
            # Markdown 파일 이름 생성: test2_0.md처럼
            # 출력 경로: 원래 구조 유지하고 싶으면 상대 경로 그대로 써도 됨
            rel_path = pdf_path.relative_to(input_dir)        # raw_db_extracted 이하 경로
            out_path = save_dir / rel_path              # 동일 구조로 저장
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # md로 저장
            res.save_to_markdown(save_path=out_path)
            print(f"Saved page {page_idx} → {out_path}")

            if len(res) < MAX_CHARS:
                print(f"[SKIP] 길이 초과({len(res)} chars) → 저장 안함: {out_path}")
                continue

            clean_path = clean_dir / rel_path              # 동일 구조로 저장
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            # md로 저장
            res.save_to_markdown(save_path=clean_path)
            print(f"Saved page {page_idx} → {clean_path}")
    ##################################################################################

    end = time.time()
    elapsed = end - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n총 걸린 시간: {int(minutes)}분 {seconds:.2f}초")