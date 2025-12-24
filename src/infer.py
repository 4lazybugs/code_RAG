import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from rag import PartialExpert, RawLlmExpert, SelfAskExpert
import json
import time
from retriever import MultiCosineRetriever  # (기존 semantic 폴더별)
from retriever import load_cleaned_md_level2_retrievers  # ✅ cleaned_md/<L1>/<L2> 폴더별 (당신이 만든 것)
from pathlib import Path
import re

# Expert 클래스들
from rag import PartialExpert, RawLlmExpert, SelfAskExpert
from models_eval.base import BaseEvaluator
from tqdm import tqdm

def make_pretty(text: str,
                max_block_chars: int = 1200,
                max_line_len: int = 140) -> list[str]:
    """
    입력이 HTML/Markdown/plain text 무엇이든 "보기 좋은 블록(list[str])"으로 정리한다.
    - HTML table(<tr>)이면: <tr> 단위로 블록 생성
    - 그 외(MD/plain): 헤더/표/리스트/문단 단위로 블록 생성
    - 너무 긴 블록은 잘라서 "...(truncated)" 처리
    """
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return []

    # 1) HTML table row(<tr>) 있으면 <tr> 기준으로 정리
    trs = re.findall(r"<tr.*?>.*?</tr>", s, flags=re.DOTALL | re.IGNORECASE)
    if trs:
        out = []
        for tr in trs:
            lines = [ln.strip() for ln in tr.splitlines() if ln.strip()]
            block = "\n".join(lines).strip()
            if block:
                out.append(block[:max_block_chars] + ("\n...(truncated)" if len(block) > max_block_chars else ""))
        return out

    # 2) HTML이 아니면(MD/plain) 블록화: 헤더/표/리스트/문단 기준
    lines = s.splitlines()
    out, cur = [], []
    in_table = False

    def flush():
        nonlocal cur
        if not cur:
            return
        block_lines = []
        for ln in cur:
            ln = ln.strip()
            if not ln:
                continue
            # 너무 긴 줄은 보기 좋게 분할
            if len(ln) > max_line_len:
                # 단어 경계 기준으로 대충 분할(외부 함수 없이)
                start = 0
                while start < len(ln):
                    cut = min(start + max_line_len, len(ln))
                    # 중간에서 끊길 때 공백 위치로 당겨오기
                    if cut < len(ln):
                        sp = ln.rfind(" ", start, cut)
                        if sp > start + 20:
                            cut = sp
                    block_lines.append(ln[start:cut].rstrip())
                    start = cut + 1 if cut < len(ln) and ln[cut:cut+1] == " " else cut
            else:
                block_lines.append(ln)

        block = "\n".join(block_lines).strip()
        if block:
            if len(block) > max_block_chars:
                block = block[:max_block_chars] + "\n...(truncated)"
            out.append(block)
        cur = []

    for raw in lines:
        ln = raw.rstrip()
        st = ln.strip()

        # 빈 줄 = 문단 경계
        if not st:
            in_table = False
            flush()
            continue

        # 헤더(# ...)는 단독 블록
        if re.match(r"^#{1,6}\s+\S+", st):
            flush()
            cur.append(st)
            flush()
            continue

        # 마크다운 표 라인(|...| 또는 구분선) 처리
        is_table_line = (st.startswith("|") and "|" in st) or bool(re.match(r"^\s*\|?[-: ]+\|[-|: ]+\s*$", st))
        if is_table_line:
            if not in_table:
                flush()
                in_table = True
            cur.append(st)
            continue
        if in_table and not is_table_line:
            flush()
            in_table = False

        # 리스트(-,*,+,1.)는 연속되는 동안 하나로 묶기
        if re.match(r"^(\s*[-*+]\s+|\s*\d+\.\s+)\S+", ln):
            cur.append(ln)
            continue

        # 일반 텍스트는 문단으로
        cur.append(st)

    flush()
    return out


class QagOnly(BaseEvaluator):
    metric_key = "qag_only"

    def __init__(self, expert, qa_data_path: str, sample_size: int = None):
        super().__init__(expert, qa_data_path, sample_size)
        self._retrieved = {}  # ✅ 추가

    def compute_scores(self, references: list, generated: list) -> list: 
        # QAG만 만들 거라 점수 계산은 사용하지 않음 
        return []

    def generate(self, questions: list, mode: str) -> list:
        preds = []
        infos = []
        retrieved_all = []  # ✅ 질문별 retrieved 저장 (핵심)

        for q in tqdm(questions, desc=f"Generating ({mode})"):
            res = self.expert.handle(q)

            # ---- retrieved 수집 (질문 1개 끝날 때마다) ----
            docs = getattr(self.expert, "retrieved_snippets", None)
            if docs is None:
                docs = getattr(self.expert, "last_retrieved_docs", [])  # 백업

            retrieved_list = []            
            for d in (docs or []):
                content_lines = make_pretty(d.page_content)
                retrieved_list.append({
                    "rank": d.metadata.get("__rank__"),
                    "filename": d.metadata.get("filename"),
                    "content": content_lines,
                })
            retrieved_all.append(retrieved_list)
            # -------------------------------------------

            # 기존 answer/info 처리 유지
            if isinstance(res, tuple):
                ans, info = res
            else:
                ans, info = res, None

            if hasattr(ans, "content"):
                ans = ans.content
            elif isinstance(ans, dict):
                ans = ans.get("content") or ans.get("text") or ans.get("answer") or str(ans)
            ans = "" if ans is None else str(ans)

            preds.append(ans)
            infos.append(info)

        self._infos[mode] = infos
        self._retrieved[mode] = retrieved_all  # ✅ mode별로 캐시
        return preds

    def save_qag(self,mode: str, qa_ids: list, questions: list,
                references: list, generated: list, metrics: dict, output_path: str):
        records = []
        infos = self._infos.get(mode, [None] * len(questions))
        n = len(questions)

        for i in range(n):
            rec = {"id": qa_ids[i], "question": questions[i], 
                    "reference": references[i], "generated": generated[i],}

            # metrics는 {"rouge1":[...], "sbert":[...]} 형태일 수도 있으니 방어적으로 처리
            for m_name, m_vals in (metrics or {}).items():
                rec[m_name] = float(m_vals[i])

            if infos[i] is not None:
                rec["info"] = infos[i]

            records.append(rec)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"[MODE={mode.upper()}] Detailed ../results saved to {output_path}")

    def save_retrieved(self, mode: str, qa_ids: list, output_path: str):
        """
        mode에 대해 질문 id별로 retrieved 결과 저장.
        self._retrieved[mode]는 질문 개수만큼의 리스트이며,
        각 원소는 [{"rank","filename","content"}, ...]
        """
        retrieved_all = self._retrieved.get(mode, [])
        n = len(qa_ids)

        records = []
        for i in range(n):
            records.append({
                "id": qa_ids[i],
                "retrieved": retrieved_all[i] if i < len(retrieved_all) else []
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"[MODE={mode.upper()}] Retrieved docs saved to {output_path}")


def run_qag_and_save(
    *,
    qa_mode: str,
    retriever_mode: str,
    selected_modes: list[str],
    qa_data_path: dict,
    results_dir: Path,
    sample_size=None,
    ndocs_init: int = 10,
    k_each: int = 10,
    top_k: int = 7,
    subdir: str = "",  # 예: "manual_book" 또는 "test" (없으면 루트에 저장)
):
    """
    - qa_data_path[qa_mode]의 GT json을 로드
    - cleaned retriever 구성
    - selected_modes 별로 Expert 생성
    - QagOnly로 QAG 생성 및 results_dir/{qag|retrieved}/... 저장

    저장 위치:
      results_dir / "qag" / subdir / f"qag_{mode}.json"
      results_dir / "retrieved" / subdir / f"retrieved_{mode}.json"
    """

    # 1) QA 데이터 로드 (파일 유효성 체크 겸)
    qa_file = qa_data_path[qa_mode]
    if not Path(qa_file).exists():
        raise FileNotFoundError(f"GT file not found: {qa_file}")

    with open(qa_file, "r", encoding="utf-8") as f:
        _merged_items = json.load(f)  # 지금 코드에서는 실제로 안 쓰지만 유지

    # Evaluator가 단일 JSON 파일로 인식하도록 유지 (원 코드 그대로 의미 보존)
    qa_data_path[qa_mode] = qa_file

    # 2) retriever 세팅
    cleaned_folder_retrievers = load_cleaned_md_level2_retrievers(ndocs=ndocs_init)
    cleaned_multi = MultiCosineRetriever(
        retrievers=cleaned_folder_retrievers,
        k_each=k_each,
        top_k=top_k
    )
    retr_map = {"cleaned_multi": cleaned_multi}

    # 3) 저장 폴더 준비
    qag_dir = results_dir / "qag"
    ret_dir = results_dir / "retrieved"
    if subdir:
        qag_dir = qag_dir / subdir
        ret_dir = ret_dir / subdir
    qag_dir.mkdir(parents=True, exist_ok=True)
    ret_dir.mkdir(parents=True, exist_ok=True)

    # 4) 모드별 실행
    for mode in selected_modes:
        if mode.startswith("partial_"):
            expert = PartialExpert(retr_map, retriever_mode)
        elif mode.startswith("selfask_"):
            k = int(mode.split("_")[1])
            expert = SelfAskExpert(retr_map, retriever_mode, max_iter=k)
        elif mode == "raw_llm":
            expert = RawLlmExpert(retr_map, retriever_mode)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        qag = QagOnly(expert, qa_data_path=qa_data_path[qa_mode], sample_size=sample_size)

        # ✅ get_data(infer.py) -> generate(base.py) -> handle(base.py) -> invoke(rag_naive.py)
        questions, references, generated, qa_id = qag.get_data(mode)

        out_path = qag_dir / f"qag_{mode}.json"
        qag.save_qag(mode, qa_id, questions, references, generated, {}, str(out_path))

        retr_path = ret_dir / f"retrieved_{mode}.json"
        qag.save_retrieved(mode, qa_id, str(retr_path))


if __name__ == '__main__':
    start_time = time.time()
    results_dir = (Path(__file__).resolve().parent / ".." / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    qa_mode = "cleaned"
    retriever_mode = "cleaned_multi"
    selected_modes = ["partial_10", "raw_llm"]
    sample_size = None

    qa_data_path = {
        "cleaned": "qa_data/GT/manual_book/gt_merged_manual_book.json"
    }

    run_qag_and_save(
        qa_mode=qa_mode,
        retriever_mode=retriever_mode,
        selected_modes=selected_modes,
        qa_data_path=qa_data_path,
        results_dir=results_dir,
        sample_size=sample_size,
        subdir="",  # manual_book을 루트에 쓰고 싶으면 "", 아니면 "manual_book"
    )

    qa_data_path = {
        "cleaned": "qa_data/GT/farm_consulting/gt_merged_farm_consulting.json"
    }

    run_qag_and_save(
        qa_mode=qa_mode,
        retriever_mode=retriever_mode,
        selected_modes=selected_modes,
        qa_data_path=qa_data_path,
        results_dir=results_dir,
        sample_size=sample_size,
        subdir="",  # manual_book을 루트에 쓰고 싶으면 "", 아니면 "manual_book"
    )
    
    '''
    # test
    qa_data_path = {
        "cleaned": "qa_data/GT/test/gt_merged_test.json"
    }

    run_qag_and_save(
        qa_mode=qa_mode,
        retriever_mode=retriever_mode,
        selected_modes=selected_modes,
        qa_data_path=qa_data_path,
        results_dir=results_dir,
        sample_size=sample_size,
        subdir="",  # manual_book을 루트에 쓰고 싶으면 "", 아니면 "manual_book"
    )
    '''
    
    elapsed = time.time() - start_time
    print(f"⏱ 전체 추론 완료: {elapsed/60:.2f}분")
