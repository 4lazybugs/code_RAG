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

def make_pretty(html: str) -> list[str]:
    """
    <tr>...</tr> 블록을 예쁘게 정리해서 반환
    """
    trs = re.findall(r"<tr.*?>.*?</tr>", html, flags=re.DOTALL)
    pretty = []

    for tr in trs:
        lines = [
            line.strip()
            for line in tr.splitlines()
            if line.strip()
        ]
        pretty.append("\n".join(lines))

    return pretty


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


if __name__ == '__main__':
    start_time = time.time()
    results_dir = (Path(__file__).resolve().parent / ".." / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) 파라미터 입력
    qa_mode = 'cleaned'
    retriever_mode = 'cleaned_multi'

    selected_modes = ['partial_10', 'raw_llm']
    selected_metrics = ['rouge1','rougeL','bert','sbert','bleu']
    #selected_metrics = ['sbert']
    sample_size = None

    qa_data_path = {'cleaned': 'qa_data/GT/farm_consulting/'}

    # --------------------------------------------------
    # ✅ [추가] GT 폴더(재귀) -> 단일 JSON으로 병합 (Evaluator 수정 없이)
    # -------------------------------------------------
    def _load_json_any(fp: Path):
        data = json.load(open(fp, "r", encoding="utf-8"))
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        if isinstance(data, list):
            return data
        return [data]

    gt_root = Path(qa_data_path[qa_mode])
    merged_items = []
    for fp in gt_root.rglob("*.json"):          # 필요하면 "*.jsonl"도 추가해서 처리
        merged_items.extend(_load_json_any(fp))

    # ✅ id(숫자) 기준으로 정렬
    merged_items.sort(key=lambda x: x["id"])

    merged_path = gt_root.parent / "_merged_gt.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_items, f, ensure_ascii=False, indent=2)

    qa_data_path[qa_mode] = str(merged_path)    # ✅ 이후 Evaluator는 “단일 파일”로 인식
    # --------------------------------------------------

    # 2) 초기 retriever 세팅
    ndocs_init = 10
    cleaned_folder_retrievers = load_cleaned_md_level2_retrievers(ndocs=ndocs_init)
    cleaned_multi = MultiCosineRetriever(retrievers=cleaned_folder_retrievers, k_each=10, top_k=7)

    retr_map = {"cleaned_multi": cleaned_multi}
    summary = []

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
        # ✅ get_data(infer.py)호출 -> generate(base.py)호출 -> handle(base.py)호출 -> invoke(rag_naive.py)호출
        questions, references, generated, qa_id = qag.get_data(mode)

        out_path = results_dir / "qag" / f"qag_{mode}.json"
        qag.save_qag(mode, qa_id, questions, references, generated, {}, str(out_path))

        retr_path = results_dir / "retrieved" / f"retrieved_{mode}.json"
        qag.save_retrieved(mode, qa_id, str(retr_path))

    elapsed = time.time() - start_time
    print(f"⏱ 전체 추론 완료: {elapsed/60:.2f}분")
