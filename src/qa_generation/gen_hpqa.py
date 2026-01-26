'''
hotpotqa generation script
구조) context 추출 -> 질문과 답변을 조합하여 HotpotQA 생성
1. target_dir안을 순회하면서 모든 md파일에 대하여 search_dir안의 모든 md파일과 ROUGE-1 비교
2. 가장 높은 점수를 가진 md파일을 선택
3. 선택된 md파일의 경로를 반환
...
'''
from pathlib import Path
from typing import Optional, Tuple, List
from eval.models_eval.rouge1 import Rouge1Evaluator  # ← 기존 코드 import
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

def get_best_rouge1(
    *,
    target_dir: Path,
    search_dir: Path,
    min_len: int = 200,
) -> List[Tuple[Path, Optional[Path], float]]: # [(tar_md1, best_md1, best_score1), (tar_md2, best_md2, best_score2), ...]
    evaluator = Rouge1Evaluator()
    results: List[Tuple[Path, Optional[Path], float]] = []

    # 1) target_dir
    idx = 0
    for ref_md in target_dir.rglob("*.md"):
        idx += 1
        ref_doc = read_md_text(ref_md)
        if len(ref_doc) < min_len:
            continue

        highest_score: float = -1.0
        best_md: Optional[Path] = None

        # 2) search_dir 전체 스캔 (target_dir 제외)
        
        for md in search_dir.rglob("*.md"):
            if target_dir in md.parents:
                continue  # 자기 자신 제외

            comparison_doc = read_md_text(md)
            if len(comparison_doc) < min_len:
                continue

            score = evaluator.compute_scores(
                references=[ref_doc],
                generated=[comparison_doc],
                gen_docs=None,
                ref_docs=None,
            )[0]

            if score > highest_score:
                highest_score = score
                best_md = md
                print("======================best md updated===================================")
                print(f"best_md: {best_md.stem}, score: {highest_score} for {idx}th md")


        results.append((ref_md, best_md, float(highest_score)))

    return results

def read_md_text(p: Path) -> str:
    if p is None:
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

######################  TO JSON  ###################################
# results to json
def results_to_json(results, output_path: Path):
    records = []

    for idx, (ref_md, best_md, score) in enumerate(results, start=1):
        ref_content = read_md_text(ref_md)
        best_content = read_md_text(best_md) if best_md is not None else ""

        records.append({
            "id": idx,
            "ref_md": str(ref_md),
            "ref_content": ref_content,
            "best_md": str(best_md) if best_md is not None else None,
            "best_content": best_content,
            "rouge1_score": float(score),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {output_path}")

# qag_list to json
def qag_to_json(qag_list, output_path: Path):
    cleaned: List[Dict[str, Any]] = []

    for item in qag_list:
        rec = {
            "id": int(item["id"]),
            "link_word": str(item["link_word"]).strip(),
            "question": str(item["question"]).strip(),
            "answer": str(item["answer"]).strip(),
            "supporting_fact_ref": str(item["supporting_fact_ref"]).strip(),
            "supporting_fact_comp": str(item["supporting_fact_comp"]).strip(),
        }

        cleaned.append(rec)

    # id 기준 정렬(원하면 제거 가능)
    cleaned.sort(key=lambda x: x["id"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {output_path} (n={len(cleaned)})")


hotpot_prompt = ChatPromptTemplate.from_template(
"""
다음 context에서 **HotpotQA 스타일 multi-hop RAG 평가용 Ground Truth 1개 샘플**을
JSON 배열로 생성하라.

# 0) 절대 원칙
- question/answer/link_word/supporting_fact는 **context의 원문에 명시된 정보만** 사용한다.
- 질문은 **정확히 1개**로만 생성한다.
- 출력은 반드시 **JSON 배열만** 출력한다(추가 설명 금지).

# 1) Multi-hop 강제 규칙 (가장 중요)
- 질문은 반드시 **문단 A + 문단 B를 함께 읽어야만** 답할 수 있어야 한다.
- 단일 문단만으로 답이 나오면 실패다.

# 2) link_word 정의 및 강제 규칙 (supporting fact와 다름)
- link_word는 **두 문단 A와 B에 동시에 등장**하는 단어/구/고유명칭(bridge entity)이다.
- link_word는 두 문단을 연결하는 열쇠이며, supporting fact와 별도 필드로 출력한다.
- link_word는 문단 A/B에서 **표기(철자)가 동일**해야 한다(띄어쓰기 차이는 허용 가능).
- link_word가 두 문단에 동시에 존재하지 않으면 샘플 생성 실패 → 다른 pair를 찾아 재작성한다.

# 3) Supporting facts 정의 및 강제 규칙
- supporting_fact_ref: **문단 A**에서 답을 얻는 데 필요한 핵심 근거 1개를 원문 그대로 발췌한다.
- supporting_fact_comp: **문단 B**에서 답을 얻는 데 필요한 핵심 근거 1개를 원문 그대로 발췌한다.
- 각 supporting fact는 해당 문단에 실제로 존재하는 **완전한 문장 또는 의미가 자족적인 핵심 구(ngram)** 여야 한다.
- 각 supporting fact는 **문맥을 단독으로 이해할 수 있을 만큼 충분히 길어야 한다.**
  - 권장 기준: **최소 200 tokens 이상**
- 지나치게 짧아 지시어(예: “그는”, “해당”, “이것”)만 포함되거나,
  문맥 없이 의미가 불완전한 구는 허용하지 않는다.
- supporting_fact_ref에는 **문단 A의 내용만** 포함해야 하며,
  supporting_fact_comp에는 **문단 B의 내용만** 포함해야 한다.
- 두 supporting fact의 내용을 **하나의 필드에 혼합하는 것은 절대 금지**한다.

# 4) 모호성(ambiguity) 금지 규칙
- 질문에는 고유 식별자(정식 명칭/기관명/품목명/프로젝트명 등)가 포함되어야 한다.
- 지시어/대명사 금지: “이것, 그것, 해당, 본 문서, 여기, 거기”
- 일반명사 단독 금지: “장비, 시스템, 자료, 내용, 현황”
  (단, 뒤에 고유명칭이 붙어 구체화되는 경우만 허용)

# 5) 생성 절차 (필수)
- Step A: context에서 **서로 다른 두 문단(A,B)** 을 선택한다.
- Step B: 두 문단에 **동시에 등장하는 link_word** 1개를 찾는다.
- Step C: 문단 A에서 supporting_fact_ref 1개를 원문 그대로 발췌한다.
- Step D: 문단 B에서 supporting_fact_comp 1개를 원문 그대로 발췌한다.
- Step E: A와 B를 연결해야만 답이 나오는 multi-hop 질문 1개를 생성한다.
- Step F: 답을 단일 값으로 작성한다.

# 6) 자체 검증 체크 (통과 못 하면 재작성)
- [ ] link_word가 문단 A와 문단 B에 **모두 존재**하는가?
- [ ] supporting_fact_ref는 문단 A의 원문 발췌 1개인가?
- [ ] supporting_fact_comp는 문단 B의 원문 발췌 1개인가?
- [ ] 질문은 두 문단을 모두 사용해야만 답이 나오는가?
- [ ] 답변은 context에 명시된 단일 값인가?

# 7) 출력 형식 (매우 중요)
- 아래 JSON 배열 형식만 출력한다.
- 설명/주석/자연어 문장 절대 금지.
- key는 반드시 아래 5개만 사용한다:
  "id", "link_word", "question", "answer", "supporting_fact_ref", "supporting_fact_comp"

[
  {{
    "id": {id},
    "link_word": "두 문단에 동시에 등장하는 연결 단어/구(원문 그대로)",
    "question": "multi-hop 질문",
    "answer": "질문에 대한 최종 정답(짧은 명칭/수치/값 또는 yes/no)"
    "supporting_fact_ref": "문단 A의 핵심 근거 1개(원문 그대로 발췌)",
    "supporting_fact_comp": "문단 B의 핵심 근거 1개(원문 그대로 발췌)"
  }}
]

# context
{context_ref}
{context_comp}
"""
)

def gen_qag(context_ref: str, context_comp: str, chain, id: int) -> List[dict]:
    resp = chain.invoke({"context_ref": context_ref, "context_comp": context_comp, "id": id})

    # LangChain 메시지(AIMessage)든 문자열이든 처리
    content = resp.content if hasattr(resp, "content") else str(resp)

    qas_list: List[Dict] = []

    try:
        arr = json.loads(content)
    except Exception:
        return []

    for item in arr:
        qas_list.append({
            "id": id,
            "link_word": item["link_word"],
            "question": item["question"],
            "answer": item["answer"],
            "supporting_fact_ref": item["supporting_fact_ref"],
            "supporting_fact_comp": item["supporting_fact_comp"],
        })

    return qas_list

################################  MAIN  #############################################################
if __name__ == "__main__":
    # llm 초기화
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = hotpot_prompt | llm

    # 경로 설정
    target_dir = Path("db/cleaned_md/manual_book/[OCR]strawberry_standart_protocl.pdf")
    search_dir = Path("db/cleaned_md/manual_book/")
    output_path = Path("db/post_processed_md/hotpotqa_test/all_results.json")
    qag_path = Path("qa_data/GT/hotpotqa_test/gt_merged_hotpotqa_test.json")

    #################### 2개의 Pargraph 추출 #####################################################
    # (tar_md_i, best_md_i, best_score_i)= result[i]
    results = get_best_rouge1(
        target_dir=target_dir,
        search_dir=search_dir,
        min_len=300,
    )

    results_to_json(results, output_path=output_path)

    #################### HotPotQA 생성 ########################################################
    # retrieval 결과 JSON 로드
    results_path = output_path
    with open(results_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 3) JSON 기반 QA 생성
    all_list = []
    for rec in records:
        # qas_list = [{id:, link_word:, question:, answer:, supporting_fact_ref:, supporting_fact_comp:},{...}...]
        qag_list = gen_qag(
            id=rec["id"],
            context_ref=rec["ref_content"],
            context_comp=rec["best_content"],
            chain=chain,
        )
        all_list.extend(qag_list)
    
    qag_to_json(
        all_list,
        output_path=qag_path,
    )