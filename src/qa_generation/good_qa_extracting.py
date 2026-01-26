import json

# 1. score 기반 id 로딩
with open("results/eval_score/mcq/qwen/naive_rag_mcq/em.json", "r", encoding="utf-8") as f:
    data = json.load(f)

threshold = 0

# sbert >= threshold 인 id만 추출
selected_ids = [
    item["id"]
    for item in data
    if item.get("rouge1", 0) >= threshold
]

# 2. 원본 데이터 로딩
with open(
    "results/inferenced/manual_book/mcq/naive_rag.json",
    "r",
    encoding="utf-8"
) as f:
    original_data = json.load(f)

# 3. id 기준 필터링
filtered_data = [
    item
    for item in original_data
    if item.get("id") in selected_ids
]

# 4. 새 파일로 저장
output_path = "qa_data/mcq_filtered_em_rag.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(filtered_data)} samples to {output_path}")
