from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

qa_pairs = [
    {
        "context": "2022년 12월 12일, 한국온실작물연구소 소속 서범석 컨설팅팀은 대형 단동 온실의 중앙부 환기가 불량하여 환경 차이에 의한 작물 개체 생육이 불균일하다고 보고하며, 천창을 수동으로 개폐해 환경 조절 시 작물이 정상적으로 균일한 생육을 이루도록 제안하였다. 김은상 농가의 딸기 스마트팜 컨설팅에서 육묘 온실의 내부 차광량 사용 계획과 두상 살수 장치의 변경이 논의되었다.",
        "answer":  "2022년 12월 12일 한국온실작물연구소 서범석 컨설팅팀은 김은상 농가 딸기 스마트팜을 방문하여 대형 단동 온실 중앙부의 환기 불량으로 작물 개체 생육 불균일 문제를 지적하고 천창 수동 개폐를 통한 균일 생육 유도를 제안하였으며, 육묘 온실의 내부 차광량 사용 계획과 두상 살수 장치 변경도 함께 논의하였다."
    }
]

def predict(context, answer):
    inputs = tokenizer(
        context,
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        # token_type_ids=None  ← 이거 제거
    )
    inputs.pop("token_type_ids", None)  # 결과물에서 제거

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = torch.argmax(probs).item()
    return LABEL_MAP[pred], round(probs[pred].item(), 4)

results, entailed = [], []
for i, pair in enumerate(qa_pairs):
    label, score = predict(pair["context"], pair["answer"])
    result = {"idx": i, **pair, "label": label, "score": score}
    results.append(result)
    if label == "entailment":
        entailed.append(result)

# 출력
print("=" * 60)
for r in results:
    mark = "✅" if r["label"] == "entailment" else ("❌" if r["label"] == "contradiction" else "⚪")
    print(f"[{r['idx']}] {mark} {r['label'].upper()} ({r['score']})")
    print(f"  Context: {r['context']}")
    print(f"  Answer : {r['answer']}\n")

print(f"Entailment 선택: {len(entailed)}/{len(results)} ({len(entailed)/len(results)*100:.1f}%)")