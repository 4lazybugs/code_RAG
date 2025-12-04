import json
import numpy as np

from models_eval import (
    Rouge1Evaluator, RougeLEvaluator, BertEvaluator,
    SbertEvaluator, MoverEvaluator, BleurtEvaluator, BleuEvaluator
)

# 필요하면 여기서 골라 쓰면 됨
METRIC_CLASS_MAP = {
    'rouge1': Rouge1Evaluator,
    'rougeL': RougeLEvaluator,
    'bert':   BertEvaluator,
    'sbert':  SbertEvaluator,
    'mover':  MoverEvaluator,
    'bleurt': BleurtEvaluator,
    'bleu':   BleuEvaluator
}


def compute_metrics_from_lists(references, generated, metrics=None):
    assert len(references) == len(generated), "refs / gens 길이가 다름"

    if metrics is None:
        metrics = ['rouge1', 'rougeL', 'bert', 'sbert', 'bleu']

    summary = []

    for metric in metrics:
        EvCls = METRIC_CLASS_MAP[metric]

        # BaseEvaluator는 expert, qa_data_path를 받게 되어 있지만
        # 여기서는 generate/get_data 안 쓸 거라 dummy 값 넣어도 됨
        ev = EvCls(expert=None, qa_data_path="", sample_size=None)

        # 핵심: 이미 가지고 있는 refs / gens를 그대로 넣어서 점수만 계산
        scores = ev.compute_scores(references, generated)

        mean_val = round(float(np.mean(scores)), 3)
        std_val  = round(float(np.std(scores)), 3)

        summary.append({
            'metric': metric,
            'mean':   mean_val,
            'std':    std_val,
            'scores': scores,   # 원하면 per-sample 점수도 같이 보관
        })

    return summary


# ---------------- 사용 예시 ----------------
if __name__ == "__main__":
    # 예: 네가 이미 갖고 있는 json에서 직접 뽑아온다고 치면
    # records = json.load(open("results/partial_10_rouge1.json", "r", encoding="utf-8"))
    # references = [r["reference"] for r in records]
    # generated  = [r["generated"] for r in records]

    references = [
                "내부 온습도 센서 3EA 점검 및 센서 교체와 센서 통신 상태 점검 및 팜모닝 시스템 정상 동작 확인을 진행하였습니다."
    ]
    generated = [
                "제공된 정보에 따르면, 2022년 스마트팜 A/S 컨설팅에서 구례 김건수 농가에 대한 유지보수 내용이 명시적으로 언급되어 있지 않습니다. 주로 구례군 간전면 간문리에 위치한 김건수 농가에 대한 컨설팅 내용이 기술되어 있으며, 이는 팜모닝 내외부 온습도 센서 그래프를 통한 컨설팅 진행으로 표고버섯 자실체 생장 최적온도 범위와 상대습도에 대한 제시가 포함되었습니다.\n\n하지만 구례 김건수 농가의 유지보수 내용은 별도로 언급되지 않았습니다. 만약 구례 김건수 농가의 유지보수 내용을 확인하고자 한다면, 보고서나 관련 문서에서 더 자세한 정보를 찾아볼 필요가 있습니다.\n\n만약 다른 농가에 대한 유지보수 내용이 궁금하신 경우, 해당 농가명과 일시를 제공해주시면 더욱 구체적인 답변을 드릴 수 있을 것 같습니다."
    ]

    metrics = ['rouge1', 'rougeL', 'bert', 'sbert', 'bleu']
    summary = compute_metrics_from_lists(references, generated, metrics)

    for item in summary:
        print(f"{item['metric']:7s} | mean={item['mean']:.3f} | std={item['std']:.3f}")
