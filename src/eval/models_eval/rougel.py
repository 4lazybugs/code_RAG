from .base import BaseEvaluator

def _normalize(text: str) -> str:
    if text is None:
        return ""
    # 양끝 공백 제거 + 공백 제거(원하면 이 부분은 취향껏 조정)
    text = text.strip()
    # 필요하면 특수 공백 제거도 가능:
    # text = text.replace("\u200b", "").replace("\u00a0", " ")
    return text

def _lcs_len(a: str, b: str) -> int:
    """a, b: 공백 제거된 문자열(또는 토큰 시퀀스), LCS 길이 계산"""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    # DP 테이블: (la+1) x (lb+1)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[la][lb]

class RougeLEvaluator(BaseEvaluator):
    metric_key = 'rougeL'

    def __init__(self):
        # BaseEvaluator의 다른 기능(캐시, sbert 등) 쓰려면 유지
        super().__init__()

    def compute_scores(self, references: list, generated: list, gen_docs: list, ref_docs: list) -> list:
        scores = []
        for ref, gen in zip(references, generated):
            ref_norm = _normalize(ref).replace(" ", "")
            gen_norm = _normalize(gen).replace(" ", "")

            if not ref_norm and not gen_norm:
                scores.append(1.0)
                continue

            L = _lcs_len(ref_norm, gen_norm)
            # precision = L / len(gen_norm)
            # recall    = L / len(ref_norm)
            prec = L / len(gen_norm) if len(gen_norm) > 0 else 0.0
            rec  = L / len(ref_norm) if len(ref_norm) > 0 else 0.0

            if prec + rec == 0:
                f1 = 0.0
            else:
                f1 = 2 * prec * rec / (prec + rec)

            scores.append(float(f1))
        return scores
