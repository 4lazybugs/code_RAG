from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import math

from .base import BaseModel


@dataclass
class tokGate_agent(BaseModel):
    """
    Z-score 기반 RAG routing agent.

    Z = (μ_qc - μ_q) / (σ_q + ε)

    * μ_qc : context 포함 시 answer 토큰들의 평균 logprob
    * μ_q  : context 없이 질문만 줬을 때 answer 토큰들의 평균 logprob
    * σ_q  : no-RAG 상태의 logprob 표준편차 (baseline noise)
    * ε    : σ_q 가 0 에 가까울 때 overflow 방지

    Z 가 tok_thre 이상이면 RAG 답변 채택, 미만이면 SOTA fallback.
    """

    RAG_agent: Any
    LLM_agent: Any = None 
    SOTA_agent: Any = None

    tok_thre: float = 1.0       # Z-score threshold (default 1σ)
    epsilon: float = 1e-6       # numerical stability for σ denominator
    n_samples: int = 1          # no-RAG 추정을 위한 샘플 수 (>1 이면 σ 더 안정적)

    # ------------------------------------------------------------------ #
    #  internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _mean_std(self, logprobs: List[float]):
        """
        토큰 logprob 리스트 → (mean, std).
        logprobs 가 비어있으면 (0.0, 0.0) 반환.
        """
        n = len(logprobs)
        if n == 0:
            return 0.0, 0.0

        mu = sum(logprobs) / n
        if n == 1:
            return mu, 0.0

        variance = sum((x - mu) ** 2 for x in logprobs) / (n - 1)  # sample var
        return mu, math.sqrt(variance)

    def _extract_logprobs(self, agent_output: Dict[str, Any]) -> List[float]:
        """
        agent 출력 딕셔너리에서 토큰 logprob 리스트를 꺼낸다.

        표준 키 후보:
            output_dic["logprobs"]                    # flat list[float]
            output_dic["token_logprobs"]              # flat list[float]
            output_dic["generation"]["logprobs"]      # nested
        """
        for key in ("logprobs", "token_logprobs"):
            val = agent_output.get(key)
            if isinstance(val, list) and len(val) > 0:
                return [float(v) for v in val if v is not None]

        gen = agent_output.get("generation", {})
        if isinstance(gen, dict):
            for key in ("logprobs", "token_logprobs"):
                val = gen.get(key)
                if isinstance(val, list) and len(val) > 0:
                    return [float(v) for v in val if v is not None]

        return []

    # ------------------------------------------------------------------ #
    #  core score                                                          #
    # ------------------------------------------------------------------ #

    def tok_z_score(
        self,
        logprobs_with_ctx: List[float],
        logprobs_no_ctx: List[float],
    ) -> Dict[str, float]:
        """
        Z = (μ_qc - μ_q) / (σ_q + ε)

        Returns
        -------
        dict with keys: mu_qc, mu_q, sigma_q, z_score
        """
        mu_qc, _ = self._mean_std(logprobs_with_ctx)
        mu_q, sigma_q = self._mean_std(logprobs_no_ctx)

        z = (mu_qc - mu_q) / (sigma_q + self.epsilon)

        print(
            f"[tokGate] μ_qc={mu_qc:.4f}  μ_q={mu_q:.4f}  "
            f"σ_q={sigma_q:.4f}  Z={z:.4f}"
        )

        return {
            "mu_qc": mu_qc,
            "mu_q": mu_q,
            "sigma_q": sigma_q,
            "z_score": z,
        }

    def _get_no_rag_logprobs(self, raw_input: Dict[str, Any]) -> List[float]:
        if self.LLM_agent is None:
            return []

        # gate_llm_output이 judge_score를 요구하므로 없으면 임시로 추가
        safe_input = dict(raw_input)
        if "judge_score" not in safe_input:
            safe_input["judge_score"] = [{"know_prob": None, "relv_score": None, "faith_score": None}]

        all_logprobs: List[float] = []
        for _ in range(self.n_samples):
            out = self.LLM_agent.answer_once(safe_input)
            all_logprobs.extend(self._extract_logprobs(out))
        return all_logprobs

    def answer_once(self, raw_input: Dict[str, Any] = None) -> Dict[str, Any]:
        # 1. RAG 답변
        output_dic = self.RAG_agent.answer_once(raw_input)
        logprobs_qc = self._extract_logprobs(output_dic)

        # 2. no-context baseline (LLM_agent)
        logprobs_q = self._get_no_rag_logprobs(raw_input)

        # 3. Z-score
        z_info = self.tok_z_score(logprobs_qc, logprobs_q)
        z_score = z_info["z_score"]

        if "judge_score" not in output_dic or not output_dic["judge_score"]:
            output_dic["judge_score"] = [{}]
        output_dic["judge_score"][0]["tok_z_score"] = z_score
        output_dic["judge_score"][0].update(
            {f"tok_{k}": v for k, v in z_info.items()}
        )

        # 4. Routing
        if z_score >= self.tok_thre:
            print(f"==== tokGate Good  (Z={z_score:.4f} ≥ {self.tok_thre}) → RAG 반환 =======")
            return output_dic

        print(f"==== tokGate Bad   (Z={z_score:.4f} < {self.tok_thre}) → SOTA fallback =======\n")
        if self.SOTA_agent is not None:
            return self.SOTA_agent.answer_once(output_dic)
        return output_dic