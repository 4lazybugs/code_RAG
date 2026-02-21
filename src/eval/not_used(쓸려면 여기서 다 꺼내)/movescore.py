from .base import BaseEvaluator
import re
import math
import numpy as np
import torch
from collections import Counter

class MoverEvaluator(BaseEvaluator):
    metric_key = 'mover'
    _tok_pat = re.compile(r"\w+|\S", flags=re.UNICODE)

    def _tokenize(self, text: str):
        return self._tok_pat.findall(text.lower())

    def _get_idf_dict(self, texts):
        n = len(texts)
        df = Counter()
        for t in texts:
            df.update(set(self._tokenize(t)))
        return {w: math.log((n + 1) / (df[w] + 1)) for w in df}

    def _embed_tokens(self, tokens):
        if not tokens:
            return np.zeros((1, self.sbert_model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.sbert_model.encode(tokens, convert_to_numpy=True, show_progress_bar=False)

    def _sinkhorn_wasserstein(self, w_h, w_r, C, eps=0.1, n_iter=100):
        device = torch.device("cpu")
        a = torch.from_numpy(w_h).to(device)
        b = torch.from_numpy(w_r).to(device)
        M = torch.from_numpy(C).to(device)
        K = torch.exp(-M / eps)
        u = torch.ones_like(a) / a.size(0)
        v = torch.ones_like(b) / b.size(0)
        for _ in range(n_iter):
            u = a / (K @ v + 1e-12)
            v = b / (K.t() @ u + 1e-12)
        P = torch.diag(u) @ K @ torch.diag(v)
        dist = (P * M).sum().item()
        return dist

    def _one_pair_score(self, ref, hyp, idf_ref, idf_hyp, stop_words=None):
        r_toks = self._tokenize(ref)
        h_toks = self._tokenize(hyp)
        if stop_words:
            r_toks = [t for t in r_toks if t not in stop_words]
            h_toks = [t for t in h_toks if t not in stop_words]
        if len(r_toks) == 0 or len(h_toks) == 0:
            return 0.0

        r_emb = self._embed_tokens(r_toks)
        h_emb = self._embed_tokens(h_toks)
        C = np.linalg.norm(h_emb[:, None, :] - r_emb[None, :, :], ord=2, axis=-1)
        w_r = np.array([idf_ref.get(t, 1.0) for t in r_toks], dtype=np.float64)
        w_h = np.array([idf_hyp.get(t, 1.0) for t in h_toks], dtype=np.float64)
        w_r = w_r / (w_r.sum() + 1e-12)
        w_h = w_h / (w_h.sum() + 1e-12)
        dist = self._sinkhorn_wasserstein(w_h, w_r, C, eps=0.1, n_iter=50)
        return 1.0 - float(dist)

    def compute_scores(self, references: list, generated: list) -> list:
        idf_ref = self._get_idf_dict(references)
        idf_hyp = self._get_idf_dict(generated)
        return [self._one_pair_score(r, g, idf_ref, idf_hyp) for r, g in zip(references, generated)]
