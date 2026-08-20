# -*- coding: utf-8 -*-
"""Numerical verification of flash_attn_sdpa_shim against naive attention.

Run: python test_shim.py  (CPU is fine; no flash-attn needed)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from flash_attn_sdpa_shim import flash_attn_varlen_func

torch.manual_seed(0)


def naive_ref(q, k, v, cu_q, cu_k, causal):
    nh, nkv = q.shape[1], k.shape[1]
    rep = nh // nkv
    out = torch.zeros_like(q)
    for i in range(len(cu_q) - 1):
        qs, qe, ks, ke = cu_q[i], cu_q[i + 1], cu_k[i], cu_k[i + 1]
        lq, lk = qe - qs, ke - ks
        qi = q[qs:qe].float()
        ki = k[ks:ke].float().repeat_interleave(rep, dim=1)
        vi = v[ks:ke].float().repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qi, ki) / math.sqrt(q.shape[-1])
        if causal:
            iq = torch.arange(lq)[:, None]
            ik = torch.arange(lk)[None, :]
            mask = ik <= iq + (lk - lq)  # flash-attn bottom-right alignment
            scores = scores.masked_fill(~mask[None], float("-inf"))
        attn = scores.softmax(-1)
        out[qs:qe] = torch.einsum("hqk,khd->qhd", attn, vi).to(q.dtype)
    return out


def run_case(name, lens_q, lens_k, nh=8, nkv=2, d=32, causal=False):
    cu_q = [0]
    for l in lens_q:
        cu_q.append(cu_q[-1] + l)
    cu_k = [0]
    for l in lens_k:
        cu_k.append(cu_k[-1] + l)
    q = torch.randn(cu_q[-1], nh, d)
    k = torch.randn(cu_k[-1], nkv, d)
    v = torch.randn(cu_k[-1], nkv, d)
    got = flash_attn_varlen_func(
        q=q, k=k, v=v,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32),
        max_seqlen_q=max(lens_q), max_seqlen_k=max(lens_k), causal=causal,
    )
    ref = naive_ref(q, k, v, cu_q, cu_k, causal)
    err = (got - ref).abs().max().item()
    ok = err < 1e-4
    print(f"{name:45s} max_abs_err={err:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [
        run_case("full attn, square, 2 segments", [5, 9], [5, 9], causal=False),
        run_case("causal, square (prefill no cache)", [7, 4], [7, 4], causal=True),
        run_case("causal, lq<lk (prefill with cache)", [6, 3], [15, 10], causal=True),
        run_case("causal, decode lq=1", [1, 1], [20, 13], causal=True),
        run_case("full attn, lq<lk (image gen w/ context)", [16], [40], causal=False),
        run_case("MHA (nh==nkv)", [8], [8], nh=4, nkv=4, causal=True),
    ]
    sys.exit(0 if all(results) else 1)
