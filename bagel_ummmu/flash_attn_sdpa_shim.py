# -*- coding: utf-8 -*-
"""
Drop-in replacement for flash-attn's `flash_attn_varlen_func`, built on
torch.nn.functional.scaled_dot_product_attention (SDPA).

Why: on some environments (e.g. molab containers: Python 3.13 + torch cu130,
no nvcc) flash-attn has no prebuilt wheel and cannot be compiled. BAGEL only
uses `flash_attn_varlen_func`, whose semantics can be reproduced exactly with
SDPA, which ships fused attention kernels inside torch itself.

Semantics matched:
- varlen packing via cu_seqlens_q / cu_seqlens_k (loop over segments)
- GQA/MQA: fewer kv heads than query heads
- causal masking with flash-attn's BOTTOM-RIGHT alignment when
  seqlen_q != seqlen_k (query i attends to keys <= i + (len_k - len_q));
  note torch's is_causal flag uses top-left alignment, so an explicit mask
  is built for that case.

Install with `install_shim()` BEFORE any module does `from flash_attn import ...`.
"""

import sys
import types

import torch
import torch.nn.functional as F

# One-slot mask cache: within a single forward pass all layers request the
# same (len_q, len_k) mask, so this avoids rebuilding it ~28 times.
_MASK_CACHE = {"key": None, "mask": None}


def _bottom_right_causal_mask(len_q: int, len_k: int, device) -> torch.Tensor:
    key = (len_q, len_k, str(device))
    if _MASK_CACHE["key"] == key:
        return _MASK_CACHE["mask"]
    idx_q = torch.arange(len_q, device=device)
    idx_k = torch.arange(len_k, device=device)
    mask = idx_k[None, :] <= (idx_q[:, None] + (len_k - len_q))
    _MASK_CACHE["key"] = key
    _MASK_CACHE["mask"] = mask
    return mask


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    """
    q: [total_q, num_heads, head_dim]
    k, v: [total_k, num_kv_heads, head_dim]
    cu_seqlens_q / cu_seqlens_k: int32 cumulative lengths, shape [batch + 1]
    Returns: [total_q, num_heads, head_dim] (same as flash-attn)
    """
    out = torch.empty_like(q)
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]

    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()

    for i in range(len(cu_q) - 1):
        qs, qe = cu_q[i], cu_q[i + 1]
        ks, ke = cu_k[i], cu_k[i + 1]
        len_q, len_k = qe - qs, ke - ks
        if len_q == 0:
            continue

        # [1, heads, len, dim]
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        if num_kv_heads != num_heads:
            rep = num_heads // num_kv_heads
            ki = ki.repeat_interleave(rep, dim=1)
            vi = vi.repeat_interleave(rep, dim=1)

        attn_mask = None
        use_is_causal = False
        if causal:
            if len_q == len_k:
                use_is_causal = True  # top-left == bottom-right when square
            elif len_q == 1:
                pass  # single query attends to all keys under bottom-right causal
            else:
                attn_mask = _bottom_right_causal_mask(len_q, len_k, q.device)

        oi = F.scaled_dot_product_attention(
            qi, ki, vi,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=use_is_causal,
            scale=softmax_scale,
        )
        out[qs:qe] = oi.squeeze(0).transpose(0, 1)

    return out


def install_shim() -> None:
    """Register a fake `flash_attn` module in sys.modules (overwrites any
    broken/partial installation already imported)."""
    import importlib.machinery

    mod = types.ModuleType("flash_attn")
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    mod.__version__ = "0.0.0+sdpa-shim"
    # A valid spec is required: libraries (e.g. transformers) probe for
    # flash-attn with importlib.util.find_spec, which raises ValueError on
    # modules whose __spec__ is None. With a spec but no dist metadata,
    # transformers correctly treats flash-attn as unavailable, while direct
    # `from flash_attn import flash_attn_varlen_func` still gets the shim.
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    sys.modules["flash_attn"] = mod


def real_flash_attn_available() -> bool:
    try:
        from flash_attn import flash_attn_varlen_func as _f  # noqa: F401
        return True
    except Exception:
        return False
