"""
Regional Cross-Attention Masking (RCAM)
=========================================
Implementation of the RCAM mechanism described in Section 2.3 of:

  "Spatial-Aware Fusion LoRA: Enhancing Consistency in Multi-Subject
   Image Generation via Parameter Decoupling and Regional Attention"

Core equation (Eq. 2):
    Attention(Q, K, V) = Softmax( (Q K^T / √d  +  M) · V )

Bias matrix M ∈ ℝ^{L×S}:
    M[i, j] = 0    if pixel i is in subject-n's region AND token j describes n,
                   OR if token j is a global/background token
    M[i, j] = -∞  otherwise  →  attention weight becomes 0 after Softmax

FLUX attention specifics
-------------------------
FLUX forms a joint image+text sequence for attention:
    q = cat(txt_q, img_q)  shape (B, H, S_txt + S_img, d_head)
    k = cat(txt_k, img_k)  shape (B, H, S_txt + S_img, d_head)

The raw attention matrix is (S_txt + S_img) × (S_txt + S_img).
RCAM acts only on the img-query × txt-key block [S_txt:, :S_txt],
which governs how each image token attends to text descriptions.
The other three blocks (txt→txt, img→img, txt→img) are left unconstrained.

Injection strategy
------------------
A context-manager (rcam_scope) registers the bias tensor before inference
and clears it on exit.  The bias is injected by monkey-patching
DoubleStreamBlock._forward and SingleStreamBlock._forward so no changes to
sd-scripts source are required.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Token attribution: maps text token indices to subject indices
# ---------------------------------------------------------------------------

GLOBAL_TOKEN = -1   # sentinel for background / style / layout tokens


class TokenAttribution:
    """
    Assigns each text-token position to a subject index (0-based) or GLOBAL.

    Parameters
    ----------
    total_tokens         : total number of text tokens (e.g. 256 for T5)
    subject_token_ranges : list of (start, end_exclusive) per subject,
                           e.g. [(2, 5), (8, 11)] for two subjects
    """

    def __init__(self,
                 total_tokens:         int,
                 subject_token_ranges: List[Tuple[int, int]]) -> None:
        self.total_tokens         = total_tokens
        self.subject_token_ranges = subject_token_ranges

        # token_owner[j] = subject index (≥0) or GLOBAL_TOKEN (-1)
        self.token_owner: List[int] = [GLOBAL_TOKEN] * total_tokens
        for subj, (start, end) in enumerate(subject_token_ranges):
            for j in range(start, min(end, total_tokens)):
                self.token_owner[j] = subj

    def owner(self, token_idx: int) -> int:
        """Return subject index or GLOBAL_TOKEN for text position j."""
        if token_idx >= self.total_tokens:
            return GLOBAL_TOKEN
        return self.token_owner[token_idx]


# ---------------------------------------------------------------------------
# Build M: the (B, H, S, S) additive bias tensor
# ---------------------------------------------------------------------------

def build_rcam_bias(
    spatial_masks:        List[Tensor],
    attribution:          TokenAttribution,
    img_seq_len:          int,
    txt_seq_len:          int,
    num_heads:            int,
    spatial_hw:           Optional[Tuple[int, int]] = None,
    device:               torch.device = torch.device("cpu"),
    dtype:                torch.dtype  = torch.float32,
) -> Tensor:
    """
    Construct the additive bias matrix M for the full FLUX joint attention.

    Only the image-row × text-column block carries non-zero values; the
    other three blocks remain 0 (unconstrained).

    Parameters
    ----------
    spatial_masks  : N masks, each (B, H_lat, W_lat) or (B, L_lat), float [0,1]
    attribution    : TokenAttribution mapping text positions to subjects
    img_seq_len    : S_img – number of image latent tokens at this resolution
    txt_seq_len    : S_txt – number of text tokens
    num_heads      : H – number of attention heads (bias is head-agnostic here)
    spatial_hw     : (H_feat, W_feat) at current resolution; inferred if None
    device, dtype  : target device / dtype

    Returns
    -------
    bias : (B, H, S_txt + S_img, S_txt + S_img)
    """
    NEG_INF = -1e9          # large negative constant (avoids NaN from true -inf)
    N = len(spatial_masks)
    B = spatial_masks[0].shape[0]
    S = txt_seq_len + img_seq_len

    bias = torch.zeros(B, num_heads, S, S, device=device, dtype=dtype)

    # ── 1. Infer feature spatial size ────────────────────────────────────────
    if spatial_hw is not None:
        H_f, W_f = spatial_hw
    else:
        s = int(math.isqrt(img_seq_len))
        assert s * s == img_seq_len, \
            f"Cannot infer spatial size from img_seq_len={img_seq_len}. " \
            f"Pass spatial_hw explicitly."
        H_f = W_f = s

    # ── 2. Resample all N masks to (B, img_seq_len) ──────────────────────────
    resized: List[Tensor] = []
    for mask in spatial_masks:
        if mask.dim() == 2:                           # (B, L_lat)
            L = mask.shape[1]
            h = w = int(math.isqrt(L))
            m = mask.float().reshape(B, 1, h, w)
        elif mask.dim() == 3:                         # (B, H_lat, W_lat)
            m = mask.float().unsqueeze(1)
        else:
            raise ValueError(f"Each mask must be 2-D or 3-D, got {mask.dim()}")

        m = F.interpolate(m, size=(H_f, W_f), mode="bilinear", align_corners=False)
        resized.append(m.reshape(B, img_seq_len))     # (B, L_img)

    # ── 3. Assign each image pixel to a subject (majority rule) ──────────────
    # pixel_owner[b, l] = subject index or -1
    pixel_owner = torch.full((B, img_seq_len), GLOBAL_TOKEN,
                             dtype=torch.long, device=device)
    for subj_idx, rm in enumerate(resized):
        pixel_owner[rm > 0.5] = subj_idx

    # ── 4. Fill the img-query × txt-key block ────────────────────────────────
    # M[b, :, txt_seq_len + img_row, txt_col] = NEG_INF
    # iff token j belongs to subject k AND pixel l is NOT in subject k's region
    for b in range(B):
        for j in range(txt_seq_len):
            owner_j = attribution.owner(j)
            if owner_j == GLOBAL_TOKEN:
                # Global / background token → no masking for any image pixel
                continue

            # Pixels NOT owned by the same subject as token j → block attention
            blocked = (pixel_owner[b] != owner_j)                # (img_seq_len,)
            row_indices = torch.nonzero(blocked, as_tuple=True)[0] + txt_seq_len
            bias[b, :, row_indices, j] = NEG_INF

    return bias     # (B, H, S, S)


# ---------------------------------------------------------------------------
# Context manager: register / clear RCAM bias around a model forward call
# ---------------------------------------------------------------------------

class RCAMContext:
    """
    Thread-local storage for the active RCAM bias tensor.

    Usage::

        ctx  = RCAMContext()
        bias = build_rcam_bias(...)
        with ctx.scope(bias):
            model(img, txt, ...)   # patched _forward reads from ctx
    """

    def __init__(self) -> None:
        self._bias: Optional[Tensor] = None

    @property
    def bias(self) -> Optional[Tensor]:
        return self._bias

    @contextmanager
    def scope(self, bias: Tensor) -> Iterator[None]:
        self._bias = bias
        try:
            yield
        finally:
            self._bias = None


# ---------------------------------------------------------------------------
# Patched _forward for DoubleStreamBlock
# ---------------------------------------------------------------------------

def _make_double_forward(original_forward, rcam_ctx: RCAMContext):
    """
    Returns a patched _forward that injects RCAM into DoubleStreamBlock.

    The patch reproduces the original logic verbatim, adding only the RCAM
    bias in the scaled_dot_product_attention call.
    """
    def patched(self_blk, img, txt, vec, pe, txt_attention_mask=None):
        from einops import rearrange
        from library.flux_models import apply_rope

        # ── modulation (unchanged) ────────────────────────────────────────
        img_mod1, img_mod2 = self_blk.img_mod(vec)
        txt_mod1, txt_mod2 = self_blk.txt_mod(vec)

        img_mod = self_blk.img_norm1(img)
        img_mod = (1 + img_mod1.scale) * img_mod + img_mod1.shift
        img_qkv = self_blk.img_attn.qkv(img_mod)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self_blk.num_heads)
        img_q, img_k = self_blk.img_attn.norm(img_q, img_k, img_v)

        txt_mod = self_blk.txt_norm1(txt)
        txt_mod = (1 + txt_mod1.scale) * txt_mod + txt_mod1.shift
        txt_qkv = self_blk.txt_attn.qkv(txt_mod)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self_blk.num_heads)
        txt_q, txt_k = self_blk.txt_attn.norm(txt_q, txt_k, txt_v)

        # ── joint Q, K, V ─────────────────────────────────────────────────
        q = torch.cat((txt_q, img_q), dim=2)   # (B, H, S, D)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # ── padding mask (original logic) ─────────────────────────────────
        NEG_INF = -1e9
        combined: Optional[Tensor] = None
        if txt_attention_mask is not None:
            am = txt_attention_mask.to(torch.bool)
            am = torch.cat((
                am,
                torch.ones(am.shape[0], img.shape[1],
                           device=am.device, dtype=torch.bool)
            ), dim=1)
            am = am[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)
            combined = torch.where(
                am,
                torch.zeros(1, device=q.device, dtype=q.dtype),
                torch.full((1,), NEG_INF, device=q.device, dtype=q.dtype),
            )

        # ── RCAM bias injection (Eq. 2 in the paper) ──────────────────────
        rcam = rcam_ctx.bias
        if rcam is not None:
            rb = rcam.to(dtype=q.dtype, device=q.device)
            combined = rb if combined is None else combined + rb

        # ── RoPE + scaled dot-product attention ───────────────────────────
        q_r, k_r = apply_rope(q, k, pe)
        attn_out = F.scaled_dot_product_attention(q_r, k_r, v, attn_mask=combined)
        attn_out = rearrange(attn_out, "B H L D -> B L (H D)")

        txt_attn = attn_out[:, :txt.shape[1]]
        img_attn = attn_out[:, txt.shape[1]:]

        # ── residual + MLP (unchanged) ────────────────────────────────────
        img = img + img_mod1.gate * self_blk.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self_blk.img_mlp(
            (1 + img_mod2.scale) * self_blk.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod1.gate * self_blk.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self_blk.txt_mlp(
            (1 + txt_mod2.scale) * self_blk.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

    return patched


# ---------------------------------------------------------------------------
# Patched _forward for SingleStreamBlock
# ---------------------------------------------------------------------------

def _make_single_forward(original_forward, rcam_ctx: RCAMContext):
    """
    Returns a patched _forward that injects RCAM into SingleStreamBlock.
    In SingleStreamBlock img and txt are merged into a single sequence x.
    """
    def patched(self_blk, x, vec, pe, txt_attention_mask=None):
        from einops import rearrange
        from library.flux_models import apply_rope

        mod, _ = self_blk.modulation(vec)
        x_mod  = (1 + mod.scale) * self_blk.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self_blk.linear1(x_mod),
            [3 * self_blk.hidden_size, self_blk.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D",
                             K=3, H=self_blk.num_heads)
        q, k = self_blk.norm(q, k, v)

        NEG_INF  = -1e9
        combined = None
        if txt_attention_mask is not None:
            am = txt_attention_mask.to(torch.bool)
            am = torch.cat((
                am,
                torch.ones(am.shape[0], x.shape[1] - txt_attention_mask.shape[1],
                           device=am.device, dtype=torch.bool)
            ), dim=1)
            am = am[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)
            combined = torch.where(
                am,
                torch.zeros(1, device=q.device, dtype=q.dtype),
                torch.full((1,), NEG_INF, device=q.device, dtype=q.dtype),
            )

        rcam = rcam_ctx.bias
        if rcam is not None:
            rb = rcam.to(dtype=q.dtype, device=q.device)
            combined = rb if combined is None else combined + rb

        q_r, k_r = apply_rope(q, k, pe)
        attn_out = F.scaled_dot_product_attention(q_r, k_r, v, attn_mask=combined)
        attn_out = rearrange(attn_out, "B H L D -> B L (H D)")

        out = self_blk.linear2(torch.cat((attn_out, self_blk.mlp_act(mlp)), 2))
        return x + mod.gate * out

    return patched


# ---------------------------------------------------------------------------
# RCAMInjector: patches FLUX backbone and manages the bias context
# ---------------------------------------------------------------------------

class RCAMInjector:
    """
    Patches every DoubleStreamBlock and SingleStreamBlock inside a FLUX
    backbone to use RCAM-aware _forward methods.

    Usage
    -----
    >>> injector = RCAMInjector(flux_model)
    >>> bias = build_rcam_bias(masks, attribution, img_seq_len=4096, ...)
    >>> with injector.rcam_ctx.scope(bias):
    ...     output = flux_model(img, txt, ...)
    >>> injector.restore()    # undo all patches
    """

    def __init__(self, unet) -> None:
        self.unet                            = unet
        self.rcam_ctx                        = RCAMContext()
        self._saved: Dict[str, object]       = {}
        self._patch(unet)

    def _patch(self, unet) -> None:
        try:
            from library.flux_models import DoubleStreamBlock, SingleStreamBlock
        except ImportError as e:
            raise ImportError(
                "flux_models.py not found. Make sure sd-scripts is on sys.path."
            ) from e

        n_d = n_s = 0
        for name, mod in unet.named_modules():
            if isinstance(mod, DoubleStreamBlock):
                key = f"{name}._forward"
                self._saved[key] = mod._forward
                mod._forward = _make_double_forward(
                    mod._forward, self.rcam_ctx).__get__(mod, type(mod))
                n_d += 1
            elif isinstance(mod, SingleStreamBlock):
                key = f"{name}._forward"
                self._saved[key] = mod._forward
                mod._forward = _make_single_forward(
                    mod._forward, self.rcam_ctx).__get__(mod, type(mod))
                n_s += 1

        print(f"[RCAMInjector] Patched {n_d} DoubleStreamBlocks "
              f"+ {n_s} SingleStreamBlocks.")

    def restore(self) -> None:
        """Remove RCAM patches and restore original _forward methods."""
        try:
            from library.flux_models import DoubleStreamBlock, SingleStreamBlock
        except ImportError:
            return

        for name, mod in self.unet.named_modules():
            if isinstance(mod, (DoubleStreamBlock, SingleStreamBlock)):
                key = f"{name}._forward"
                if key in self._saved:
                    mod._forward = self._saved[key]
        self._saved.clear()
        print("[RCAMInjector] Patches removed; original _forward restored.")


# ---------------------------------------------------------------------------
# Convenience: build TokenAttribution + bias in one call
# ---------------------------------------------------------------------------

def prepare_rcam_bias(
    spatial_masks:        List[Tensor],
    subject_token_ranges: List[Tuple[int, int]],
    txt_seq_len:          int,
    img_seq_len:          int,
    num_heads:            int,
    spatial_hw:           Optional[Tuple[int, int]] = None,
    device:               torch.device = torch.device("cpu"),
    dtype:                torch.dtype  = torch.float32,
) -> Tensor:
    """
    One-call helper: build the RCAM bias tensor ready for injection.

    Parameters
    ----------
    spatial_masks         : N masks (B, H_lat, W_lat) or (B, L_lat)
    subject_token_ranges  : [(start, end), ...] per subject
    txt_seq_len           : number of text tokens S_txt
    img_seq_len           : number of image tokens S_img
    num_heads             : number of attention heads H
    spatial_hw            : (H_feat, W_feat); inferred from img_seq_len if None
    device, dtype         : output tensor placement

    Returns
    -------
    bias : (B, H, S_txt + S_img, S_txt + S_img)
    """
    attr = TokenAttribution(txt_seq_len, subject_token_ranges)
    return build_rcam_bias(
        spatial_masks  = spatial_masks,
        attribution    = attr,
        img_seq_len    = img_seq_len,
        txt_seq_len    = txt_seq_len,
        num_heads      = num_heads,
        spatial_hw     = spatial_hw,
        device         = device,
        dtype          = dtype,
    )
