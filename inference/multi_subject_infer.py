"""
multi_subject_infer.py
=======================
End-to-end multi-subject inference combining SALF + RCAM on FLUX.1-dev.

This script:
1. Loads FLUX.1-dev backbone (flux1-dev.sft) + text encoders + VAE.
2. Injects SALFNetwork (N subject LoRA branches with spatial gating).
3. Injects RCAMInjector (attention bias for semantic-spatial alignment).
4. Loads one pre-trained LoRA safetensors file per subject into its
   dedicated SALF branch.
5. Runs the FLUX.1 diffusion denoising loop with:
   - dynamic weight scheduling (Section 2.4)
   - RCAM bias activated inside each denoising step
6. Decodes the final latent and saves the generated image.

Usage
-----
    python inference/multi_subject_infer.py \\
        --base_model  /path/to/flux1-dev.sft \\
        --clip_l      /path/to/clip_l.safetensors \\
        --t5xxl       /path/to/t5xxl_fp16.safetensors \\
        --ae          /path/to/ae.sft \\
        --lora_paths  subjectA.safetensors subjectB.safetensors \\
        --lora_weights 1.0 1.0 \\
        --masks       mask_A.png mask_B.png \\
        --prompt      "a dog and a cat sitting on a sofa" \\
        --subject_token_ranges "2,4" "6,8" \\
        --output      result.png \\
        --resolution  1024 \\
        --seed        42

Mask format
-----------
Grayscale PNG where white (255) = subject region.
Automatically resized to latent resolution (resolution // 8).

Alternatively, use --bboxes instead of --masks:
    --bboxes "0.05,0.1,0.45,0.9" "0.55,0.1,0.95,0.9"
    (normalised x1,y1,x2,y2 in [0,1])

Experiment settings (Section 3.1)
----------------------------------
  learning_rate  : 2e-4  (for training; fixed here as script default)
  lora_rank      : 16
  num_repeats    : 10  per epoch
  max_train_epochs: 15
  batch_size     : 1
  seed           : 42
  base_model     : FLUX.1-dev
  text_encoders  : T5-XXL + CLIP-L
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sd-scripts"))

from networks.lora_salf import (
    SALFNetwork, SALFModule, apply_dynamic_weight_schedule
)
from library.rcam_attention import (
    RCAMInjector, prepare_rcam_bias
)


# ---------------------------------------------------------------------------
# Mask / spatial-layout helpers
# ---------------------------------------------------------------------------

def load_mask_from_file(path: str,
                        latent_h: int,
                        latent_w: int,
                        device: torch.device) -> Tensor:
    """Load a grayscale PNG mask and resize to latent resolution."""
    from torch import Tensor
    img  = Image.open(path).convert("L")
    img  = img.resize((latent_w, latent_h), Image.BILINEAR)
    arr  = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)   # (1, H_lat, W_lat)


def build_mask_from_bbox(bbox: Tuple[float, float, float, float],
                         latent_h: int,
                         latent_w: int,
                         device: torch.device) -> torch.Tensor:
    """
    Create a binary mask from a normalised bounding box (x1, y1, x2, y2).
    All values are in [0, 1] relative to image size.
    """
    x1n, y1n, x2n, y2n = bbox
    mask = torch.zeros(1, latent_h, latent_w, device=device)
    x1 = int(x1n * latent_w)
    x2 = int(x2n * latent_w)
    y1 = int(y1n * latent_h)
    y2 = int(y2n * latent_h)
    mask[:, y1:y2, x1:x2] = 1.0
    return mask     # (1, H_lat, W_lat)


# ---------------------------------------------------------------------------
# FLUX model loading (sd-scripts API)
# ---------------------------------------------------------------------------

def load_flux_components(base_model: str,
                         clip_l:     str,
                         t5xxl:      str,
                         ae:         str,
                         device:     torch.device,
                         dtype:      torch.dtype):
    """
    Load FLUX.1-dev backbone, text encoders, and VAE using sd-scripts helpers.
    All components are moved to `device` and cast to `dtype` (except VAE
    which stays in float32 for numerical stability).
    """
    from library.flux_utils import (
        load_flow_model, load_ae, load_clip_l, load_t5xxl
    )

    print("[Inference] Loading FLUX backbone …")
    flux = load_flow_model("flux-dev", base_model, dtype=dtype, device=device)
    flux.eval()
    for p in flux.parameters():
        p.requires_grad_(False)

    print("[Inference] Loading VAE …")
    vae = load_ae("flux-dev", ae, dtype=torch.float32, device=device)
    vae.eval()

    print("[Inference] Loading CLIP-L …")
    clip = load_clip_l(clip_l, dtype=dtype, device=device)
    clip.eval()

    print("[Inference] Loading T5-XXL …")
    t5 = load_t5xxl(t5xxl, dtype=dtype, device=device)
    t5.eval()

    return flux, vae, clip, t5


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def encode_text(prompt:   str,
                clip_model,
                t5_model,
                device:   torch.device,
                dtype:    torch.dtype,
                t5_max_length:   int = 256,
                clip_max_length: int = 77):
    """
    Encode `prompt` with T5-XXL and CLIP-L.
    Returns (t5_hidden, clip_pooled).
      t5_hidden  : (1, T5_seq_len, 4096)   – sequence used as K/V in cross-attn
      clip_pooled: (1, 768)                – pooled embedding used as vec input
    """
    from transformers import T5TokenizerFast, CLIPTokenizer

    # T5-XXL
    t5_tok  = T5TokenizerFast.from_pretrained(
        "google/t5-v1_1-xxl", legacy=False)
    t5_ids  = t5_tok(prompt,
                     return_tensors="pt",
                     padding="max_length",
                     max_length=t5_max_length,
                     truncation=True).input_ids.to(device)
    with torch.no_grad():
        t5_hidden = t5_model(input_ids=t5_ids).last_hidden_state.to(dtype)

    # CLIP-L
    clip_tok  = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_ids  = clip_tok(prompt,
                         return_tensors="pt",
                         padding="max_length",
                         max_length=clip_max_length,
                         truncation=True).input_ids.to(device)
    with torch.no_grad():
        clip_pooled = clip_model(clip_ids).pooler_output.to(dtype)

    return t5_hidden, clip_pooled   # shapes: (1, 256, 4096),  (1, 768)


# ---------------------------------------------------------------------------
# FLUX positional id helpers
# ---------------------------------------------------------------------------

def make_img_ids(resolution: int,
                 device:     torch.device,
                 dtype:      torch.dtype) -> torch.Tensor:
    """
    Build FLUX image position IDs for a square image.
    FLUX packs 2×2 spatial patches → token grid is (res/16) × (res/16).
    Returns (1, L_img, 3).
    """
    h = w = resolution // 16
    ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
    return ids.reshape(1, h * w, 3)


def make_txt_ids(seq_len: int,
                 device:  torch.device,
                 dtype:   torch.dtype) -> torch.Tensor:
    """Build FLUX text position IDs (all zeros). Returns (1, seq_len, 3)."""
    return torch.zeros(1, seq_len, 3, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# FLUX latent pack / unpack
# ---------------------------------------------------------------------------

def pack_latent(z: torch.Tensor) -> torch.Tensor:
    """
    (B, 16, H, W) → (B, L, 64)
    FLUX packs 2×2 spatial patches into one token.
    H and W must be divisible by 2.
    """
    B, C, H, W = z.shape
    z = z.reshape(B, C, H // 2, 2, W // 2, 2)
    z = z.permute(0, 2, 4, 1, 3, 5)       # (B, H/2, W/2, C, 2, 2)
    z = z.reshape(B, (H // 2) * (W // 2), C * 4)
    return z


def unpack_latent(z: torch.Tensor, resolution: int) -> torch.Tensor:
    """
    (B, L, 64) → (B, 16, H, W)  where H = W = resolution // 8.
    """
    B   = z.shape[0]
    h   = w = resolution // 16
    out = z.reshape(B, h, w, 16, 2, 2)
    out = out.permute(0, 3, 1, 4, 2, 5)   # (B, 16, h, 2, w, 2)
    out = out.reshape(B, 16, h * 2, w * 2)
    return out


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference(
    # Model paths
    base_model:  str,
    clip_l:      str,
    t5xxl:       str,
    ae:          str,
    # Per-subject LoRA
    lora_paths:   List[str],
    lora_weights: List[float],
    # Spatial layout
    masks:                List[torch.Tensor],    # each (1, H_lat, W_lat)
    subject_token_ranges: List[Tuple[int, int]], # [(start,end), ...]
    # Prompt
    prompt:       str,
    # Generation
    resolution:   int   = 1024,
    num_steps:    int   = 20,
    guidance:     float = 3.5,
    seed:         int   = 42,
    # Output
    output:       str   = "output.png",
    device_str:   str   = "cuda",
    # SALF config
    lora_rank:    int   = 16,
    alpha_early:  float = 1.0,
    alpha_late:   float = 0.5,
) -> Image.Image:
    """
    Generate a multi-subject image using SALF + RCAM.

    Returns the PIL Image and saves it to `output`.
    """
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    n_subjects = len(lora_paths)
    latent_hw  = resolution // 8      # spatial size of FLUX latent
    img_tokens = (resolution // 16) ** 2   # number of packed image tokens

    # ── 1. Load model components ─────────────────────────────────────────────
    flux, vae, clip_model, t5_model = load_flux_components(
        base_model, clip_l, t5xxl, ae, device, dtype)
    num_heads = flux.num_heads

    # ── 2. Inject SALF ───────────────────────────────────────────────────────
    print(f"[Inference] Injecting SALFNetwork ({n_subjects} subjects, "
          f"rank={lora_rank}) …")
    salf_net = SALFNetwork(
        flux, num_subjects=n_subjects,
        lora_rank=lora_rank, lora_alpha=float(lora_rank),
        multiplier=1.0,
    )

    for i, (path, w) in enumerate(zip(lora_paths, lora_weights)):
        salf_net.load_lora_safetensors(i, path, weight=w)
    salf_net.eval()

    # ── 3. Inject RCAM ───────────────────────────────────────────────────────
    print("[Inference] Injecting RCAMInjector …")
    rcam_injector = RCAMInjector(flux)

    # ── 4. Encode text ───────────────────────────────────────────────────────
    print(f"[Inference] Encoding prompt: '{prompt}'")
    t5_hidden, clip_pooled = encode_text(
        prompt, clip_model, t5_model, device, dtype)
    txt_seq_len = t5_hidden.shape[1]        # 256 for T5-XXL default

    # ── 5. Build RCAM bias once per generation ───────────────────────────────
    print("[Inference] Building RCAM bias matrix …")
    rcam_bias = prepare_rcam_bias(
        spatial_masks        = [m.to(device) for m in masks],
        subject_token_ranges = subject_token_ranges,
        txt_seq_len          = txt_seq_len,
        img_seq_len          = img_tokens,
        num_heads            = num_heads,
        device               = device,
        dtype                = dtype,
    )

    # ── 6. Initial latent noise ───────────────────────────────────────────────
    # FLUX in-channels = 64 (16 channels × 2×2 packed patches)
    z = torch.randn(
        1, img_tokens, flux.in_channels,
        device=device, dtype=dtype, generator=torch.manual_seed(seed),
    )

    img_ids = make_img_ids(resolution, device, dtype)
    txt_ids = make_txt_ids(txt_seq_len, device, dtype)

    # ── 7. Denoising loop ─────────────────────────────────────────────────────
    # Linear timestep schedule from 1.0 → 0.0 (flow-matching convention)
    timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps,
                               device=device, dtype=dtype)

    print(f"[Inference] Denoising ({num_steps} steps) …")
    for step_idx, t in enumerate(timesteps):
        dt = -1.0 / num_steps   # constant Euler step for linear schedule

        # ── dynamic SALF weight schedule (Section 2.4) ───────────────────
        apply_dynamic_weight_schedule(
            salf_net,
            timestep   = float(t) * 1000.0,
            t_max      = 1000.0,
            alpha_early = alpha_early,
            alpha_late  = alpha_late,
        )

        # Register spatial masks for this step
        SALFModule.set_masks([m.to(device) for m in masks])

        t_vec = torch.full((1,), float(t) * 1000.0, device=device, dtype=dtype)

        # ── model forward inside RCAM context ────────────────────────────
        with rcam_injector.rcam_ctx.scope(rcam_bias):
            noise_pred = flux(
                img        = z,
                img_ids    = img_ids,
                txt        = t5_hidden,
                txt_ids    = txt_ids,
                timesteps  = t_vec,
                y          = clip_pooled,
                guidance   = torch.full((1,), guidance, device=device, dtype=dtype),
            )

        SALFModule.clear_masks()

        # Euler step
        z = z + dt * noise_pred

        if (step_idx + 1) % 5 == 0 or step_idx == 0:
            print(f"  [{step_idx + 1:02d}/{num_steps}]  t={float(t):.3f}")

    # ── 8. Decode ─────────────────────────────────────────────────────────────
    print("[Inference] Decoding …")
    latent = unpack_latent(z, resolution).to(torch.float32)
    with torch.no_grad():
        img_tensor = vae.decode(latent).sample

    img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)
    img_np  = img_tensor[0].permute(1, 2, 0).cpu().float().numpy()
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_pil.save(output)
    print(f"[Inference] Saved → {output}")

    # ── 9. Clean up patches ───────────────────────────────────────────────────
    rcam_injector.restore()
    SALFModule.clear_masks()

    return img_pil


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_bbox(s: str) -> Tuple[float, float, float, float]:
    parts = [float(v) for v in s.split(",")]
    assert len(parts) == 4, "bbox must be x1,y1,x2,y2"
    return tuple(parts)


def _parse_range(s: str) -> Tuple[int, int]:
    a, b = s.split(",")
    return int(a), int(b)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Multi-subject inference with SALF + RCAM on FLUX.1-dev")

    # model paths
    p.add_argument("--base_model",  required=True, help="Path to flux1-dev.sft")
    p.add_argument("--clip_l",      required=True, help="Path to clip_l.safetensors")
    p.add_argument("--t5xxl",       required=True, help="Path to t5xxl_fp16.safetensors")
    p.add_argument("--ae",          required=True, help="Path to ae.sft (VAE)")

    # per-subject LoRA
    p.add_argument("--lora_paths",   nargs="+", required=True,
                   help="One .safetensors per subject (space-separated)")
    p.add_argument("--lora_weights", nargs="+", type=float, default=None,
                   help="Scale per LoRA (default: 1.0 for each)")

    # spatial layout (masks XOR bboxes)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--masks",  nargs="+",
                     help="Grayscale PNG files, one per subject")
    grp.add_argument("--bboxes", nargs="+",
                     help="Normalised bounding boxes as x1,y1,x2,y2")

    # token attribution
    p.add_argument("--subject_token_ranges", nargs="+", required=True,
                   help="Token ranges as start,end — e.g. '2,4' '6,8'")

    # generation
    p.add_argument("--prompt",     required=True)
    p.add_argument("--resolution", type=int,   default=1024)
    p.add_argument("--num_steps",  type=int,   default=20)
    p.add_argument("--guidance",   type=float, default=3.5)
    p.add_argument("--seed",       type=int,   default=42,
                   help="Fixed seed used in all experiments (Section 3.1)")
    p.add_argument("--output",     default="output.png")
    p.add_argument("--device",     default="cuda")

    # SALF config (matches Section 3.1 defaults)
    p.add_argument("--lora_rank",    type=int,   default=16,
                   help="LoRA rank r (default 16 from Section 3.1)")
    p.add_argument("--alpha_early",  type=float, default=1.0,
                   help="SALF multiplier at start of denoising")
    p.add_argument("--alpha_late",   type=float, default=0.5,
                   help="SALF multiplier at end of denoising")

    args = p.parse_args()

    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n         = len(args.lora_paths)
    lat_hw    = args.resolution // 8
    weights   = args.lora_weights or [1.0] * n

    if len(weights) != n:
        p.error(f"--lora_weights must have the same length as --lora_paths ({n})")

    # Build masks
    if args.masks:
        if len(args.masks) != n:
            p.error(f"--masks must have the same length as --lora_paths ({n})")
        mask_list = [
            load_mask_from_file(path, lat_hw, lat_hw, device)
            for path in args.masks
        ]
    else:
        if len(args.bboxes) != n:
            p.error(f"--bboxes must have the same length as --lora_paths ({n})")
        mask_list = [
            build_mask_from_bbox(_parse_bbox(b), lat_hw, lat_hw, device)
            for b in args.bboxes
        ]

    token_ranges = [_parse_range(r) for r in args.subject_token_ranges]
    if len(token_ranges) != n:
        p.error(f"--subject_token_ranges must have the same length "
                f"as --lora_paths ({n})")

    run_inference(
        base_model            = args.base_model,
        clip_l                = args.clip_l,
        t5xxl                 = args.t5xxl,
        ae                    = args.ae,
        lora_paths            = args.lora_paths,
        lora_weights          = weights,
        masks                 = mask_list,
        subject_token_ranges  = token_ranges,
        prompt                = args.prompt,
        resolution            = args.resolution,
        num_steps             = args.num_steps,
        guidance              = args.guidance,
        seed                  = args.seed,
        output                = args.output,
        device_str            = args.device,
        lora_rank             = args.lora_rank,
        alpha_early           = args.alpha_early,
        alpha_late            = args.alpha_late,
    )


if __name__ == "__main__":
    main()
