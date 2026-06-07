"""
Multi-Subject Inference with SALF + RCAM
==========================================
End-to-end inference script for the Spatial-Aware Fusion LoRA framework.

This script loads a FLUX.1-dev backbone, injects SALFNetwork and RCAMInjector,
and generates images with N spatially-isolated subjects.

Usage example
-------------
    python inference/multi_subject_infer.py \\
        --base_model  /path/to/flux1-dev.sft \\
        --clip_l      /path/to/clip_l.safetensors \\
        --t5xxl       /path/to/t5xxl_fp16.safetensors \\
        --ae          /path/to/ae.sft \\
        --lora_paths  lora_subject_A.safetensors lora_subject_B.safetensors \\
        --lora_weights 1.0 1.0 \\
        --masks       mask_A.png mask_B.png \\
        --prompt      "A dog and a cat sitting together on a sofa" \\
        --subject_token_ranges "2,5" "6,9" \\
        --output      result.png \\
        --resolution  1024 \\
        --seed        42

Mask format
-----------
Each mask is a grayscale PNG (or any PIL-readable image) where white (255)
marks the subject's spatial region.  The mask is automatically resized to the
latent resolution (resolution // 8).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image

# Allow running from the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sd-scripts"))

from networks.lora_salf import SALFNetwork, SALFModule, apply_dynamic_weight_schedule
from library.rcam_attention import RCAMInjector, prepare_rcam_bias


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def load_mask(path: str, latent_h: int, latent_w: int, device: torch.device) -> torch.Tensor:
    """
    Load a grayscale mask image and resize it to latent resolution.

    Returns
    -------
    mask : (1, latent_h, latent_w)  float32 tensor in [0, 1]
    """
    img = Image.open(path).convert("L")
    img = img.resize((latent_w, latent_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)   # (1, H, W)


def masks_from_bboxes(
    bboxes: List[Tuple[float, float, float, float]],
    latent_h: int,
    latent_w: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Alternative to image-file masks: create binary masks from bounding boxes.

    Parameters
    ----------
    bboxes : list of (x1_norm, y1_norm, x2_norm, y2_norm) in [0, 1]

    Returns
    -------
    masks : list of (1, latent_h, latent_w) float32 tensors
    """
    masks = []
    for x1n, y1n, x2n, y2n in bboxes:
        mask = torch.zeros(1, latent_h, latent_w, device=device)
        x1 = int(x1n * latent_w)
        x2 = int(x2n * latent_w)
        y1 = int(y1n * latent_h)
        y2 = int(y2n * latent_h)
        mask[:, y1:y2, x1:x2] = 1.0
        masks.append(mask)
    return masks


# ---------------------------------------------------------------------------
# LoRA weight loading
# ---------------------------------------------------------------------------

def load_lora_weights_into_salf(
    salf_net: SALFNetwork,
    lora_paths: List[str],
    lora_weights: List[float],
) -> None:
    """
    Load individual per-subject LoRA safetensors files into SALF branches.

    Each safetensors file contains a standard FLUX LoRA trained with FluxGym
    (keys like ``lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight``).
    We map each file to the corresponding subject branch inside every
    SALFModule.

    Parameters
    ----------
    salf_net    : the SALFNetwork whose branches will receive the weights
    lora_paths  : list of N safetensors paths (one per subject)
    lora_weights: list of N scale floats (multiplied into branch.scale)
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors is required: pip install safetensors")

    for subj_idx, (path, weight) in enumerate(zip(lora_paths, lora_weights)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"LoRA file not found: {path}")

        state = load_file(path, device="cpu")
        # Build a mapping:  module_position → (lora_down, lora_up) weights
        # Key pattern: lora_unet_double_blocks_{i}_{layer_name}.lora_{down/up}.weight
        loaded = 0
        for salf_idx, salf_mod in enumerate(salf_net.salf_modules):
            branch = salf_mod.branches[subj_idx]
            # Try to find matching keys heuristically by position
            down_key = _find_key(state, salf_idx, "lora_down")
            up_key   = _find_key(state, salf_idx, "lora_up")
            if down_key and up_key:
                branch.lora_A.weight.data = state[down_key].to(branch.lora_A.weight)
                branch.lora_B.weight.data = state[up_key].to(branch.lora_B.weight)
                loaded += 1

        # Apply user-specified weight as additional multiplier
        for salf_mod in salf_net.salf_modules:
            branch = salf_mod.branches[subj_idx]
            branch.scale *= weight

        print(f"[Inference] Subject {subj_idx}: loaded {loaded} LoRA layers from {path}")


def _find_key(state: dict, position: int, suffix: str) -> Optional[str]:
    """Return the state-dict key at a given module position, or None."""
    candidates = [k for k in state if suffix in k]
    if position < len(candidates):
        return candidates[position]
    return None


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference(
    # Model paths
    base_model: str,
    clip_l: str,
    t5xxl: str,
    ae: str,
    # LoRA
    lora_paths: List[str],
    lora_weights: List[float],
    # Spatial layout
    masks: List[torch.Tensor],               # (1, H_lat, W_lat) per subject
    subject_token_ranges: List[Tuple[int, int]],
    # Generation
    prompt: str,
    resolution: int = 1024,
    num_steps: int = 20,
    guidance_scale: float = 3.5,
    seed: int = 42,
    # Output
    output: str = "output.png",
    device_str: str = "cuda",
    # SALF/RCAM config
    lora_dim: int = 16,
    salf_alpha_early: float = 1.0,
    salf_alpha_late: float = 0.5,
) -> Image.Image:
    """
    Generate a multi-subject image using SALF + RCAM.

    Returns the generated PIL Image and saves it to `output`.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    latent_h = resolution // 8
    latent_w = resolution // 8
    num_subjects = len(lora_paths)

    # ── 1. Load FLUX backbone ────────────────────────────────────────────────
    print("[Inference] Loading FLUX backbone …")
    from library.flux_utils import load_flow_model, load_ae
    from library.flux_utils import load_clip_l, load_t5xxl

    flux_model = load_flow_model("flux-dev", base_model, dtype=dtype, device=device)
    flux_model.eval()
    vae        = load_ae("flux-dev", ae, dtype=torch.float32, device=device)
    clip_model = load_clip_l(clip_l, dtype=dtype, device=device)
    t5_model   = load_t5xxl(t5xxl, dtype=dtype, device=device)

    # ── 2. Inject SALF ──────────────────────────────────────────────────────
    print(f"[Inference] Injecting SALFNetwork for {num_subjects} subjects …")
    salf_net = SALFNetwork(
        flux_model,
        num_subjects=num_subjects,
        lora_dim=lora_dim,
        alpha=float(lora_dim),
        multiplier=1.0,
    )

    # Load per-subject LoRA weights into SALF branches
    load_lora_weights_into_salf(salf_net, lora_paths, lora_weights)
    salf_net.eval()

    # Register spatial masks (broadcast to batch size 1)
    SALFModule.set_spatial_masks(
        [m.to(device) for m in masks],
        spatial_size=None,   # auto-infer per layer
    )

    # ── 3. Inject RCAM ──────────────────────────────────────────────────────
    print("[Inference] Injecting RCAMInjector …")
    rcam_injector = RCAMInjector(flux_model)

    # ── 4. Encode text ──────────────────────────────────────────────────────
    print("[Inference] Encoding text prompt …")
    from library.flux_train_utils import encode_prompts_single

    # Encode once to get token counts (approximate: 256 for T5, 77 for CLIP)
    txt_tokens, clip_tokens = _encode_prompt(
        prompt, clip_model, t5_model, device, dtype
    )
    txt_seq_len = txt_tokens.shape[1]   # T5 sequence length

    # ── 5. Build RCAM bias ───────────────────────────────────────────────────
    img_seq_len = (resolution // 16) ** 2   # FLUX packs 2×2 patches → /16
    num_heads   = flux_model.num_heads

    rcam_bias = prepare_rcam_bias(
        spatial_masks=masks,
        subject_token_ranges=subject_token_ranges,
        txt_seq_len=txt_seq_len,
        img_seq_len=img_seq_len,
        num_heads=num_heads,
        device=device,
        dtype=dtype,
    )

    # ── 6. Denoising loop ────────────────────────────────────────────────────
    print(f"[Inference] Running {num_steps}-step denoising …")
    torch.manual_seed(seed)

    # Initial latent noise (FLUX latent dim = 16 channels, packed 2×2 → /2 each side)
    z = torch.randn(1, (resolution // 16) ** 2, flux_model.in_channels,
                    device=device, dtype=dtype)

    # Timestep schedule (linear from 1 → 0)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for step_idx in range(num_steps):
        t = timesteps[step_idx]
        t_next = timesteps[step_idx + 1]
        dt = t_next - t

        # Dynamic SALF weight schedule (Section 2.4)
        apply_dynamic_weight_schedule(
            salf_net,
            timestep=float(t) * 1000,
            t_max=1000.0,
            alpha_early=salf_alpha_early,
            alpha_late=salf_alpha_late,
        )

        # Build timestep tensor
        t_vec = torch.full((1,), float(t) * 1000, device=device, dtype=dtype)

        # Run model inside RCAM context
        with rcam_injector.rcam_ctx.active(rcam_bias):
            noise_pred = flux_model(
                img=z,
                img_ids=_make_img_ids(resolution, device, dtype),
                txt=txt_tokens,
                txt_ids=_make_txt_ids(txt_seq_len, device, dtype),
                timesteps=t_vec,
                y=clip_tokens,
                guidance=torch.full((1,), guidance_scale, device=device, dtype=dtype),
            )

        # Euler step
        z = z + dt * noise_pred

        if (step_idx + 1) % 5 == 0:
            print(f"  step {step_idx + 1}/{num_steps}")

    # ── 7. Decode latent → pixel ─────────────────────────────────────────────
    print("[Inference] Decoding latent …")
    # Un-pack FLUX latent tokens back to (B, 16, H/8, W/8)
    img_latent = _unpack_flux_latent(z, resolution)

    with torch.no_grad():
        img_tensor = vae.decode(img_latent.to(torch.float32)).sample

    img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)
    img_np = img_tensor[0].permute(1, 2, 0).cpu().float().numpy()
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

    img_pil.save(output)
    print(f"[Inference] Saved → {output}")

    # ── 8. Clean up ──────────────────────────────────────────────────────────
    SALFModule.clear_spatial_masks()
    rcam_injector.restore()

    return img_pil


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_prompt(prompt, clip_model, t5_model, device, dtype):
    """Encode text with T5+CLIP; return (txt_tokens, pooled_clip)."""
    from transformers import CLIPTokenizer, T5TokenizerFast
    import torch

    # T5 encoding
    t5_tok = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl", legacy=False)
    t5_ids = t5_tok(prompt, return_tensors="pt", padding="max_length",
                    max_length=256, truncation=True).input_ids.to(device)
    with torch.no_grad():
        txt_tokens = t5_model(input_ids=t5_ids).last_hidden_state.to(dtype)

    # CLIP encoding (pooled)
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_ids = clip_tok(prompt, return_tensors="pt", padding="max_length",
                        max_length=77, truncation=True).input_ids.to(device)
    with torch.no_grad():
        pooled = clip_model(clip_ids).pooler_output.to(dtype)

    return txt_tokens, pooled


def _make_img_ids(resolution: int, device, dtype) -> torch.Tensor:
    """Build FLUX image position ids for a square image."""
    h = w = resolution // 16   # FLUX token grid
    ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=device, dtype=dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=device, dtype=dtype)[None, :]
    return ids.reshape(1, h * w, 3)


def _make_txt_ids(seq_len: int, device, dtype) -> torch.Tensor:
    """Build FLUX text position ids (all zeros in FLUX convention)."""
    return torch.zeros(1, seq_len, 3, device=device, dtype=dtype)


def _unpack_flux_latent(z: torch.Tensor, resolution: int) -> torch.Tensor:
    """
    Reverse FLUX's patch-packing: (B, L, C) → (B, 16, H/8, W/8).
    FLUX packs 2×2 spatial patches, so L = (H/8 * W/8) / 4 = (H/16 * W/16).
    """
    B = z.shape[0]
    h = w = resolution // 16
    # z : (B, h*w, 16*4=64) → need to unpack
    z = z.reshape(B, h, w, 16, 2, 2)
    z = z.permute(0, 3, 1, 4, 2, 5)       # (B, 16, h, 2, w, 2)
    z = z.reshape(B, 16, h * 2, w * 2)    # (B, 16, H/8, W/8)
    return z


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_token_range(s: str) -> Tuple[int, int]:
    parts = s.split(",")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(description="Multi-subject inference with SALF + RCAM")

    # Model
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--clip_l",     required=True)
    parser.add_argument("--t5xxl",      required=True)
    parser.add_argument("--ae",         required=True)

    # LoRA
    parser.add_argument("--lora_paths",   nargs="+", required=True)
    parser.add_argument("--lora_weights", nargs="+", type=float, default=None)

    # Spatial layout
    parser.add_argument("--masks",                nargs="+", default=None,
                        help="Grayscale PNG files, one per subject")
    parser.add_argument("--bboxes",               nargs="+", default=None,
                        help="Bounding boxes as x1,y1,x2,y2 in [0,1] coords "
                             "(alternative to --masks)")
    parser.add_argument("--subject_token_ranges", nargs="+", required=True,
                        help="Token ranges as start,end (e.g. '2,5')")

    # Generation
    parser.add_argument("--prompt",      required=True)
    parser.add_argument("--resolution",  type=int, default=1024)
    parser.add_argument("--num_steps",   type=int, default=20)
    parser.add_argument("--guidance",    type=float, default=3.5)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      default="output.png")
    parser.add_argument("--device",      default="cuda")

    # SALF/RCAM config
    parser.add_argument("--lora_dim",         type=int,   default=16)
    parser.add_argument("--salf_alpha_early", type=float, default=1.0)
    parser.add_argument("--salf_alpha_late",  type=float, default=0.5)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    latent_hw = args.resolution // 8
    n = len(args.lora_paths)

    # Build masks
    if args.masks:
        mask_list = [load_mask(p, latent_hw, latent_hw, device) for p in args.masks]
    elif args.bboxes:
        parsed_boxes = [tuple(float(v) for v in b.split(",")) for b in args.bboxes]
        mask_list = masks_from_bboxes(parsed_boxes, latent_hw, latent_hw, device)
    else:
        raise ValueError("Provide either --masks or --bboxes")

    if len(mask_list) != n:
        raise ValueError(f"Number of masks ({len(mask_list)}) must match "
                         f"number of LoRAs ({n})")

    lora_weights = args.lora_weights or [1.0] * n
    token_ranges = [parse_token_range(r) for r in args.subject_token_ranges]

    run_inference(
        base_model=args.base_model,
        clip_l=args.clip_l,
        t5xxl=args.t5xxl,
        ae=args.ae,
        lora_paths=args.lora_paths,
        lora_weights=lora_weights,
        masks=mask_list,
        subject_token_ranges=token_ranges,
        prompt=args.prompt,
        resolution=args.resolution,
        num_steps=args.num_steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        output=args.output,
        device_str=args.device,
        lora_dim=args.lora_dim,
        salf_alpha_early=args.salf_alpha_early,
        salf_alpha_late=args.salf_alpha_late,
    )


if __name__ == "__main__":
    main()
