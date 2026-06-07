"""
compute_metrics.py
==================
Compute the three evaluation metrics used in the paper (Section 3.1 / 3.3):

  DINO Score  – cosine similarity of DINO ViT-S/16 [CARON et al. 2021]
                features between generated and reference images.
                Measures subject identity fidelity.

  CLIP-I      – cosine similarity of CLIP ViT-L/14 image embeddings
                between generated and reference images.
                Measures visual feature distribution alignment.

  CLIP-T      – cosine similarity between CLIP text embedding (prompt)
                and CLIP image embedding (generated image).
                Measures text-image semantic alignment.

All similarities are computed in the respective model's embedding space
and averaged over all evaluated image pairs.

Usage (batch evaluation)
------------------------
    python eval/compute_metrics.py \\
        --generated_dir  outputs/ours \\
        --reference_dir  datasets/dreambooth \\
        --prompts_file   configs/test_prompts.txt \\
        --output_csv     results/metrics.csv

Usage (single pair, for debugging)
-----------------------------------
    python eval/compute_metrics.py \\
        --generated  outputs/result.png \\
        --reference  datasets/dog/01.png \\
        --prompt     "a dog sitting on the beach"

References
----------
[CARON 2021]  Caron et al., "Emerging Properties in Self-Supervised Vision
              Transformers", ICCV 2021. (DINOv1, ViT-S/16)
[RADFORD 2021] Radford et al., "Learning Transferable Visual Models from
              Natural Language Supervision", ICML 2021. (CLIP ViT-L/14)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


# ---------------------------------------------------------------------------
# DINO feature extractor  (ViT-S/16, DINOv1)
# ---------------------------------------------------------------------------

class DINOExtractor:
    """
    Extracts [CLS] features from DINOv1 ViT-S/16.

    The model is loaded from torch.hub (facebookresearch/dino) and
    cached locally.  Input images are resized to 224×224 and normalised
    with ImageNet statistics.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        print("[Metrics] Loading DINO ViT-S/16 …")
        self.model = torch.hub.load(
            "facebookresearch/dino:main", "dino_vits16",
            pretrained=True, verbose=False
        ).to(device).eval()
        self._preprocess = _build_preprocess(img_size=224)

    @torch.no_grad()
    def encode(self, imgs: List[Image.Image]) -> torch.Tensor:
        """
        Return L2-normalised [CLS] features, shape (N, 384).
        """
        tensors = torch.stack(
            [self._preprocess(img) for img in imgs]
        ).to(self.device)
        feats = self.model(tensors)          # (N, 384)
        return F.normalize(feats, dim=-1)


# ---------------------------------------------------------------------------
# CLIP extractor (ViT-L/14)
# ---------------------------------------------------------------------------

class CLIPExtractor:
    """
    Extracts L2-normalised image and text embeddings from CLIP ViT-L/14.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        print("[Metrics] Loading CLIP ViT-L/14 …")
        try:
            import clip
        except ImportError:
            raise ImportError(
                "openai-clip is required: pip install git+https://github.com/openai/CLIP.git"
            )
        self.model, self.preprocess_fn = clip.load("ViT-L/14", device=device)
        self.model.eval()
        self._clip = clip

    @torch.no_grad()
    def encode_images(self, imgs: List[Image.Image]) -> torch.Tensor:
        """Return L2-normalised CLIP image embeddings, shape (N, 768)."""
        tensors = torch.stack(
            [self.preprocess_fn(img) for img in imgs]
        ).to(self.device)
        feats = self.model.encode_image(tensors)
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_texts(self, prompts: List[str]) -> torch.Tensor:
        """Return L2-normalised CLIP text embeddings, shape (N, 768)."""
        tokens = self._clip.tokenize(prompts, truncate=True).to(self.device)
        feats  = self.model.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)


# ---------------------------------------------------------------------------
# Pre-processing transform
# ---------------------------------------------------------------------------

def _build_preprocess(img_size: int = 224):
    """
    Build a pre-processing function for DINO-style input.
    Resizes to `img_size` × `img_size`, normalises with ImageNet stats.
    Returns a function  PIL.Image → torch.Tensor (3, H, W).
    """
    import torchvision.transforms as T
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return lambda img: transform(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Per-metric functions
# ---------------------------------------------------------------------------

def dino_score(generated_imgs:  List[Image.Image],
               reference_imgs:  List[Image.Image],
               extractor:       DINOExtractor) -> float:
    """
    DINO Score: mean cosine similarity between generated and reference
    DINO [CLS] features.

    If multiple reference images are provided per subject, the final
    DINO score is the maximum similarity over references (best-match
    semantics, following DreamBooth evaluation convention).

    Parameters
    ----------
    generated_imgs : N generated images
    reference_imgs : M reference images (M can differ from N)
    extractor      : DINOExtractor instance

    Returns
    -------
    float in [-1, 1], higher = more identity-faithful
    """
    gen_feats = extractor.encode(generated_imgs)   # (N, D)
    ref_feats = extractor.encode(reference_imgs)   # (M, D)

    # (N, M) pairwise cosine similarities
    sim_matrix = gen_feats @ ref_feats.T

    # For each generated image, take the maximum similarity over references
    scores = sim_matrix.max(dim=1).values   # (N,)
    return float(scores.mean().item())


def clip_i_score(generated_imgs: List[Image.Image],
                 reference_imgs: List[Image.Image],
                 extractor:      CLIPExtractor) -> float:
    """
    CLIP-I Score: mean cosine similarity between generated and reference
    CLIP image embeddings.

    Parameters
    ----------
    generated_imgs : N generated images
    reference_imgs : M reference images

    Returns
    -------
    float in [-1, 1], higher = more visually similar to reference
    """
    gen_feats = extractor.encode_images(generated_imgs)   # (N, D)
    ref_feats = extractor.encode_images(reference_imgs)   # (M, D)

    sim_matrix = gen_feats @ ref_feats.T     # (N, M)
    scores     = sim_matrix.max(dim=1).values
    return float(scores.mean().item())


def clip_t_score(generated_imgs: List[Image.Image],
                 prompts:        List[str],
                 extractor:      CLIPExtractor) -> float:
    """
    CLIP-T Score: mean cosine similarity between prompt text embeddings
    and generated image embeddings.

    Parameters
    ----------
    generated_imgs : N generated images
    prompts        : N text prompts (one per generated image)

    Returns
    -------
    float in [-1, 1], higher = better text-image alignment
    """
    assert len(generated_imgs) == len(prompts), \
        "Number of generated images must match number of prompts."

    img_feats  = extractor.encode_images(generated_imgs)   # (N, D)
    txt_feats  = extractor.encode_texts(prompts)           # (N, D)

    # Diagonal: each image paired with its own prompt
    scores = (img_feats * txt_feats).sum(dim=-1)           # (N,)
    return float(scores.mean().item())


# ---------------------------------------------------------------------------
# Batch evaluation over a directory pair
# ---------------------------------------------------------------------------

def evaluate_directory(
    generated_dir:  str,
    reference_dir:  str,
    prompts:        List[str],
    device:         torch.device,
    batch_size:     int = 32,
) -> Dict[str, float]:
    """
    Compute DINO, CLIP-I, and CLIP-T for all generated images in
    `generated_dir` against reference images in `reference_dir`.

    Directory conventions
    ---------------------
    generated_dir/
        subject_A_prompt_0.png
        subject_A_prompt_1.png
        ...
    reference_dir/
        subject_A/
            01.jpg
            02.jpg
            ...

    If `generated_dir` is flat (no sub-directories), all images are
    compared against all images in `reference_dir`.

    Parameters
    ----------
    generated_dir : directory of generated images
    reference_dir : directory of reference images (or sub-directories)
    prompts       : list of prompts, one per generated image (in sort order)
    device        : torch device
    batch_size    : images per batch for encoding

    Returns
    -------
    dict with keys "dino", "clip_i", "clip_t"
    """
    dino_ext  = DINOExtractor(device)
    clip_ext  = CLIPExtractor(device)

    gen_paths = sorted(
        p for p in Path(generated_dir).iterdir() if _is_image(p))
    ref_paths = sorted(
        p for p in Path(reference_dir).rglob("*") if _is_image(p))

    if len(gen_paths) == 0:
        raise FileNotFoundError(f"No images found in {generated_dir}")
    if len(ref_paths) == 0:
        raise FileNotFoundError(f"No images found in {reference_dir}")

    gen_imgs = [Image.open(p).convert("RGB") for p in gen_paths]
    ref_imgs = [Image.open(p).convert("RGB") for p in ref_paths]

    if len(prompts) != len(gen_imgs):
        raise ValueError(
            f"Number of prompts ({len(prompts)}) must match "
            f"number of generated images ({len(gen_imgs)})."
        )

    # ── compute metrics ──────────────────────────────────────────────────────
    d_score  = _batched_dino(gen_imgs,  ref_imgs, dino_ext, batch_size)
    ci_score = _batched_clip_i(gen_imgs, ref_imgs, clip_ext, batch_size)
    ct_score = _batched_clip_t(gen_imgs, prompts, clip_ext, batch_size)

    return {"dino": d_score, "clip_i": ci_score, "clip_t": ct_score}


def _batched_dino(gen_imgs:  List[Image.Image],
                  ref_imgs:  List[Image.Image],
                  extractor: DINOExtractor,
                  batch_size: int) -> float:
    gen_feats = _encode_in_batches(gen_imgs, extractor.encode, batch_size)
    ref_feats = _encode_in_batches(ref_imgs, extractor.encode, batch_size)
    sim = gen_feats @ ref_feats.T
    return float(sim.max(dim=1).values.mean().item())


def _batched_clip_i(gen_imgs:  List[Image.Image],
                    ref_imgs:  List[Image.Image],
                    extractor: CLIPExtractor,
                    batch_size: int) -> float:
    gen_feats = _encode_in_batches(gen_imgs, extractor.encode_images, batch_size)
    ref_feats = _encode_in_batches(ref_imgs, extractor.encode_images, batch_size)
    sim = gen_feats @ ref_feats.T
    return float(sim.max(dim=1).values.mean().item())


def _batched_clip_t(gen_imgs:  List[Image.Image],
                    prompts:   List[str],
                    extractor: CLIPExtractor,
                    batch_size: int) -> float:
    img_feats = _encode_in_batches(gen_imgs, extractor.encode_images, batch_size)
    txt_feats = _encode_texts_in_batches(prompts, extractor.encode_texts, batch_size)
    scores = (img_feats * txt_feats).sum(dim=-1)
    return float(scores.mean().item())


def _encode_in_batches(imgs:    List[Image.Image],
                       fn,
                       bs:      int) -> torch.Tensor:
    """Run `fn` on `imgs` in batches of size `bs`, return concatenated feats."""
    parts = []
    for i in range(0, len(imgs), bs):
        parts.append(fn(imgs[i: i + bs]))
    return torch.cat(parts, dim=0)


def _encode_texts_in_batches(prompts: List[str],
                              fn,
                              bs:     int) -> torch.Tensor:
    parts = []
    for i in range(0, len(prompts), bs):
        parts.append(fn(prompts[i: i + bs]))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Pretty print + CSV export
# ---------------------------------------------------------------------------

def print_results(results: Dict[str, float], method_name: str = "Ours") -> None:
    print("\n" + "=" * 45)
    print(f"  Evaluation Results  —  {method_name}")
    print("=" * 45)
    print(f"  DINO  ↑  : {results['dino']:.4f}")
    print(f"  CLIP-I ↑ : {results['clip_i']:.4f}")
    print(f"  CLIP-T ↑ : {results['clip_t']:.4f}")
    print("=" * 45 + "\n")


def save_csv(results:     Dict[str, float],
             output_csv:  str,
             method_name: str = "Ours") -> None:
    """Append (or create) a CSV with one row per evaluated method."""
    path    = Path(output_csv)
    is_new  = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "dino", "clip_i", "clip_t"])
        if is_new:
            writer.writeheader()
        writer.writerow({
            "method": method_name,
            "dino":   f"{results['dino']:.4f}",
            "clip_i": f"{results['clip_i']:.4f}",
            "clip_t": f"{results['clip_t']:.4f}",
        })
    print(f"[Metrics] Results appended → {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute DINO / CLIP-I / CLIP-T metrics (Section 3.1)")

    # batch mode
    p.add_argument("--generated_dir",  default=None,
                   help="Directory of generated images (batch mode)")
    p.add_argument("--reference_dir",  default=None,
                   help="Directory of reference images (batch mode)")
    p.add_argument("--prompts_file",   default=None,
                   help="Text file with one prompt per line (batch mode)")

    # single-pair mode
    p.add_argument("--generated",  default=None, help="Single generated image")
    p.add_argument("--reference",  default=None, help="Single reference image")
    p.add_argument("--prompt",     default=None, help="Prompt for single image")

    # common
    p.add_argument("--output_csv",   default=None,
                   help="CSV file to append results to")
    p.add_argument("--method_name",  default="Ours",
                   help="Label for this method in the CSV")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--batch_size",   type=int, default=32)

    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── single-pair mode ─────────────────────────────────────────────────────
    if args.generated and args.reference and args.prompt:
        dino_ext = DINOExtractor(device)
        clip_ext = CLIPExtractor(device)

        gen_img  = [Image.open(args.generated).convert("RGB")]
        ref_img  = [Image.open(args.reference).convert("RGB")]
        prompts  = [args.prompt]

        results = {
            "dino":   dino_score(gen_img, ref_img, dino_ext),
            "clip_i": clip_i_score(gen_img, ref_img, clip_ext),
            "clip_t": clip_t_score(gen_img, prompts, clip_ext),
        }
        print_results(results, args.method_name)
        if args.output_csv:
            save_csv(results, args.output_csv, args.method_name)
        return

    # ── batch mode ───────────────────────────────────────────────────────────
    if not (args.generated_dir and args.reference_dir and args.prompts_file):
        p.error(
            "Provide either (--generated, --reference, --prompt) for single "
            "mode or (--generated_dir, --reference_dir, --prompts_file) for "
            "batch mode."
        )

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    results = evaluate_directory(
        generated_dir = args.generated_dir,
        reference_dir = args.reference_dir,
        prompts       = prompts,
        device        = device,
        batch_size    = args.batch_size,
    )
    print_results(results, args.method_name)
    if args.output_csv:
        save_csv(results, args.output_csv, args.method_name)


if __name__ == "__main__":
    main()
