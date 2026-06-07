"""
Spatial-Aware LoRA Fusion (SALF)
=================================
Implementation of the SALF mechanism described in Section 2.2 of:

  "Spatial-Aware Fusion LoRA: Enhancing Consistency in Multi-Subject
   Image Generation via Parameter Decoupling and Regional Attention"

Core equation (Eq. 1):
    h = W0·x + Σᵢ αᵢ · ( R(Mᵢ) ⊙ (Bᵢ Aᵢ x) )

Symbols:
  W0      – frozen pre-trained weight matrix
  Bᵢ, Aᵢ – low-rank decomposition matrices for subject i  (ΔWᵢ = BᵢAᵢ)
  Mᵢ      – binary spatial mask for subject i in latent space
  R(·)    – resampling operator: aligns Mᵢ to the current feature-map size
  ⊙       – element-wise (Hadamard) product, broadcast over the feature dim
  αᵢ      – per-subject scaling coefficient (= lora_alpha / lora_rank)

Key design choices
------------------
* One independent LoRA branch per subject, each with its own Aᵢ and Bᵢ.
* Spatial masks are registered globally before each forward pass via
  SALFModule.set_masks() and cleared afterwards with SALFModule.clear_masks().
* The dynamic mask weight scheduler Φ(M) (Section 2.4) is implemented as
  Gaussian boundary smoothing + strict non-zero enforcement, so that feature
  updates at a given token are driven exclusively by the owning branch.
* SALFNetwork wraps every target Linear in a FLUX DoubleStreamBlock /
  SingleStreamBlock and registers all branches as proper nn.Module children.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Low-rank branch  Bᵢ Aᵢ  for one subject
# ---------------------------------------------------------------------------

class SubjectLoRABranch(nn.Module):
    """
    Trainable low-rank adapter for a single subject.

    ΔWᵢ = Bᵢ Aᵢ   where  Aᵢ ∈ ℝ^{r×in},  Bᵢ ∈ ℝ^{out×r},  r ≪ min(in,out)

    Initialisation: Aᵢ ~ Kaiming-uniform, Bᵢ = 0  →  ΔWᵢ = 0 at step 0.
    """

    def __init__(self, in_features: int, out_features: int,
                 lora_rank: int = 16, lora_alpha: float = 16.0) -> None:
        super().__init__()
        self.lora_rank  = lora_rank
        self.scale      = lora_alpha / lora_rank      # αᵢ in the paper

        self.lora_down  = nn.Linear(in_features,  lora_rank,    bias=False)   # Aᵢ
        self.lora_up    = nn.Linear(lora_rank,     out_features, bias=False)   # Bᵢ

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Return  Bᵢ Aᵢ x  before spatial gating and αᵢ scaling."""
        return self.lora_up(self.lora_down(x))


# ---------------------------------------------------------------------------
# Φ(M):  dynamic mask weight allocation  (Section 2.2 / Section 2.4)
# ---------------------------------------------------------------------------

def _build_gaussian_kernel(sigma: float, radius: int,
                            device: torch.device, dtype: torch.dtype) -> Tensor:
    """1-D normalised Gaussian kernel for separable boundary smoothing."""
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def phi_mask(mask: Tensor,
             target_len: int,
             spatial_hw: Optional[Tuple[int, int]] = None,
             sigma: float = 1.0,
             radius: int = 2) -> Tensor:
    """
    Φ(M): resample mask to current feature resolution and apply Gaussian
    boundary smoothing with strict non-zero enforcement (Section 2.2).

    Parameters
    ----------
    mask       : (B, H_lat, W_lat) or (B, L_lat)  binary float mask
    target_len : L  – token sequence length at the current layer
    spatial_hw : (H_feat, W_feat); inferred from target_len if None
    sigma      : Gaussian σ for boundary smoothing
    radius     : half-width of the Gaussian kernel

    Returns
    -------
    gate : (B, L, 1)  soft gate ∈ [0, 1]
    """
    B        = mask.shape[0]
    device   = mask.device
    dtype    = mask.float().dtype

    # ── 1. Reshape to (B, 1, H, W) ──────────────────────────────────────────
    if mask.dim() == 2:
        L_lat = mask.shape[1]
        s     = int(math.isqrt(L_lat))
        assert s * s == L_lat, \
            f"Cannot infer spatial size from L_lat={L_lat}. Pass spatial_hw."
        m4d = mask.float().reshape(B, 1, s, s)
    elif mask.dim() == 3:
        m4d = mask.float().unsqueeze(1)
    else:
        raise ValueError(f"mask must be 2-D or 3-D, got {mask.dim()}-D")

    # ── 2. Resample to current feature-map resolution: R(Mᵢ) ────────────────
    if spatial_hw is not None:
        H_f, W_f = spatial_hw
    else:
        s_f = int(math.isqrt(target_len))
        assert s_f * s_f == target_len, \
            f"Cannot infer feature spatial size from target_len={target_len}. " \
            f"Pass spatial_hw."
        H_f = W_f = s_f

    if (m4d.shape[2], m4d.shape[3]) != (H_f, W_f):
        m4d = F.interpolate(m4d, size=(H_f, W_f),
                            mode="bilinear", align_corners=False)

    # ── 3. Gaussian boundary smoothing (Φ component) ────────────────────────
    if sigma > 0 and radius > 0:
        k1d = _build_gaussian_kernel(sigma, radius, device, dtype)
        k   = k1d.shape[0]
        pad = k // 2
        k_w = k1d.view(1, 1, 1, k)
        k_h = k1d.view(1, 1, k, 1)
        m4d = F.conv2d(F.pad(m4d, (pad, pad, 0, 0), mode="replicate"), k_w)
        m4d = F.conv2d(F.pad(m4d, (0, 0, pad, pad), mode="replicate"), k_h)

    # ── 4. Flatten → (B, L, 1) ──────────────────────────────────────────────
    gate = m4d.reshape(B, H_f * W_f, 1).clamp(0.0, 1.0)

    # ── 5. Strict non-zero enforcement (Section 2.2, last paragraph) ─────────
    # Values below threshold are zeroed so that each token is driven by
    # exactly its owning branch, preventing cross-contamination.
    gate = torch.where(gate < 1e-6, torch.zeros_like(gate), gate)

    return gate     # (B, L, 1)


# ---------------------------------------------------------------------------
# SALFModule: wraps one nn.Linear with N subject branches
# ---------------------------------------------------------------------------

class SALFModule(nn.Module):
    """
    Wraps a single nn.Linear with N spatially-gated LoRA branches.

    Forward computes Eq. 1:
        h = W0·x + Σᵢ αᵢ · ( R(Mᵢ) ⊙ (Bᵢ Aᵢ x) )

    Class-level mask storage (shared by all SALFModule instances in one model):
        SALFModule.set_masks(masks)    # call before model forward
        SALFModule.clear_masks()       # call after model forward
    """

    # ── class-level mask registry ────────────────────────────────────────────
    _masks:       Optional[List[Tensor]]        = None  # list[N] of (B, H, W)
    _spatial_hw:  Optional[Tuple[int, int]]     = None  # optional explicit size

    @classmethod
    def set_masks(cls, masks: List[Tensor],
                  spatial_hw: Optional[Tuple[int, int]] = None) -> None:
        """
        Register spatial masks before every model forward pass.

        Args
        ----
        masks      : N tensors, each (B, H_lat, W_lat) or (B, L_lat),
                     float or bool in [0, 1] / {0, 1}
        spatial_hw : optional explicit (H_feat, W_feat) for all layers
        """
        cls._masks      = masks
        cls._spatial_hw = spatial_hw

    @classmethod
    def clear_masks(cls) -> None:
        cls._masks      = None
        cls._spatial_hw = None

    # ────────────────────────────────────────────────────────────────────────

    def __init__(self,
                 org_module: nn.Linear,
                 num_subjects: int,
                 lora_rank:    int   = 16,
                 lora_alpha:   float = 16.0,
                 sigma:        float = 1.0,
                 radius:       int   = 2,
                 multiplier:   float = 1.0) -> None:
        super().__init__()

        self.num_subjects = num_subjects
        self.sigma        = sigma
        self.radius       = radius
        self.multiplier   = multiplier

        in_f  = org_module.in_features
        out_f = org_module.out_features

        # N independent LoRA branches, one per subject
        self.branches = nn.ModuleList([
            SubjectLoRABranch(in_f, out_f, lora_rank, lora_alpha)
            for _ in range(num_subjects)
        ])

        # Hook the original Linear; keep its forward callable
        self.org_forward     = org_module.forward
        org_module.forward   = self.forward   # monkey-patch in place

    # ────────────────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, L, C)

        If no masks are registered, falls back to sum of all branches
        (useful during training of individual LoRAs before SALF injection).
        """
        h = self.org_forward(x)   # W0·x

        masks = SALFModule._masks
        B, L, _ = x.shape

        if masks is None:
            # No spatial layout → plain LoRA sum (training mode)
            for branch in self.branches:
                h = h + self.multiplier * branch.scale * branch(x)
            return h

        # ── Spatially-gated accumulation: Σᵢ αᵢ · (R(Mᵢ) ⊙ ΔWᵢ x) ─────────
        for i, branch in enumerate(self.branches):
            delta = branch(x)   # Bᵢ Aᵢ x  →  (B, L, out_f)

            # Pick mask for this branch (fall back to last mask if fewer masks)
            mask_i = masks[i] if i < len(masks) else masks[-1]

            # R(Mᵢ) with Gaussian smoothing: (B, L, 1)
            gate = phi_mask(
                mask_i.to(device=x.device),
                target_len=L,
                spatial_hw=SALFModule._spatial_hw,
                sigma=self.sigma,
                radius=self.radius,
            ).to(dtype=x.dtype)

            h = h + self.multiplier * branch.scale * (gate * delta)

        return h


# ---------------------------------------------------------------------------
# SALFNetwork: injects SALFModules into every target Linear in a FLUX backbone
# ---------------------------------------------------------------------------

class SALFNetwork(nn.Module):
    """
    Walks a FLUX denoising backbone and replaces matching nn.Linear layers
    with SALFModule wrappers.

    Target linear names are identical to those targeted by lora_flux.py so
    that pre-trained LoRA weights can be loaded directly into SALF branches.

    Usage
    -----
    Training (single-subject, per branch):
        net = SALFNetwork(flux, num_subjects=2, lora_rank=16)
        opt = torch.optim.AdamW(net.trainable_params(), lr=1e-4)

    Inference (multi-subject):
        SALFModule.set_masks([mask_A, mask_B])
        out = flux(img, ...)
        SALFModule.clear_masks()
    """

    # Linear sub-modules targeted in DoubleStreamBlock / SingleStreamBlock
    _TARGET_SUFFIXES: Tuple[str, ...] = (
        "img_attn.qkv",
        "img_attn.proj",
        "txt_attn.qkv",
        "txt_attn.proj",
        "img_mlp.0",
        "img_mlp.2",
        "txt_mlp.0",
        "txt_mlp.2",
        "linear1",
        "linear2",
    )

    def __init__(self,
                 unet:         nn.Module,
                 num_subjects: int,
                 lora_rank:    int   = 16,
                 lora_alpha:   float = 16.0,
                 sigma:        float = 1.0,
                 radius:       int   = 2,
                 multiplier:   float = 1.0,
                 target_blocks: str  = "all") -> None:
        """
        Parameters
        ----------
        unet          : FLUX denoising backbone (Flux instance from flux_models.py)
        num_subjects  : N – number of independent subject LoRA branches
        lora_rank     : r – rank of each low-rank decomposition matrix
        lora_alpha    : α – LoRA scaling numerator
        sigma         : Gaussian σ for Φ(M) boundary smoothing
        radius        : half-width of the Gaussian kernel
        multiplier    : global multiplier applied to the whole Σ ΔW sum
        target_blocks : "all" | "double" | "single"
        """
        super().__init__()

        self.num_subjects  = num_subjects
        self.lora_rank     = lora_rank
        self.salf_modules: List[SALFModule] = []

        self._inject(unet, num_subjects, lora_rank, lora_alpha,
                     sigma, radius, multiplier, target_blocks)

        # Register modules so PyTorch can track parameters and state_dict
        for idx, mod in enumerate(self.salf_modules):
            self.add_module(f"salf_{idx}", mod)

        total_params = sum(
            p.numel()
            for mod in self.salf_modules
            for branch in mod.branches
            for p in branch.parameters()
        )
        print(f"[SALFNetwork] Injected {len(self.salf_modules)} SALF modules "
              f"| {num_subjects} subjects | rank={lora_rank} "
              f"| trainable params: {total_params:,}")

    # ── injection ────────────────────────────────────────────────────────────

    def _inject(self, unet, num_subjects, lora_rank, lora_alpha,
                sigma, radius, multiplier, target_blocks):
        for block_name, block in unet.named_modules():
            is_double = "double_blocks" in block_name
            is_single = "single_blocks" in block_name
            if not (is_double or is_single):
                continue
            if target_blocks == "double" and not is_double:
                continue
            if target_blocks == "single" and not is_single:
                continue

            for child_name, child in block.named_modules():
                if not isinstance(child, nn.Linear):
                    continue
                if not any(child_name.endswith(s) for s in self._TARGET_SUFFIXES):
                    continue

                salf_mod = SALFModule(
                    child, num_subjects,
                    lora_rank, lora_alpha,
                    sigma, radius, multiplier,
                )
                self.salf_modules.append(salf_mod)

    # ── training helpers ─────────────────────────────────────────────────────

    def trainable_params(self, lr: float = 1e-4) -> List[Dict]:
        """Return optimizer parameter groups (LoRA branches only)."""
        params = [
            p
            for mod in self.salf_modules
            for branch in mod.branches
            for p in branch.parameters()
        ]
        return [{"params": params, "lr": lr}]

    def set_multiplier(self, value: float) -> None:
        for mod in self.salf_modules:
            mod.multiplier = value

    # ── checkpoint helpers ───────────────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        """Save only the LoRA branch weights (no backbone)."""
        state: Dict[str, Tensor] = {}
        for idx, mod in enumerate(self.salf_modules):
            for s_idx, branch in enumerate(mod.branches):
                pfx = f"salf_{idx}.branch_{s_idx}"
                for k, v in branch.state_dict().items():
                    state[f"{pfx}.{k}"] = v
        torch.save(state, path)
        print(f"[SALFNetwork] Weights saved → {path}")

    def load_weights(self, path: str, strict: bool = True) -> None:
        """Load previously saved LoRA branch weights."""
        state = torch.load(path, map_location="cpu")
        for idx, mod in enumerate(self.salf_modules):
            for s_idx, branch in enumerate(mod.branches):
                pfx = f"salf_{idx}.branch_{s_idx}."
                sub = {k[len(pfx):]: v for k, v in state.items()
                       if k.startswith(pfx)}
                branch.load_state_dict(sub, strict=strict)
        print(f"[SALFNetwork] Weights loaded ← {path}")

    def load_lora_safetensors(self,
                               subject_idx: int,
                               path: str,
                               weight: float = 1.0) -> None:
        """
        Load a standard FLUX LoRA safetensors file into branch `subject_idx`.

        Key mapping (FluxGym / sd-scripts convention):
          lora_unet_double_blocks_{i}_{layer}_lora_down.weight  →  lora_down.weight
          lora_unet_double_blocks_{i}_{layer}_lora_up.weight    →  lora_up.weight
        """
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")

        # Collect ordered (down, up) pairs from safetensors
        down_keys = sorted([k for k in sd if k.endswith("lora_down.weight")])
        up_keys   = sorted([k for k in sd if k.endswith("lora_up.weight")])

        assert len(down_keys) == len(up_keys), \
            "Mismatched lora_down / lora_up counts in safetensors file."

        loaded = 0
        for mod_idx, mod in enumerate(self.salf_modules):
            if mod_idx >= len(down_keys):
                break
            branch = mod.branches[subject_idx]
            branch.lora_down.weight.data = (
                sd[down_keys[mod_idx]].to(branch.lora_down.weight))
            branch.lora_up.weight.data   = (
                sd[up_keys[mod_idx]].to(branch.lora_up.weight))
            # Apply user-defined weight multiplier
            branch.scale *= weight
            loaded += 1

        print(f"[SALFNetwork] Subject {subject_idx}: "
              f"loaded {loaded} layers from '{path}' (weight={weight})")


# ---------------------------------------------------------------------------
# Dynamic weight balancing  (Section 2.4)
# ---------------------------------------------------------------------------

def apply_dynamic_weight_schedule(salf_net:    SALFNetwork,
                                   timestep:    float,
                                   t_max:       float = 1000.0,
                                   alpha_early: float = 1.0,
                                   alpha_late:  float = 0.5) -> None:
    """
    Linearly interpolate the SALF multiplier between early and late denoising.

    In early steps (high t → spatial constraints tightly "lock" identities).
    In later steps (low t → constraints relax so global attention handles
    illumination fusion and cross-subject interaction).

    Parameters
    ----------
    salf_net    : SALFNetwork instance
    timestep    : current diffusion timestep (0 … t_max)
    t_max       : total timestep range (typically 1000)
    alpha_early : multiplier at t = t_max (start of denoising)
    alpha_late  : multiplier at t = 0     (end of denoising)
    """
    ratio = float(timestep) / float(t_max)          # 1.0 → 0.0
    mult  = alpha_late + (alpha_early - alpha_late) * ratio
    salf_net.set_multiplier(mult)
