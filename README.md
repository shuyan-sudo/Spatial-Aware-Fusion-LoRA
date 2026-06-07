# Spatial-Aware Fusion LoRA (SALF)

> **Spatial-Aware Fusion LoRA: Enhancing Consistency in Multi-Subject Image Generation via Parameter Decoupling and Regional Attention**

A dual-layer constrained generation framework for FLUX.1-dev that resolves identity confusion and attribute leakage in multi-subject image synthesis.

---

## Overview

Large-scale diffusion models (FLUX.1) achieve impressive image quality, but their global cross-attention mechanism causes **attribute leakage** when generating multiple subjects simultaneously — the colour of Subject A bleeds onto Subject B, facial features merge, and textures cross-contaminate.

This work proposes two tightly coupled modules:

| Module | Layer | What it does |
|--------|-------|-------------|
| **SALF** – Spatial-Aware LoRA Fusion | Parameter space | Assigns a dedicated low-rank adapter branch to each subject; spatial masks gate which branch is active at each image token |
| **RCAM** – Regional Cross-Attention Masking | Feature space | Injects a layout-driven bias matrix *M* into every attention score map, forcing text tokens to interact only with their designated image regions |

### Key results (DreamBooth dataset, FLUX.1-dev)

| Method | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ |
|--------|--------|---------|---------|
| MIP-Adapter | 0.482 | 0.726 | 0.311 |
| MS-Diffusion | 0.525 | 0.726 | 0.319 |
| OmniGen | 0.511 | 0.722 | 0.331 |
| SSR-Encoder | 0.502 | 0.718 | 0.323 |
| **Ours (SALF + RCAM)** | **0.546** | **0.731** | 0.325 |

---

## Repository Structure

```
Spatial-Aware-Fusion-LoRA/
├── networks/
│   └── lora_salf.py          # SALF: spatial-gated LoRA fusion (Section 2.2)
├── library/
│   └── rcam_attention.py     # RCAM: regional cross-attention masking (Section 2.3)
├── inference/
│   └── multi_subject_infer.py  # End-to-end inference script
├── eval/
│   └── compute_metrics.py    # DINO / CLIP-I / CLIP-T evaluation
├── sd-scripts/               # kohya-ss training backend (git submodule)
├── app.py                    # FluxGym-based training WebUI
├── requirements.txt
├── models.yaml               # Model path configuration
└── README.md
```

---

## Method

### SALF – Spatial-Aware LoRA Fusion (Section 2.2)

Standard LoRA applies a single global ΔW to all image tokens simultaneously, causing identity features to interfere across subject boundaries.

SALF reformulates the linear projection as:

$$h = W_0 x + \sum_{i=1}^{N} \alpha_i \cdot \bigl( R(M_i) \odot (B_i A_i x) \bigr)$$

where:
- $W_0$ — frozen pre-trained weight
- $B_i, A_i$ — subject-specific low-rank matrices ($\Delta W_i = B_i A_i$)
- $M_i$ — binary spatial mask for subject $i$ in latent space
- $R(\cdot)$ — bilinear resampling to the current feature-map resolution
- $\odot$ — element-wise multiplication (broadcast over feature dimension)
- $\alpha_i = \text{lora\_alpha} / r$ — scaling coefficient

The dynamic mask weight function $\Phi(M)$ applies Gaussian boundary smoothing at mask edges to ensure natural transitions, and enforces strict non-zero constraints so each token is governed exclusively by its owning branch.

### RCAM – Regional Cross-Attention Masking (Section 2.3)

RCAM injects a spatial bias matrix $M \in \mathbb{R}^{L \times S}$ into the attention score computation:

$$\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

The bias elements are defined as:

$$M_{i,j} = \begin{cases} 0 & \text{if pixel } i \text{ is in subject-}n\text{'s region and token } j \text{ describes subject }n \\ 0 & \text{if token } j \text{ is a global/background token} \\ -\infty & \text{otherwise} \end{cases}$$

This mathematically blocks cross-regional text–image attention, eliminating semantic attribute leakage at the feature level.

### Synergistic inference (Section 2.4)

Both modules operate simultaneously:
- **SALF** ensures each image region uses the correct identity parameters
- **RCAM** ensures each image region attends only to its subject's text tokens
- **Dynamic weight scheduling** linearly relaxes SALF's spatial constraint from $\alpha_\text{early}=1.0$ to $\alpha_\text{late}=0.5$, letting the global self-attention handle illumination fusion in the final denoising steps

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Spatial-Aware-Fusion-LoRA.git
cd Spatial-Aware-Fusion-LoRA
git submodule update --init --recursive   # initialise sd-scripts
pip install -r requirements.txt
```

### Requirements

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.40.0
safetensors>=0.4.0
diffusers>=0.28.0
accelerate>=0.28.0
einops>=0.7.0
Pillow>=10.0.0
numpy>=1.24.0
# for evaluation
git+https://github.com/openai/CLIP.git
```

---

## Training (per-subject LoRA)

Train one standard FLUX LoRA per subject using the FluxGym WebUI or directly:

```bash
# Launch the WebUI
python app.py

# Or train from CLI via sd-scripts
accelerate launch \
  --mixed_precision bf16 \
  sd-scripts/flux_train_network.py \
  --pretrained_model_name_or_path /path/to/flux1-dev.sft \
  --clip_l  /path/to/clip_l.safetensors \
  --t5xxl   /path/to/t5xxl_fp16.safetensors \
  --ae      /path/to/ae.sft \
  --network_module networks.lora_flux \
  --network_dim 16 \
  --learning_rate 2e-4 \
  --max_train_epochs 15 \
  --dataset_config outputs/SUBJECT_NAME/dataset.toml \
  --output_dir outputs/SUBJECT_NAME \
  --output_name SUBJECT_NAME \
  --seed 42 \
  ...
```

Training hyperparameters used in the paper (Section 3.1):

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 2 × 10⁻⁴ |
| LoRA rank *r* | 16 |
| Repeats per epoch | 10 |
| Training epochs | 15 |
| Batch size | 1 |
| Seed | 42 |
| Mixed precision | bf16 |
| Timestep sampling | sigmoid |
| Guidance scale | 1.0 |

---

## Inference

### Two subjects from mask files

```bash
python inference/multi_subject_infer.py \
  --base_model  /path/to/flux1-dev.sft \
  --clip_l      /path/to/clip_l.safetensors \
  --t5xxl       /path/to/t5xxl_fp16.safetensors \
  --ae          /path/to/ae.sft \
  --lora_paths  outputs/dog/dog.safetensors outputs/cat/cat.safetensors \
  --lora_weights 1.0 1.0 \
  --masks       masks/dog_left.png masks/cat_right.png \
  --prompt      "a dog and a cat sitting on a sofa" \
  --subject_token_ranges "2,3" "5,6" \
  --output      results/dog_cat.png \
  --resolution  1024 \
  --seed        42
```

### Two subjects from bounding boxes (no mask files needed)

```bash
python inference/multi_subject_infer.py \
  --base_model  /path/to/flux1-dev.sft \
  --clip_l      /path/to/clip_l.safetensors \
  --t5xxl       /path/to/t5xxl_fp16.safetensors \
  --ae          /path/to/ae.sft \
  --lora_paths  outputs/subjectA.safetensors outputs/subjectB.safetensors \
  --bboxes      "0.05,0.1,0.48,0.9" "0.52,0.1,0.95,0.9" \
  --prompt      "a backpack and a vase on a white rug" \
  --subject_token_ranges "2,3" "5,6" \
  --output      results/out.png
```

### Python API

```python
import torch
from PIL import Image
from networks.lora_salf import SALFNetwork, SALFModule, apply_dynamic_weight_schedule
from library.rcam_attention import RCAMInjector, prepare_rcam_bias

# Assume flux, vae, t5_hidden, clip_pooled are already loaded …

masks = [mask_subjectA, mask_subjectB]  # each (1, H_lat, W_lat)

# Inject SALF
salf_net = SALFNetwork(flux, num_subjects=2, lora_rank=16)
salf_net.load_lora_safetensors(0, "subjectA.safetensors", weight=1.0)
salf_net.load_lora_safetensors(1, "subjectB.safetensors", weight=1.0)

# Inject RCAM
rcam = RCAMInjector(flux)
bias = prepare_rcam_bias(
    spatial_masks        = masks,
    subject_token_ranges = [(2, 3), (5, 6)],
    txt_seq_len          = 256,
    img_seq_len          = 4096,
    num_heads            = flux.num_heads,
)

# Denoising step
SALFModule.set_masks(masks)
with rcam.rcam_ctx.scope(bias):
    noise_pred = flux(img=z, txt=t5_hidden, y=clip_pooled, ...)
SALFModule.clear_masks()

# Cleanup
rcam.restore()
```

---

## Evaluation

```bash
# Batch evaluation (reproduces Table 1)
python eval/compute_metrics.py \
  --generated_dir  outputs/ours \
  --reference_dir  datasets/dreambooth \
  --prompts_file   configs/test_prompts.txt \
  --output_csv     results/metrics.csv \
  --method_name    "SALF+RCAM"

# Single pair (quick debug)
python eval/compute_metrics.py \
  --generated  outputs/result.png \
  --reference  datasets/dog/01.jpg \
  --prompt     "a dog on the beach"
```

Output:

```
=============================================
  Evaluation Results  —  SALF+RCAM
=============================================
  DINO  ↑  : 0.5460
  CLIP-I ↑ : 0.7310
  CLIP-T ↑ : 0.3250
=============================================
```

---

## Ablation Study

Quantitative ablation results (Table 2):

| Configuration | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ |
|---------------|--------|---------|---------|
| w/o SALF (global LoRA) | 0.492 | 0.708 | 0.318 |
| w/o RCAM | 0.528 | 0.715 | 0.306 |
| **Full method** | **0.546** | **0.731** | **0.325** |

- Removing SALF causes identity blending (feature weights of different subjects superpose without physical isolation)
- Removing RCAM causes attribute leakage (text descriptors affect non-target spatial regions)

---

## Model Paths

Edit `models.yaml` to match your local model storage:

```yaml
flux-dev:
  file: flux1-dev.sft
  repo: black-forest-labs/FLUX.1-dev
clip_l: /path/to/clip_l.safetensors
t5xxl:  /path/to/t5xxl_fp16.safetensors
ae:     /path/to/ae.sft
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{salf2025,
  title     = {Spatial-Aware Fusion LoRA: Enhancing Consistency in Multi-Subject
               Image Generation via Parameter Decoupling and Regional Attention},
  author    = {[Author Names]},
  year      = {2025},
}
```

---

## Acknowledgements

This project builds on:
- [FLUX.1](https://github.com/black-forest-labs/flux) by Black Forest Labs
- [sd-scripts / FluxGym](https://github.com/cocktailpeanut/fluxgym) by kohya-ss & cocktailpeanut
- [DINO](https://github.com/facebookresearch/dino) by Facebook Research
- [CLIP](https://github.com/openai/CLIP) by OpenAI

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
