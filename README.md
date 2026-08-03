# Spatial-Aware Fusion LoRA (SALF)

## Project Title

Spatial-Aware Fusion LoRA: Enhancing Consistency in Multi-Subject Image Generation via Parameter Decoupling and Regional Attention

---

# Description

This repository contains the source code, configuration files, and supplementary materials associated with the manuscript:

**Spatial-Aware Fusion LoRA: Enhancing Consistency in Multi-Subject Image Generation via Parameter Decoupling and Regional Attention**

This work proposes a dual-layer constrained framework for multi-subject image generation based on FLUX.1-dev. The framework consists of two complementary components:

- **Spatial-Aware LoRA Fusion (SALF)**, which dynamically activates subject-specific LoRA adapters according to spatial layouts to reduce identity interference.
- **Regional Cross-Attention Masking (RCAM)**, which introduces region-aware attention constraints to suppress attribute leakage between different subjects.

The proposed framework improves identity preservation and semantic consistency in complex multi-subject image generation tasks.

---

# Dataset Information

The experiments reported in the manuscript are conducted using the public DreamBooth dataset.

**Dataset**

DreamBooth Dataset

**Source**

https://github.com/google/dreambooth

**Dataset Description**

The dataset contains multiple object and animal categories commonly used for subject-driven image generation research.

Reference images are resized, center-cropped, and normalized before training.

No modified version of the dataset is distributed with this repository.

Users should obtain the dataset directly from the official source and comply with its original license and terms of use.

---

# Code Information

The repository contains the following major components.

| File / Folder | Description |
|---------------|-------------|
| app.py | FluxGym training interface |
| networks/ | LoRA network implementation |
| library/ | RCAM implementation |
| inference/ | Multi-subject image generation |
| eval/ | Quantitative evaluation |
| sd-scripts/ | Training backend |
| models.yaml | Model configuration |
| requirements.txt | Python dependencies |

All source code is written in Python 3.10.

---

# Repository Structure

```
Spatial-Aware-Fusion-LoRA/
│
├── app.py
├── models.yaml
├── requirements.txt
│
├── networks/
├── library/
├── inference/
├── eval/
├── sd-scripts/
│
└── README.md
```

---

# Requirements

Operating System

- Ubuntu 22.04 (recommended)

Programming Language

- Python 3.10

GPU

- NVIDIA GPU with CUDA support

Major Python Packages

```
torch
torchvision
diffusers
transformers
accelerate
peft
numpy
Pillow
opencv-python
scipy
scikit-image
matplotlib
safetensors
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/your_repository.git

cd Spatial-Aware-Fusion-LoRA
```

Install dependencies

```bash
pip install -r requirements.txt
```

Initialize submodules if necessary

```bash
git submodule update --init --recursive
```

---

# Training

Each subject is first fine-tuned independently using LoRA.

Example

```bash
python app.py
```

or

```bash
accelerate launch sd-scripts/flux_train_network.py
```

Training parameters used in the manuscript

| Parameter | Value |
|------------|--------|
| Learning Rate | 2×10⁻⁴ |
| LoRA Rank | 16 |
| Epochs | 15 |
| Batch Size | 1 |
| Random Seed | 42 |

---

# Inference

Generate images using multiple LoRA adapters

```bash
python inference/multi_subject_infer.py
```

Users should specify

- pretrained FLUX model
- LoRA checkpoints
- spatial masks (or bounding boxes)
- prompts

The generated images will be saved to the specified output directory.

---

# Evaluation

Quantitative evaluation includes

- DINO Score
- CLIP-I Score
- CLIP-T Score

Example

```bash
python eval/compute_metrics.py
```

The evaluation reproduces the results reported in the manuscript.

---

# Methodology

Overall workflow

```
Reference Images
        │
        ▼
LoRA Fine-tuning
        │
        ▼
Spatial-Aware LoRA Fusion (SALF)
        │
        ▼
Regional Cross-Attention Masking (RCAM)
        │
        ▼
FLUX.1-dev Image Generation
        │
        ▼
Evaluation
        │
        ▼
DINO / CLIP-I / CLIP-T
```

---

# Reproducibility

Random Seed

42

Training Environment

- Python 3.10
- PyTorch
- CUDA
- NVIDIA RTX 5090 GPU

All experiments were conducted using the hyperparameters described in the manuscript.

Following the procedures described in this README should reproduce the reported experimental results.

---

# Citation

If you use this repository, please cite

```bibtex
@article{Mei2026SALF,
  title={Spatial-Aware Fusion LoRA: Enhancing Consistency in Multi-Subject Image Generation via Parameter Decoupling and Regional Attention},
  author={Mei, Shuyan and Jiang, Dan},
  journal={PeerJ Computer Science},
  year={2026}
}
```

(The citation can be updated after publication.)

---

# License

This repository is provided for academic peer review and research purposes.

The authors retain all rights during the peer-review process.

Users should comply with the original licenses of all third-party datasets and pretrained models used in this work.

---

# Contribution Guidelines

Contributions are welcome after publication.

Please submit issues or pull requests through the project repository.

All code, documentation, and comments should be written in English.

---

# Acknowledgements

This work is built upon several outstanding open-source projects, including

- FLUX.1-dev
- DreamBooth
- HuggingFace Diffusers
- PEFT (LoRA)

The authors sincerely thank the developers of these projects for making their work publicly available.

## License & Contribution Guidelines

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

Contributions are welcome via pull request. Please open an issue first to discuss significant changes, keep code comments/documentation in English, and follow the existing code style.
