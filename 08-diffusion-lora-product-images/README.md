# 08: Diffusion LoRA Fine-Tuning on Product Images

**Author:** Adebanji Oluwatimileyin Adelowo

## Overview

LoRA fine-tuning of a Stable Diffusion model on a curated product image dataset (eyewear / e-commerce photography). Demonstrates domain adaptation of a generative diffusion model with minimal compute, training only low-rank adapter weights on top of a frozen base model.

The pipeline is designed to produce a LoRA checkpoint that steers SDXL or SD 1.5 toward consistent, on-brand product imagery when prompted with a trigger token.

**Status:** the curation, training, inference, and evaluation code (scripts and notebooks below) is implemented and configured, but the training run itself has not been executed against a real image dataset in this repository. `data/` currently holds only a metadata schema and sample CSV, and `results/results_template.md` is an unfilled template, not a results writeup.

---

## Project Scope

| Dimension | Decision |
|---|---|
| Task | Text-to-image domain adaptation via LoRA |
| Base model | SDXL 1.0 (primary) · SD 1.5 (low-VRAM fallback) |
| Domain | Eyewear product photography |
| Dataset size | 80–150 high-quality images |
| Captioning | BLIP-2 auto-caption + manual trigger word prefix |
| Training | diffusers + PEFT · 1000–3000 steps · ~2–4h on T4 |
| Deliverables | LoRA weights · before/after grids · inference notebook |

**Why this scope:** 100 images × 1000 steps runs in under 4 hours on a free T4 (Kaggle/Colab). SDXL produces commercially presentable quality; SD 1.5 is the fallback if VRAM is constrained.

---

## Key Skills Demonstrated

- LoRA adapter injection into U-Net cross-attention layers
- Dataset curation pipeline: filtering, aspect-ratio normalization, deduplication
- Automated captioning with BLIP-2 + structured trigger-word schema
- diffusers `DreamBoothTrainingArguments` / custom training loop
- Before/after qualitative comparison and CLIP-score quantitative evaluation
- ComfyUI integration for end-to-end generation workflow documentation

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| Fine-tuning | `diffusers` · `peft` · `accelerate` |
| Captioning | `Salesforce/blip2-opt-2.7b` |
| Image processing | Pillow · OpenCV |
| Evaluation | `open_clip` (CLIP score) · visual inspection |
| Logging | Weights & Biases |
| Workflow UI | ComfyUI |

---

## Repository Structure

```
08-diffusion-lora-product-images/
├── configs/
│   └── lora_config.yaml          # Training hyperparameters
├── data/
│   └── metadata_schema.json      # Caption + metadata format
├── scripts/
│   ├── curate_dataset.py         # Filtering, resizing, dedup
│   ├── generate_captions.py      # BLIP-2 auto-captioning
│   ├── train.py                  # Training launcher
│   ├── dataset_loader.py         # PyTorch dataset class
│   └── infer.py                  # Inference + comparison grid
├── notebooks/
│   ├── 01_dataset_curation.ipynb
│   ├── 02_training_workflow.ipynb
│   ├── 03_inference_comparison.ipynb
│   └── 04_results_analysis.ipynb
├── comfyui/
│   └── comfyui_notes.md          # ComfyUI workflow setup and documentation
├── results/
│   └── results_template.md       # Structured results writeup
├── requirements.txt
└── README.md
```

---

## Notebooks

| Notebook | Covers |
|---|---|
| [01_dataset_curation.ipynb](notebooks/01_dataset_curation.ipynb) | Image pipeline: filter → resize → deduplicate → caption → export |
| [02_training_workflow.ipynb](notebooks/02_training_workflow.ipynb) | End-to-end training from config → checkpoint → validation |
| [03_inference_comparison.ipynb](notebooks/03_inference_comparison.ipynb) | Base vs. fine-tuned side-by-side generation on held-out prompts |
| [04_results_analysis.ipynb](notebooks/04_results_analysis.ipynb) | CLIP score, failure analysis, training curve, honest limitations |

---

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key        # optional but recommended
```

Recommended compute: **NVIDIA T4 16GB** (Kaggle free tier) or **A100 40GB** (Colab Pro). CPU-only is not viable for training.

---

## Training Quick Start

```bash
# 1. Curate and caption dataset
python scripts/curate_dataset.py --input data/raw/ --output data/processed/

python scripts/generate_captions.py \
  --image_dir data/processed/ \
  --output data/metadata.csv \
  --trigger_word "sks eyewear"

# 2. Launch training
accelerate launch scripts/train.py --config configs/lora_config.yaml

# 3. Run inference comparison
python scripts/infer.py \
  --base_model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora_weights outputs/lora_weights/ \
  --prompt "a product photo of sks eyewear on white background, studio lighting" \
  --output results/comparison_grid.png
```

---

## Results

No training run has been executed in this repository yet, so there are no measured CLIP scores or comparison grids to report here. `notebooks/04_results_analysis.ipynb` and [results/results_template.md](results/results_template.md) define the report format (CLIP score, style-consistency rating, before/after grids, training curve, failure analysis) to be filled in once a run completes on real product images.

---

## Limitations

- Dataset size (80–150 images) constrains generalization; model memorizes rather than learns broad distribution.
- Trained on free-tier GPU; no hyperparameter sweep performed.
- No automated perceptual metric (FID) computed: requires a reference dataset of sufficient size.
- ComfyUI workflow tested locally; not deployed as a service.

---

## ComfyUI Integration

A complete ComfyUI workflow is documented in [comfyui/](comfyui/) covering:
- Loading the LoRA checkpoint alongside the base SDXL model
- Prompt conditioning for product photography variations
- Batch generation for style consistency review

See [comfyui/comfyui_notes.md](comfyui/comfyui_notes.md) for setup and workflow explanation.

---

## Key Concepts

### LoRA in Diffusion Models
In diffusion U-Nets, LoRA injects trainable low-rank matrices into cross-attention layers:

```
W' = W + α · (A · B)   where A ∈ R^(d×r), B ∈ R^(r×k), r << d
```

Only `A` and `B` are trained, typically <0.5% of total parameters. The frozen base model's general image knowledge is preserved; the adapters steer it toward the target domain.

### Trigger-Word Captioning
Captions follow the pattern:

```
"a photo of sks eyewear [auto-generated scene description]"
```

`sks` is an arbitrary rare token used as a domain identifier. At inference, including `sks eyewear` in the prompt activates the learned style.
