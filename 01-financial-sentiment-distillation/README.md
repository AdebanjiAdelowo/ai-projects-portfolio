# Financial Sentiment Distillation

**Author:** Adebanji Oluwatimileyin Adelowo  
**Domain:** FinTech / NLP / Model Compression

---

## Overview

This project implements a full model compression pipeline on **FinBERT** (`ProsusAI/finbert`), a BERT model further pre-trained on financial text and fine-tuned for 3-class sentiment classification (positive / negative / neutral) on the Financial PhraseBank (Araci, 2019).

The goal is to produce a significantly smaller and faster model without sacrificing accuracy, using two complementary techniques:

1. **Structural Pruning**, remove unimportant attention heads and entire encoder layers
2. **Knowledge Distillation**, train the pruned model to mimic the original's soft output distributions

### Why this matters
Deploying 110M-parameter transformer models in production is expensive. This pipeline shows how to compress a model by ~19% in parameter count while **recovering almost all of the accuracy lost to pruning** through distillation, a technique used by major AI labs in building smaller model families.

---

## Pipeline

```
FinBERT (110M params, baseline)
        │
        ▼
  [Notebook 02] Taylor-Gradient Head Pruning
        │   Importance score: |W × ∇W L|
        │   Zeroes output projection of least-important heads
        ▼
  [Notebook 02] Activation-Norm Layer Dropping
        │   Importance score: mean ||hidden state|| per layer
        │   Removes 3 least-important encoder layers
        ▼
  Pruned Student (88M params)
        │
        ▼
  [Notebook 03] Knowledge Distillation
        │   Loss: α·T²·KL(student‖teacher) + (1-α)·CE(student, y)
        │   T=3, α=0.7, teacher frozen
        ▼
  Distilled Student (88M params, accuracy recovered)
```

---

## Results

Measured on the `financial_phrasebank` test split (453 sentences), from the saved `baseline_results.json`, `pruning_results.json`, and `kd_results.json` outputs:

| Model | Accuracy | Weighted F1 | Params | Size |
|-------|----------|-------------|--------|------|
| Baseline FinBERT | 97.57% | 0.9761 | 109.48M | 417.67 MB |
| Head Pruned (30% of heads) | 95.36% | 0.9530 | 109.48M | 417.67 MB |
| Layer Dropped (3 layers) | 61.81% | 0.4804 | 88.22M | 336.55 MB |
| **Distilled Student** | **96.91%** | **0.9691** | **88.22M** | **336.55 MB** |

**Test-set overlap.** `ProsusAI/finbert` was fine-tuned on the Financial PhraseBank, and the 453-sentence test split here is a random 20% of that dataset's `sentences_allagree` subset, so most test sentences were probably seen during the baseline's fine-tuning (the model's exact training split is not reproduced here). The pruned and distilled models inherit that exposure through the baseline weights and the distillation targets. The absolute accuracies are therefore optimistic and are not an independent test; the finding is the relative effect of the compression steps (layer dropping 97.57% to 61.81%, distillation back to 96.91%) on the same split. An independent evaluation would need a baseline fine-tuned only on this project's training split, or test data outside the Financial PhraseBank.

The distilled result comes from a single distillation epoch: `03_Knowledge_Distillation.ipynb` uses 1 epoch on CPU and 3 on a GPU, and the saved run used 1.

Dropping 3 encoder layers cuts accuracy sharply (97.57% to 61.81%); distillation recovers nearly all of it (96.91%, 0.66pp below the uncompressed baseline) while keeping the ~19% reduction in parameters and model size. Head pruning alone (zeroing low-importance attention heads without removing them structurally) does not reduce parameter count or size, only the layer-dropping step does.

![Accuracy, weighted F1, model size and per-sample inference time for the four pipeline stages](full_pipeline_comparison.png)

*The four stages side by side, plotted from the saved result JSON files. The inference-time panel
is a single measurement per model: the distilled student and the layer-dropped model have the same
architecture but different recorded times (13.38 and 9.32 ms/sample), so read the latency
differences as indicative only.*

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_Baseline_Evaluation.ipynb` | Load FinBERT, evaluate on `financial_phrasebank` (all_agree split), save baseline metrics and data splits |
| `02_Model_Pruning.ipynb` | Taylor-gradient attention head pruning + activation-norm layer dropping, save pruned student model |
| `03_Knowledge_Distillation.ipynb` | Pre-compute teacher logits, run KL+CE distillation training, benchmark and save distilled model |

**Run in order**, each notebook saves artifacts loaded by the next.

---

## Dataset

**financial_phrasebank** (`sentences_allagree` split)  
- ~2,264 financial sentences labelled by domain experts  
- Labels: the dataset's integer labels (0 negative, 1 neutral, 2 positive) are remapped to FinBERT's label ids (0 positive, 1 negative, 2 neutral)  
- Split: 80% train / 20% test

---

## Key Concepts

**Taylor-Gradient Importance (Head Pruning)**  
Approximates the loss increase caused by removing a parameter:  
`importance(W) = |W × ∇_W L|`  
Heads with the lowest scores have their output projection zeroed.

**Activation-Norm Importance (Layer Dropping)**  
Layers whose hidden states have consistently small norms contribute little to the representation, these are removed entirely.

**Knowledge Distillation Loss**  
`L = α · T² · KL(σ(z_s/T) ‖ σ(z_t/T)) + (1−α) · CE(z_s, y)`  
- T=3 softens the teacher distribution, revealing inter-class relationships  
- α=0.7 weights soft loss higher than hard labels  
- Teacher logits are cached before training to eliminate teacher forward passes from the loop

---

## Requirements

```bash
pip install -r requirements.txt
```

A GPU (e.g. Colab T4) is recommended for Notebook 3. Notebooks 1 and 2 run comfortably on CPU.

---

## Project Structure

```
01-financial-sentiment-distillation/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_Baseline_Evaluation.ipynb
    ├── 02_Model_Pruning.ipynb
    └── 03_Knowledge_Distillation.ipynb
```

Artifacts written by the notebooks:
- committed: `baseline_results.json`, `pruning_results.json`, `kd_results.json` and the PNG charts (confusion matrix, head/layer importance, training curves, pipeline comparison)
- gitignored: `data_splits.pkl`, `student_model/`, `distilled_model/`
