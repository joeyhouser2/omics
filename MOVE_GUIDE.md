# MOVE Usage Guide

This guide explains how to use MOVE (Multi-Omics Variational autoEncoder) with your multi-omics data and perturbations.

## Overview

**MOVE** is a deep learning framework that:
- Integrates multiple omics modalities (RNA-seq, methylation, CNV)
- Handles perturbations (mutations, treatments, clinical variables)
- Predicts cross-omics effects of perturbations
- Creates shared latent representations for downstream analysis

**Official Resources:**
- GitHub: https://github.com/RasmussenLab/MOVE
- Documentation: https://move-dl.readthedocs.io/
- Colab Tutorial: https://colab.research.google.com/drive/1RFWNsuGymCmppPsElBvDuA9zRbGskKmi
- Paper: [Nature Biotechnology (2023)](https://www.nature.com/articles/s41587-023-01705-7)

## Installation

```bash
pip install move-dl
```

## Your Data Setup

### Perturbations (Binary Variables)
1. **FLT3_Mutated**: FLT3 mutation status (0=wild-type, 1=mutated)
2. **NPM1_Mutated**: NPM1 mutation status (0=wild-type, 1=mutated)
3. **Gender_Male**: Gender (0=female, 1=male)

### Omics Modalities
1. **RNA-seq**: Gene expression (5000 top variable genes)
2. **Methylation**: DNA methylation (5000 top variable CpG sites)
3. **CNV**: Copy number variation (3000 top variable genes)

## Workflow

### Step 1: Preprocess Data

```bash
python src/preprocessing.py
```

Creates aligned, normalized datasets in `processed_data/`.

### Step 2: Train MOVE Model

```bash
python src/train.py
```

This script:
- Loads preprocessed data
- Configures MOVE for your specific setup
- Trains the model
- Saves model weights and embeddings

### Step 3: Perturbation Analysis

After training, you can:

#### A: Extract Latent Embeddings

```python
# Get shared latent representation for all samples
embeddings = model.get_embeddings(data)
# Shape: [n_samples, latent_dim]

# Use for clustering, visualization, etc.
```

#### B: Predict Perturbation Effects

**Example: "What if FLT3 mutation was absent?"**

```python
# Original data with FLT3=1
original_perturbations = [1, 0, 1]  # [FLT3=1, NPM1=0, Gender=1]

# Perturbed: flip FLT3 to 0
perturbed_perturbations = [0, 0, 1]  # [FLT3=0, NPM1=0, Gender=1]

# Predict how omics would change
predicted_omics = model.predict_perturbation_effect(
    original_data=omics_data,
    original_perturbations=original_perturbations,
    perturbed_perturbations=perturbed_perturbations
)

# predicted_omics contains reconstructed RNA, methylation, CNV
# under the counterfactual scenario
```

#### C: Identify Perturbation-Responsive Genes

```python
# Compare original vs perturbed predictions
delta_rna = predicted_omics['rna'] - original_data['rna']
delta_meth = predicted_omics['methylation'] - original_data['methylation']

# Find genes most affected by FLT3 mutation
top_genes = delta_rna.abs().topk(100)
```

## Key Use Cases

### 1. Cross-Omics Integration
Combine RNA, methylation, and CNV into a unified representation.

### 2. Biomarker Discovery
Identify genes/pathways associated with mutations (FLT3, NPM1).

### 3. Counterfactual Analysis
Predict: "What pathways would normalize if FLT3 mutation was corrected?"

### 4. Patient Stratification
Cluster patients based on integrated multi-omics + perturbation profiles.

### 5. Drug Target Discovery
Find genes/pathways that could restore normal state in mutated samples.

## MOVE Architecture for Your Data

```
Input Layer:
├── RNA-seq (5000 genes)
├── Methylation (5000 CpGs)
├── CNV (3000 genes)
└── Perturbations (FLT3, NPM1, Gender)

    ↓ [Modality-specific encoders]

Latent Space:
└── Shared representation (128 dims)
    - Integrates all modalities
    - Conditioned on perturbations

    ↓ [Modality-specific decoders]

Output Layer (Reconstructions):
├── RNA-seq
├── Methylation
└── CNV
```

## Configuration Options

Edit `src/train.py` to adjust:

```python
CONFIG = {
    'latent_dim': 128,              # Size of shared latent space
    'batch_size': 32,               # Batch size
    'learning_rate': 1e-3,          # Learning rate
    'num_epochs': 100,              # Training epochs
    'beta': 1.0,                    # KL divergence weight (beta-VAE)
    'encoder_layers': [512, 256],   # Hidden layers
    'decoder_layers': [256, 512],
}
```

## Expected Outputs

After training, you'll have:

1. **Trained model**: `models/move_model.pth`
2. **Latent embeddings**: `results/latent_embeddings.csv`
3. **Training logs**: Loss curves, metrics
4. **Perturbation predictions**: Counterfactual omics profiles

## Troubleshooting

### MOVE not installed
```bash
pip install move-dl
```

### GPU issues
Set `use_gpu: False` in config for CPU-only training.

### Memory errors
Reduce `batch_size` or number of features in preprocessing.

### MOVE API questions
Check the official documentation: https://move-dl.readthedocs.io/

## Next Steps

1. **Run preprocessing**: `python src/preprocessing.py`
2. **Explore the Colab tutorial**: Understand MOVE's API
3. **Adapt `train.py`**: Match MOVE's actual API (may differ from template)
4. **Train model**: `python src/train.py`
5. **Analyze results**: Visualize embeddings, test perturbations

## Citation

```
Allesøe, R.L., Lundgaard, A.T., Hernández Medina, R. et al.
Discovery of drug–omics associations in type 2 diabetes with
generative deep-learning models. Nat Biotechnol (2023).
https://doi.org/10.1038/s41587-023-01705-7
```
