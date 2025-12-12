# Custom Multi-Omics Autoencoder

A PyTorch implementation of variational and conditional autoencoders for multi-omics data integration and perturbation analysis.

## Overview

This module provides three autoencoder architectures for learning joint latent representations from multi-omics cancer data (RNA-seq, methylation, CNV):

| Model | Description |
|-------|-------------|
| **VAE** | Variational Autoencoder - learns a regularized latent space |
| **AE** | Standard Autoencoder - deterministic encoding |
| **CVAE** | Conditional VAE - incorporates perturbations (mutations, gender) for counterfactual analysis |

## Quick Start

```bash
# From project root
python custom_autoencoder/run.py --model-type vae --latent-dim 64 --epochs 200 --data-dir move_data/raw
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-type` | `vae` | Model architecture: `vae`, `ae`, or `cvae` |
| `--data-dir` | `../move_data/raw` | Path to preprocessed data directory |
| `--output-dir` | `../results/autoencoder` | Path to save results |
| `--latent-dim` | `64` | Latent space dimension |
| `--hidden-dims` | `1024 512 256` | Hidden layer dimensions (space-separated) |
| `--batch-size` | `32` | Training batch size |
| `--epochs` | `200` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--beta` | `1.0` | KL divergence weight (VAE/CVAE only) |
| `--dropout` | `0.2` | Dropout rate |
| `--seed` | `42` | Random seed for reproducibility |

## Example Configurations

### Standard VAE
```bash
python custom_autoencoder/run.py --model-type vae --latent-dim 64 --epochs 200 --data-dir move_data/raw
```

### Conditional VAE (with perturbation analysis)
```bash
python custom_autoencoder/run.py --model-type cvae --latent-dim 64 --beta 1.0 --data-dir move_data/raw
```

### Standard Autoencoder (no KL regularization)
```bash
python custom_autoencoder/run.py --model-type ae --latent-dim 64 --data-dir move_data/raw
```

### Beta-VAE (stronger disentanglement)
```bash
python custom_autoencoder/run.py --model-type vae --beta 4.0 --latent-dim 32 --data-dir move_data/raw
```

### Custom architecture
```bash
python custom_autoencoder/run.py --model-type vae --hidden-dims 512 256 128 --latent-dim 32 --data-dir move_data/raw
```

### Larger batch size with adjusted learning rate
```bash
python custom_autoencoder/run.py --model-type vae --batch-size 64 --lr 0.0005 --data-dir move_data/raw
```

## Input Data

The module expects preprocessed data in the `--data-dir` directory:

**Required files:**
- `rna_seq.tsv` - RNA-seq expression (samples x features)
- `methylation.tsv` - Methylation data (samples x features)
- `cnv.tsv` - Copy number variation (samples x features)

**Optional files (for CVAE):**
- `flt3_status.tsv` - FLT3 mutation status
- `npm1_status.tsv` - NPM1 mutation status
- `gender.tsv` - Gender information
- `sample_ids.txt` - Sample identifiers

Run `python src/preprocessing.py` to generate these files from raw data.

## Output Files

Results are saved to `--output-dir` (default: `results/autoencoder/`):

```
results/autoencoder/
├── checkpoints/
│   ├── best_model.pt          # Best model weights
│   └── training_history.json  # Loss history
├── latent_representations.csv # Learned latent space (samples x latent_dim)
├── perturbations.csv          # Perturbation labels for each sample
└── perturbation_effect_*.csv  # CVAE only: effect of each perturbation
```

## Analysis

After training, analyze the latent space:

```bash
python custom_autoencoder/analyze.py --results-dir results/autoencoder --method tsne
```

### Analysis options:
- `--method tsne` - t-SNE visualization (default)
- `--method pca` - PCA visualization

This generates:
- 2D latent space visualizations colored by perturbation status
- Statistical tests comparing latent dimensions between groups

## Module Structure

```
custom_autoencoder/
├── run.py       # Main training script (entry point)
├── model.py     # Model architectures (VAE, AE, CVAE, Encoder, Decoder)
├── trainer.py   # Training loops (VAETrainer, AETrainer, CVAETrainer)
├── dataset.py   # Data loading (MultiOmicsDataset)
├── analyze.py   # Post-hoc analysis utilities
└── __init__.py  # Package exports
```

## Model Details

### VAE (Variational Autoencoder)
- Encoder maps input to mean (μ) and log-variance (log σ²)
- Reparameterization trick for backpropagation through sampling
- Loss = Reconstruction (MSE) + β * KL divergence

### CVAE (Conditional VAE)
- Conditions on perturbation variables (FLT3, NPM1, Gender)
- Enables counterfactual generation: "What would this sample look like without the mutation?"
- Computes perturbation effects on the reconstructed omics data

### AE (Standard Autoencoder)
- Deterministic encoding (no sampling)
- Loss = Reconstruction (MSE) only
- Useful as a baseline comparison

## Training Features

- **Early stopping**: Stops if validation loss doesn't improve for 20 epochs
- **Learning rate scheduling**: Reduces LR by 0.5x after 10 epochs of no improvement
- **Checkpointing**: Saves best model based on validation loss
- **GPU support**: Automatically uses CUDA if available
