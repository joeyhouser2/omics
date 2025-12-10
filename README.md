# Multi-Omics Integration with MOVE Autoencoder

This project applies the MOVE (Multi-Omics Variational autoEncoder) framework to integrate and analyze multi-omics cancer data.

## Overview

**We will be flipping mutations to 0 or 1 as our Perturbations with MOVE. We can do this with certain clinical data as well - male or female**
**FLT3_Mutated can be 0 or 1, NPM1_Mutated can be 0 or 1, Gender_Male can be 0 or 1**
**We can theoretically measure which pathways would return to normal if a mutation was no longer present**

## Dataset

The project uses LinkedOmics multi-omics data including:

- **RNA-seq** (`rna_seq.csv`) - Gene expression data
- **Methylation** (`methylation_gene_level.csv`) - DNA methylation profiles
- **miRNA** (`gene_level_mirna.cct.csv`, `mirnaisoform.csv`) - microRNA expression
- **Mutations** (`mutation_gene_level.csv`, `mutation_site_level.csv`) - Somatic mutations
- **Copy Number Variations** (`SCNV_Genelevel.csv`, `SCNV_focal_level.csv`) - Somatic CNVs
- **Clinical Data** (`clinical.csv`) - Patient metadata and outcomes

**Data source:** [Specify your data source and any relevant citations]

**[Add details about sample size, cancer type, preprocessing steps, etc.]**

## MOVE Autoencoder

MOVE is a deep learning framework designed to:
- Integrate multiple omics data modalities
- Learn shared and modality-specific latent representations
- Handle missing data across modalities
- Enable downstream analysis (clustering, classification, survival analysis)

**[Add your specific MOVE architecture details, hyperparameters, etc.]**

## Project Structure

```
omics/
├── data/                          # Raw multi-omics CSV files (input)
├── config/                        # MOVE configuration files
│   ├── data/
│   │   └── aml_omics.yaml         # Data paths and input file definitions
│   └── task/
│       ├── aml_latent_analysis.yaml
│       ├── aml_flt3_ttest.yaml
│       ├── aml_flt3_bayes.yaml
│       ├── aml_npm1_ttest.yaml
│       └── aml_npm1_bayes.yaml
├── move_data/                     # MOVE input/output directories
│   ├── raw/                       # Preprocessed TSV files for MOVE
│   ├── interim/                   # MOVE intermediate files
│   └── results/                   # MOVE output (plots, associations)
├── src/
│   ├── preprocessing.py           # Data preprocessing pipeline
│   ├── move.py                    # MOVE command runner
│   └── train.py                   # Training scripts
├── notebooks/                     # Jupyter notebooks for analysis
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/joeyhouser2/omics.git
```

2. Create a virtual environment:

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your command prompt
```

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

3. Upgrade pip (recommended):
```bash
python -m pip install --upgrade pip
```

4. Install dependencies:
```bash
pip install -r requirements.txt
pip install move-dl
```

5. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
```

### Deactivating the Virtual Environment

When you're done working on the project:
```bash
deactivate
```

### Troubleshooting

**Issue: Python not found**
- Ensure Python 3.8+ is installed: `python --version` or `python3 --version`
- Download from [python.org](https://www.python.org/downloads/)

**Issue: Activation script not found on Windows**
- Try: `venv\Scripts\Activate.ps1` (PowerShell) or `venv\Scripts\activate.bat` (CMD)

**Issue: PyTorch installation fails**
- Visit [PyTorch.org](https://pytorch.org/get-started/locally/) for platform-specific installation commands
- For CPU-only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

## Usage

### 1. Data Preprocessing

Preprocess your multi-omics data and extract perturbations:

```bash
python src/preprocessing.py
```

This script:
- Loads clinical and mutation data to extract perturbations (FLT3, NPM1, Gender)
- Loads RNA-seq, methylation, and CNV data
- Filters and selects top variable features
- Normalizes data with z-score standardization
- Aligns samples across all modalities
- Verifies all sample IDs match across datasets
- Saves MOVE-compatible files to `move_data/raw/`

**Output (in `move_data/raw/`):**

Continuous data (TSV with sample_id as first column):
- `rna_seq.tsv` - Top 5000 variable genes
- `methylation.tsv` - Top 5000 variable CpG sites
- `cnv.tsv` - Top 3000 variable genes

Categorical data (TSV with sample_id and category columns):
- `flt3_status.tsv` - FLT3 mutation status (Wild_Type/Mutated)
- `npm1_status.tsv` - NPM1 mutation status (Wild_Type/Mutated)
- `gender.tsv` - Gender (Female/Male)

Sample IDs:
- `sample_ids.txt` - List of valid sample IDs (one per line)

### 2. Run MOVE Analyses

Use the MOVE command runner:

```bash
python src/move.py <task>
```

**Available tasks:**

| Command | Description |
|---------|-------------|
| `python src/move.py latent` | Latent space analysis - train VAE and visualize |
| `python src/move.py flt3` | FLT3 association analysis (t-test) |
| `python src/move.py flt3_bayes` | FLT3 association analysis (Bayesian) |
| `python src/move.py npm1` | NPM1 association analysis (t-test) |
| `python src/move.py npm1_bayes` | NPM1 association analysis (Bayesian) |
| `python src/move.py all` | Run all t-test analyses |

**Or run MOVE directly:**

```bash
move-dl data=aml_omics task=aml_latent_analysis
move-dl data=aml_omics task=aml_flt3_ttest
move-dl data=aml_omics task=aml_npm1_bayes
```

**Override parameters from command line:**

```bash
python src/move.py latent training_loop.num_epochs=200
python src/move.py flt3 batch_size=64 training_loop.lr=1e-3
```

### 3. Output Files

**Latent space analysis** (`move_data/results/latent_space/`):
- `loss_curve.png` - Training loss over epochs (overall, KLD, BCE, SSE)
- `reconstruction_metrics.png` - Accuracy/cosine similarity per dataset
- `latent_space_tsne.png` - 2D visualization of latent space
- `feature_importance.png` - Impact of each feature on latent space
- Corresponding TSV files for each plot

**Association analysis** (`move_data/results/`):
- `results_sig_assoc.tsv` - Associated feature pairs with median p-values

**Note:** Association analyses take ~45 min on a standard laptop. Check `logs/` folder for progress.

### 4. Resources

- MOVE Documentation: https://move-dl.readthedocs.io/
- Tutorial: [Google Colab Notebook](https://colab.research.google.com/drive/1RFWNsuGymCmppPsElBvDuA9zRbGskKmi)
- Paper: [Nature Biotechnology (2023)](https://www.nature.com/articles/s41587-023-01705-7)
- GitHub: https://github.com/RasmussenLab/MOVE

## Results

**[Add your key findings, figures, and performance metrics]**

### Model Performance
- **[Reconstruction loss:]** [Add values]
- **[Clustering metrics:]** [Add NMI, ARI, etc.]
- **[Classification accuracy:]** [If applicable]

### Visualizations
**[Include key plots: UMAP/t-SNE embeddings, heatmaps, survival curves, etc.]**

## Key Findings

**[Summarize biological insights and discoveries]**

## Requirements

See `requirements.txt` for full list of dependencies.

Core packages:
- Python 3.8+
- PyTorch 2.0+
- move-dl
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## References

1. MOVE paper: https://www.nature.com/articles/s41587-022-01520-x
2. LinkedOmics: (https://www.linkedomics.org/data_download/TCGA-LAML/)
