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
├── data/                          # Raw multi-omics CSV files
├── notebooks/                     # Jupyter notebooks for analysis
│   └── [exploratory_analysis.ipynb]
│   └── [move_training.ipynb]
├── src/                          # Source code
│   ├── preprocessing.py          # Data preprocessing and normalization
│   ├── move_model.py             # MOVE autoencoder implementation
│   ├── train.py                  # Training scripts
│   └── evaluate.py               # Evaluation and visualization
├── models/                       # Saved trained models
├── results/                      # Output figures and results
├── split_excel.py                # Utility to extract data from Excel
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

**[Create the above folders and scripts as needed]**

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
- Saves processed data to `processed_data/`

**Output:**
- `rna_seq_processed.csv` - Top 5000 variable genes
- `methylation_processed.csv` - Top 5000 variable CpG sites
- `cnv_processed.csv` - Top 3000 variable genes
- `perturbations.csv` - Binary perturbation variables (FLT3, NPM1, Gender)
- `summary_stats.csv` - Dataset statistics

### 2. Train MOVE Model

Train the MOVE autoencoder with perturbations:

```bash
python src/train.py
```

This uses the official **MOVE (Multi-Omics Variational autoEncoder)** package from [RasmussenLab/MOVE](https://github.com/RasmussenLab/MOVE).

**Training Configuration:**
- Latent dimension: 128
- Encoder layers: [512, 256]
- Decoder layers: [256, 512]
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 100

**What it does:**
1. Loads preprocessed multi-omics data
2. Encodes each modality (RNA, methylation, CNV) separately
3. Integrates with perturbations (FLT3, NPM1, Gender)
4. Creates shared latent representation
5. Reconstructs each modality
6. Saves trained model and latent embeddings

**Resources:**
- Documentation: https://move-dl.readthedocs.io/
- Tutorial: [Google Colab Notebook](https://colab.research.google.com/drive/1RFWNsuGymCmppPsElBvDuA9zRbGskKmi)
- Paper: [Nature Biotechnology (2023)](https://www.nature.com/articles/s41587-023-01705-7)

### 3. Evaluate Results

```bash
python src/eval.py --model_path models/move_best.pth
```

**[Describe evaluation metrics and visualizations]**

### 4. Notebooks

Explore the analysis notebooks:
- `notebooks/exploratory_analysis.ipynb` - Data exploration and quality control
- `notebooks/move_training.ipynb` - Interactive model training and tuning

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
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## References

1. MOVE paper: https://www.nature.com/articles/s41587-022-01520-x
2. LinkedOmics: (https://www.linkedomics.org/data_download/TCGA-LAML/)
