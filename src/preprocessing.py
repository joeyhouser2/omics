"""Not totally sure if this is needed or correct yet"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')
DATA_DIR = Path('data')
RAW_DATA_DIR = Path('move_data/raw')  # All MOVE input files go here
INTERIM_DIR = Path('move_data/interim')
RESULTS_DIR = Path('move_data/results')
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Processing parameters
TOP_GENES_RNA = 5000        # Top variable genes for RNA-seq
TOP_GENES_METH = 5000       # Top variable CpGs for methylation
TOP_GENES_CNV = 3000        # Top variable genes for CNV
MIN_EXPRESSION = 1.0        # Minimum expression threshold for gene filtering


class MultiOmicsPreprocessor:
    """Preprocessor for multi-omics data with MOVE perturbations."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.common_samples = None
        self.perturbations = None

    def load_perturbations(self) -> pd.DataFrame:
        print("loading perturbation data...")
        clinical = pd.read_csv(self.data_dir / 'clinical.csv', index_col=0)
        mutation = pd.read_csv(self.data_dir / 'mutation_gene_level.csv', index_col=0)

        # get gender and convert to Gender_Male
        gender = clinical.loc['gender']
        gender_male = (gender == 'male').astype(int)

        # get FLT3 and NPM1 mutation status
        flt3_mutated = mutation.loc['FLT3'].astype(int)
        npm1_mutated = mutation.loc['NPM1'].astype(int)
        # our identified perturbations
        perturbations = pd.DataFrame({
            'FLT3_Mutated': flt3_mutated,
            'NPM1_Mutated': npm1_mutated,
            'Gender_Male': gender_male
        })

        print(f"{len(perturbations)} samples with perturbations")
        print(f"FLT3_Mutated: {flt3_mutated.sum()} positive / {len(flt3_mutated)} total")
        print(f"NPM1_Mutated: {npm1_mutated.sum()} positive / {len(npm1_mutated)} total")
        print(f"Gender_Male: {gender_male.sum()} males / {len(gender_male)} total")
        self.perturbations = perturbations
        return perturbations

    def load_rna_seq(self) -> pd.DataFrame:
        print("\nLoading RNA-seq data...")

        # Load RNA-seq (rows are genes, columns are samples)
        rna = pd.read_csv(self.data_dir / 'rna_seq.csv', index_col=0)

        # Transpose so samples are rows
        rna = rna.T

        print(f"  Loaded {rna.shape[0]} samples x {rna.shape[1]} genes")

        # Filter low-expression genes (mean expression > threshold)
        mean_expr = rna.mean(axis=0)
        high_expr_genes = mean_expr[mean_expr > MIN_EXPRESSION].index
        rna = rna[high_expr_genes]
        print(f"  After filtering: {rna.shape[1]} genes (mean expr > {MIN_EXPRESSION})")

        # Select top variable genes
        gene_var = rna.var(axis=0)
        top_genes = gene_var.nlargest(TOP_GENES_RNA).index
        rna = rna[top_genes]
        print(f"  Selected top {len(top_genes)} variable genes")

        return rna

    def load_methylation(self) -> pd.DataFrame:
        print("\nLoading methylation data...")

        # Load methylation (rows are CpG sites, columns are samples)
        meth = pd.read_csv(self.data_dir / 'methylation_gene_level.csv', index_col=0)

        # Transpose so samples are rows
        meth = meth.T

        print(f"  Loaded {meth.shape[0]} samples x {meth.shape[1]} CpG sites")

        # Handle missing values (fill with median)
        meth = meth.fillna(meth.median())

        # Select top variable CpG sites
        cpg_var = meth.var(axis=0)
        top_cpgs = cpg_var.nlargest(TOP_GENES_METH).index
        meth = meth[top_cpgs]
        print(f"  Selected top {len(top_cpgs)} variable CpG sites")

        return meth

    def load_cnv(self) -> pd.DataFrame:
        print("Loading CNV data...")

        # Load CNV (rows are genes, columns are samples)
        cnv = pd.read_csv(self.data_dir / 'SCNV_Genelevel.csv', index_col=0)

        # Transpose so samples are rows
        cnv = cnv.T

        print(f"  Loaded {cnv.shape[0]} samples x {cnv.shape[1]} genes")

        # Handle missing values
        cnv = cnv.fillna(cnv.median())

        # Select top variable genes
        gene_var = cnv.var(axis=0)
        top_genes = gene_var.nlargest(TOP_GENES_CNV).index
        cnv = cnv[top_genes]
        print(f"  Selected top {len(top_genes)} variable genes")

        return cnv

    def align_samples(self, *dataframes: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        print("align samples")

        # Find common samples
        common_idx = set(dataframes[0].index)
        for df in dataframes[1:]:
            common_idx = common_idx.intersection(set(df.index))

        common_idx = sorted(list(common_idx))
        self.common_samples = common_idx

        print(f"  Found {len(common_idx)} common samples across all datasets")

        # Align all dataframes
        aligned = tuple(df.loc[common_idx] for df in dataframes)

        return aligned

    def normalize_data(self, *dataframes: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """z score norm"""
        print("normalizing data")
        normalized = []
        for i, df in enumerate(dataframes):
            scaler = StandardScaler()
            normalized_values = scaler.fit_transform(df)
            normalized_df = pd.DataFrame(normalized_values,index=df.index,columns=df.columns)
            normalized.append(normalized_df)
            print(f"  Normalized dataset {i+1}: shape {normalized_df.shape}")

        return tuple(normalized)

    def save_processed_data(self, rna: pd.DataFrame, meth: pd.DataFrame, cnv: pd.DataFrame, perturbations: pd.DataFrame):
        print("\nSaving processed data to raw_data directory...")

        # Verify all datasets have the same samples
        assert list(rna.index) == list(meth.index) == list(cnv.index) == list(perturbations.index), \
            "Sample IDs must match across all datasets!"

        # Save sample IDs file first (one ID per line, no header)
        # These IDs must appear in every dataset
        with open(RAW_DATA_DIR / 'sample_ids.txt', 'w') as f:
            f.write('\n'.join(rna.index))
        print(f"  Saved sample_ids.txt: {len(rna.index)} samples")

        # Save continuous data as TSV with sample IDs as first column
        rna.to_csv(RAW_DATA_DIR / 'rna_seq.tsv', sep='\t', index_label='sample_id')
        print(f"  Saved rna_seq.tsv: {rna.shape}")

        meth.to_csv(RAW_DATA_DIR / 'methylation.tsv', sep='\t', index_label='sample_id')
        print(f"  Saved methylation.tsv: {meth.shape}")

        cnv.to_csv(RAW_DATA_DIR / 'cnv.tsv', sep='\t', index_label='sample_id')
        print(f"  Saved cnv.tsv: {cnv.shape}")

        # Save each categorical/binary variable as a separate TSV file
        # Format: sample_id \t category_value (two columns only)
        for col in perturbations.columns:
            # Convert binary 0/1 to descriptive categories
            if col == 'FLT3_Mutated':
                cat_series = perturbations[col].map({0: 'Wild_Type', 1: 'Mutated'})
                col_name = 'FLT3_Status'
            elif col == 'NPM1_Mutated':
                cat_series = perturbations[col].map({0: 'Wild_Type', 1: 'Mutated'})
                col_name = 'NPM1_Status'
            elif col == 'Gender_Male':
                cat_series = perturbations[col].map({0: 'Female', 1: 'Male'})
                col_name = 'Gender'
            else:
                cat_series = perturbations[col]
                col_name = col

            # Create DataFrame with sample_id and category
            cat_df = pd.DataFrame({
                'sample_id': perturbations.index,
                col_name: cat_series.values
            })

            # Save as TSV
            cat_df.to_csv(RAW_DATA_DIR / f'{col_name.lower()}.tsv', sep='\t', index=False)
            print(f"  Saved {col_name.lower()}.tsv: {len(cat_df)} samples")

        # Save summary metadata
        summary = {
            'n_samples': len(rna),
            'n_rna_features': rna.shape[1],
            'n_meth_features': meth.shape[1],
            'n_cnv_features': cnv.shape[1],
            'n_perturbations': perturbations.shape[1],
            'flt3_positive': int(perturbations['FLT3_Mutated'].sum()),
            'npm1_positive': int(perturbations['NPM1_Mutated'].sum()),
            'male_count': int(perturbations['Gender_Male'].sum())
        }

        pd.Series(summary).to_csv(RAW_DATA_DIR / 'summary_stats.csv')
        print("\n=== Summary Statistics ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    def run_full_pipeline(self):
        print("="*60)
        print("Multi-Omics Preprocessing Pipeline for MOVE")
        print("="*60)

        # Load perturbations
        perturbations = self.load_perturbations()

        # Load omics data
        rna = self.load_rna_seq()
        meth = self.load_methylation()
        cnv = self.load_cnv()

        # Align samples
        perturbations, rna, meth, cnv = self.align_samples(
            perturbations, rna, meth, cnv
        )

        # Normalize omics data (but NOT perturbations - they're binary!)
        rna, meth, cnv = self.normalize_data(rna, meth, cnv)
        self.save_processed_data(rna, meth, cnv, perturbations)
        print("done processing")
        return rna, meth, cnv, perturbations


def main():
    preprocessor = MultiOmicsPreprocessor()
    rna, meth, cnv, perturbations = preprocessor.run_full_pipeline() # may need to utilize

    print("\nProcessed data saved to:", RAW_DATA_DIR.absolute())
    print("\nYou can now use this data to train the MOVE model!")


if __name__ == "__main__":
    main()
