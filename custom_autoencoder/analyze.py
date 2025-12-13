"""Analysis utilities for trained autoencoder."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import warnings

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Optional: gseapy for pathway enrichment
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False


def load_results(results_dir: Path):
    """Load latent representations and perturbations."""
    latent = pd.read_csv(results_dir / 'latent_representations.csv', index_col=0)
    pert_file = results_dir / 'perturbations.csv'
    perturbations = pd.read_csv(pert_file, index_col=0) if pert_file.exists() else None
    return latent, perturbations


def visualize_latent_space(
    latent: pd.DataFrame,
    perturbations: pd.DataFrame = None,
    method: str = 'tsne',
    output_path: Path = None
):
    """Visualize latent space with optional perturbation coloring."""
    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)

    coords = reducer.fit_transform(latent.values)

    if perturbations is not None:
        n_perts = perturbations.shape[1]
        fig, axes = plt.subplots(1, n_perts + 1, figsize=(5 * (n_perts + 1), 4))

        # Plot all samples
        axes[0].scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20)
        axes[0].set_title('All Samples')
        axes[0].set_xlabel(f'{method.upper()} 1')
        axes[0].set_ylabel(f'{method.upper()} 2')

        # Plot colored by each perturbation
        for i, col in enumerate(perturbations.columns):
            colors = perturbations[col].values
            scatter = axes[i + 1].scatter(
                coords[:, 0], coords[:, 1], c=colors, alpha=0.6, s=20, cmap='coolwarm'
            )
            axes[i + 1].set_title(f'Colored by {col}')
            axes[i + 1].set_xlabel(f'{method.upper()} 1')
            plt.colorbar(scatter, ax=axes[i + 1])

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20)
        ax.set_title('Latent Space')
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    plt.show()
    return coords


def compute_perturbation_effects_posthoc(
    latent: pd.DataFrame,
    perturbations: pd.DataFrame
) -> pd.DataFrame:
    """Post-hoc analysis: Compare latent means between perturbation groups.

    This is the standard VAE approach - train without perturbations,
    then analyze how groups separate in latent space.
    """
    results = []

    for col in perturbations.columns:
        pos_mask = perturbations[col] == 1
        neg_mask = perturbations[col] == 0

        pos_latent = latent[pos_mask]
        neg_latent = latent[neg_mask]

        # Mean difference in latent space
        pos_mean = pos_latent.mean()
        neg_mean = neg_latent.mean()
        diff = pos_mean - neg_mean

        # Euclidean distance between group centroids
        centroid_dist = np.sqrt((diff ** 2).sum())

        # T-test for each latent dimension
        significant_dims = 0
        for dim in latent.columns:
            t_stat, p_val = stats.ttest_ind(pos_latent[dim], neg_latent[dim])
            if p_val < 0.05:
                significant_dims += 1

        results.append({
            'perturbation': col,
            'n_positive': pos_mask.sum(),
            'n_negative': neg_mask.sum(),
            'centroid_distance': centroid_dist,
            'significant_dimensions': significant_dims,
            'total_dimensions': len(latent.columns)
        })

    return pd.DataFrame(results)


def analyze_cvae_effects(results_dir: Path) -> pd.DataFrame:
    """Analyze perturbation effects from Conditional VAE.

    Loads the perturbation effect files generated during cVAE training.
    """
    effects = {}
    for effect_file in results_dir.glob('perturbation_effect_*.csv'):
        name = effect_file.stem.replace('perturbation_effect_', '')
        effect = pd.read_csv(effect_file).values.flatten()
        effects[name] = {
            'mean_abs_effect': np.abs(effect).mean(),
            'max_abs_effect': np.abs(effect).max(),
            'std_effect': effect.std(),
            'n_large_effects': (np.abs(effect) > 0.1).sum()  # features with >0.1 effect
        }

    return pd.DataFrame(effects).T


# =============================================================================
# RECONSTRUCTION QUALITY METRICS
# =============================================================================

def load_feature_names(results_dir: Path) -> Optional[Dict[str, List[str]]]:
    """Load feature names from saved JSON."""
    features_file = Path(results_dir) / 'feature_names.json'
    if features_file.exists():
        with open(features_file, 'r') as f:
            return json.load(f)
    return None


def _compute_metrics_for_subset(
    original: np.ndarray,
    reconstructed: np.ndarray,
    name: str
) -> Dict:
    """Compute all metrics for a subset of features."""
    n_samples, n_features = original.shape

    # Flatten for overall metrics
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()

    # MSE and MAE
    mse = mean_squared_error(orig_flat, recon_flat)
    mae = mean_absolute_error(orig_flat, recon_flat)

    # R-squared
    r2 = r2_score(orig_flat, recon_flat)

    # Pearson correlation (average across features)
    pearson_per_feature = []
    for i in range(n_features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.std(original[:, i]) > 0 and np.std(reconstructed[:, i]) > 0:
                r, _ = pearsonr(original[:, i], reconstructed[:, i])
                if not np.isnan(r):
                    pearson_per_feature.append(r)
    pearson_mean = np.mean(pearson_per_feature) if pearson_per_feature else np.nan

    # Spearman correlation (average across features)
    spearman_per_feature = []
    for i in range(n_features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.std(original[:, i]) > 0 and np.std(reconstructed[:, i]) > 0:
                rho, _ = spearmanr(original[:, i], reconstructed[:, i])
                if not np.isnan(rho):
                    spearman_per_feature.append(rho)
    spearman_mean = np.mean(spearman_per_feature) if spearman_per_feature else np.nan

    # Cosine similarity (average across samples)
    cos_sims = []
    for i in range(n_samples):
        sim = cosine_similarity(original[i:i+1], reconstructed[i:i+1])[0, 0]
        cos_sims.append(sim)
    cos_sim_mean = np.mean(cos_sims)

    return {
        'modality': name,
        'n_features': n_features,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'pearson_mean': pearson_mean,
        'spearman_mean': spearman_mean,
        'cosine_similarity': cos_sim_mean
    }


def compute_reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: Dict[str, List[str]] = None
) -> pd.DataFrame:
    """Compute comprehensive reconstruction quality metrics.

    Args:
        original: Original data [n_samples, n_features]
        reconstructed: Reconstructed data [n_samples, n_features]
        feature_names: Dict mapping modality -> list of feature names

    Returns:
        DataFrame with metrics per modality and overall
    """
    results = []

    # Compute overall metrics
    overall_metrics = _compute_metrics_for_subset(original, reconstructed, "overall")
    results.append(overall_metrics)

    # Compute per-modality metrics if feature names provided
    if feature_names:
        # Features are concatenated in sorted order: cnv, methylation, rna_seq
        start_idx = 0
        for modality in sorted(feature_names.keys()):
            n_features = len(feature_names[modality])
            end_idx = start_idx + n_features

            orig_subset = original[:, start_idx:end_idx]
            recon_subset = reconstructed[:, start_idx:end_idx]

            modality_metrics = _compute_metrics_for_subset(
                orig_subset, recon_subset, modality
            )
            results.append(modality_metrics)
            start_idx = end_idx

    return pd.DataFrame(results)


def compute_per_sample_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_ids: List[str],
    feature_names: Dict[str, List[str]] = None
) -> pd.DataFrame:
    """Compute reconstruction metrics per sample.

    Useful for identifying samples with poor reconstruction.
    """
    n_samples = original.shape[0]
    results = []

    for i in range(n_samples):
        sample_metrics = {
            'sample_id': sample_ids[i] if sample_ids else f'sample_{i}',
            'mse': mean_squared_error(original[i], reconstructed[i]),
            'mae': mean_absolute_error(original[i], reconstructed[i]),
        }

        # Per-modality MSE for this sample
        if feature_names:
            start_idx = 0
            for modality in sorted(feature_names.keys()):
                n_features = len(feature_names[modality])
                end_idx = start_idx + n_features

                sample_metrics[f'mse_{modality}'] = mean_squared_error(
                    original[i, start_idx:end_idx],
                    reconstructed[i, start_idx:end_idx]
                )
                start_idx = end_idx

        results.append(sample_metrics)

    return pd.DataFrame(results)


def analyze_reconstruction_quality(results_dir: Path) -> Optional[pd.DataFrame]:
    """Load reconstructions and compute quality metrics."""
    results_dir = Path(results_dir)
    original_file = results_dir / 'original_data.npy'
    recon_file = results_dir / 'reconstructions.npy'

    if not original_file.exists() or not recon_file.exists():
        print("  Reconstruction files not found. Retrain model to generate them.")
        return None

    original = np.load(original_file)
    reconstructed = np.load(recon_file)

    feature_names = load_feature_names(results_dir)

    # Compute per-modality metrics
    metrics = compute_reconstruction_metrics(original, reconstructed, feature_names)
    metrics.to_csv(results_dir / 'reconstruction_metrics.csv', index=False)

    # Compute per-sample metrics
    sample_ids_file = results_dir / 'latent_representations.csv'
    sample_ids = None
    if sample_ids_file.exists():
        latent_df = pd.read_csv(sample_ids_file, index_col=0)
        sample_ids = list(latent_df.index)

    per_sample = compute_per_sample_metrics(original, reconstructed, sample_ids, feature_names)
    per_sample.to_csv(results_dir / 'reconstruction_per_sample.csv', index=False)

    return metrics


# =============================================================================
# PATHWAY ENRICHMENT ANALYSIS
# =============================================================================

def run_pathway_enrichment(
    gene_list: List[str],
    gene_sets: List[str] = None,
    organism: str = 'human',
    output_dir: Path = None,
    description: str = 'enrichment'
) -> Dict[str, pd.DataFrame]:
    """Run pathway enrichment analysis using Enrichr.

    Args:
        gene_list: List of gene symbols (HGNC)
        gene_sets: Gene set libraries to query (default: GO_BP, KEGG, Reactome)
        organism: 'human' or 'mouse'
        output_dir: Directory to save results
        description: Description for output files

    Returns:
        Dict mapping gene_set -> enrichment results DataFrame
    """
    if not GSEAPY_AVAILABLE:
        raise ImportError("gseapy is required for pathway enrichment. Install with: pip install gseapy")

    if gene_sets is None:
        gene_sets = [
            'GO_Biological_Process_2023',
            'KEGG_2021_Human',
            'Reactome_2022'
        ]

    results = {}

    for gs in gene_sets:
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gs,
                organism=organism,
                outdir=None,  # Don't auto-save
                cutoff=0.05
            )

            results[gs] = enr.results

            if output_dir:
                output_dir = Path(output_dir)
                enr.results.to_csv(
                    output_dir / f'{description}_{gs}.csv',
                    index=False
                )
                n_sig = len(enr.results[enr.results['Adjusted P-value'] < 0.05])
                print(f"    {gs}: {n_sig} significant pathways")

        except Exception as e:
            print(f"    Warning: Failed to query {gs}: {e}")
            results[gs] = pd.DataFrame()

    return results


def get_top_effect_genes(
    results_dir: Path,
    feature_names: Dict[str, List[str]],
    top_n: int = 100,
    effect_threshold: float = None,
    modalities: List[str] = None
) -> Dict[str, Dict[str, List[str]]]:
    """Extract top genes affected by each perturbation.

    Args:
        results_dir: Directory containing perturbation_effect_*.csv files
        feature_names: Dict mapping modality -> feature names
        top_n: Number of top genes to return (by absolute effect)
        effect_threshold: Alternative to top_n - return genes with |effect| > threshold
        modalities: Which modalities to include (default: ['rna_seq', 'cnv'])

    Returns:
        Dict[perturbation_name][modality] -> list of gene names
    """
    results_dir = Path(results_dir)

    if modalities is None:
        modalities = ['rna_seq', 'cnv']  # Skip methylation by default

    # Build feature index mapping
    all_features = []
    feature_to_modality = {}
    idx = 0
    for modality in sorted(feature_names.keys()):
        for feat in feature_names[modality]:
            all_features.append(feat)
            feature_to_modality[idx] = modality
            idx += 1

    results = {}

    for effect_file in results_dir.glob('perturbation_effect_*.csv'):
        pert_name = effect_file.stem.replace('perturbation_effect_', '')
        effect_df = pd.read_csv(effect_file)
        effects = effect_df.values.flatten()

        results[pert_name] = {}

        for target_modality in modalities:
            if target_modality not in feature_names:
                continue

            # Get indices for this modality
            modality_indices = [
                i for i in range(len(effects))
                if feature_to_modality.get(i) == target_modality
            ]

            if not modality_indices:
                continue

            # Get effects for this modality
            modality_effects = np.array([effects[i] for i in modality_indices])
            modality_features = [all_features[i] for i in modality_indices]

            # Get top genes by absolute effect
            abs_effects = np.abs(modality_effects)

            if effect_threshold is not None:
                top_indices = np.where(abs_effects > effect_threshold)[0]
            else:
                n_select = min(top_n, len(abs_effects))
                top_indices = np.argsort(abs_effects)[-n_select:][::-1]

            top_genes = [modality_features[i] for i in top_indices]
            results[pert_name][target_modality] = top_genes

    return results


def _save_enrichment_summary(
    all_results: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path
):
    """Save a summary of top enriched pathways."""
    summary_rows = []

    for pert, gene_set_results in all_results.items():
        for gs, df in gene_set_results.items():
            if df.empty:
                continue
            # Get top 5 pathways by adjusted p-value
            if 'Adjusted P-value' not in df.columns:
                continue
            top = df.nsmallest(5, 'Adjusted P-value')
            for _, row in top.iterrows():
                summary_rows.append({
                    'perturbation': pert,
                    'gene_set': gs,
                    'term': row.get('Term', ''),
                    'adjusted_pvalue': row.get('Adjusted P-value', np.nan),
                    'combined_score': row.get('Combined Score', np.nan),
                    'genes': row.get('Genes', '')
                })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / 'enrichment_summary.csv', index=False)
        print(f"\nSaved enrichment summary to {output_dir / 'enrichment_summary.csv'}")


def run_perturbation_pathway_analysis(
    results_dir: Path,
    feature_names: Dict[str, List[str]],
    top_n_genes: int = 100,
    gene_sets: List[str] = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Run full pathway enrichment on perturbation effects.

    Args:
        results_dir: Directory with perturbation_effect_*.csv files
        feature_names: Dict mapping modality -> feature names
        top_n_genes: Number of top-effect genes to use
        gene_sets: Enrichr gene set libraries

    Returns:
        Nested dict: results[perturbation][gene_set] -> DataFrame
    """
    results_dir = Path(results_dir)

    print("\n" + "=" * 60)
    print("PATHWAY ENRICHMENT ANALYSIS")
    print("=" * 60)

    if not GSEAPY_AVAILABLE:
        print("  gseapy not installed. Install with: pip install gseapy")
        return {}

    # Create output directory for enrichment results
    enrichment_dir = results_dir / 'pathway_enrichment'
    enrichment_dir.mkdir(exist_ok=True)

    # Get top genes for each perturbation
    top_genes = get_top_effect_genes(
        results_dir,
        feature_names,
        top_n=top_n_genes,
        modalities=['rna_seq', 'cnv']
    )

    all_results = {}

    for pert_name, modality_genes in top_genes.items():
        print(f"\nPerturbation: {pert_name}")
        all_results[pert_name] = {}

        # Combine genes from RNA-seq and CNV
        combined_genes = []
        for modality, genes in modality_genes.items():
            combined_genes.extend(genes)
            print(f"  {modality}: {len(genes)} top-effect genes")

        # Remove duplicates while preserving order
        unique_genes = list(dict.fromkeys(combined_genes))
        print(f"  Combined unique genes: {len(unique_genes)}")

        if len(unique_genes) < 10:
            print("  Skipping enrichment (too few genes)")
            continue

        # Run enrichment
        print("  Running Enrichr queries...")
        enrichment = run_pathway_enrichment(
            unique_genes,
            gene_sets=gene_sets,
            output_dir=enrichment_dir,
            description=f'{pert_name}_combined'
        )

        all_results[pert_name] = enrichment

    # Save summary
    _save_enrichment_summary(all_results, enrichment_dir)

    return all_results


def compare_approaches(results_dir: Path, run_enrichment: bool = True):
    """Compare post-hoc VAE analysis with cVAE perturbation effects.

    Enhanced version with reconstruction metrics and pathway enrichment.

    Args:
        results_dir: Directory containing model outputs
        run_enrichment: Whether to run pathway enrichment (requires gseapy)
    """
    results_dir = Path(results_dir)
    latent, perturbations = load_results(results_dir)

    print("=" * 60)
    print("POST-HOC ANALYSIS (Standard VAE)")
    print("=" * 60)
    posthoc = compute_perturbation_effects_posthoc(latent, perturbations)
    print(posthoc.to_string(index=False))
    posthoc.to_csv(results_dir / 'posthoc_analysis.csv', index=False)

    # === RECONSTRUCTION METRICS ===
    print("\n" + "=" * 60)
    print("RECONSTRUCTION QUALITY METRICS")
    print("=" * 60)

    recon_metrics = analyze_reconstruction_quality(results_dir)
    if recon_metrics is not None:
        print(recon_metrics.to_string(index=False))
        print(f"\nSaved to: {results_dir / 'reconstruction_metrics.csv'}")
        print(f"Per-sample: {results_dir / 'reconstruction_per_sample.csv'}")

    # === CVAE EFFECTS ===
    cvae_files = list(results_dir.glob('perturbation_effect_*.csv'))
    if cvae_files:
        print("\n" + "=" * 60)
        print("CONDITIONAL VAE EFFECTS")
        print("=" * 60)
        cvae_effects = analyze_cvae_effects(results_dir)
        print(cvae_effects.to_string())
        cvae_effects.to_csv(results_dir / 'cvae_effects_summary.csv')

        # === PATHWAY ENRICHMENT ===
        if run_enrichment:
            feature_names = load_feature_names(results_dir)
            if feature_names:
                run_perturbation_pathway_analysis(
                    results_dir,
                    feature_names,
                    top_n_genes=100
                )
            else:
                print("\n  Feature names not found. Cannot run pathway enrichment.")
                print("  Retrain model to generate feature_names.json")

    return posthoc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze trained autoencoder')
    parser.add_argument('--results-dir', type=str, default='../results/autoencoder',
                        help='Path to results directory')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='Dimensionality reduction method')
    parser.add_argument('--no-enrichment', action='store_true',
                        help='Skip pathway enrichment analysis')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip latent space visualization')

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    latent, perturbations = load_results(results_dir)
    print(f"Loaded latent representations: {latent.shape}")

    if perturbations is not None:
        print(f"Loaded perturbations: {perturbations.shape}")
        print(f"Perturbation columns: {list(perturbations.columns)}")

        # Run comparison analysis with reconstruction metrics and pathway enrichment
        compare_approaches(results_dir, run_enrichment=not args.no_enrichment)

    # Visualize
    if not args.no_visualize:
        visualize_latent_space(
            latent, perturbations,
            method=args.method,
            output_path=results_dir / f'latent_space_{args.method}.png'
        )
