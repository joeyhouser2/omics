"""Analysis utilities for trained autoencoder."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt


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


def compare_approaches(results_dir: Path):
    """Compare post-hoc VAE analysis with cVAE perturbation effects."""
    latent, perturbations = load_results(results_dir)

    print("=" * 60)
    print("POST-HOC ANALYSIS (Standard VAE)")
    print("=" * 60)
    posthoc = compute_perturbation_effects_posthoc(latent, perturbations)
    print(posthoc.to_string(index=False))
    posthoc.to_csv(results_dir / 'posthoc_analysis.csv', index=False)

    # Check for cVAE effects
    cvae_files = list(results_dir.glob('perturbation_effect_*.csv'))
    if cvae_files:
        print("\n" + "=" * 60)
        print("CONDITIONAL VAE EFFECTS")
        print("=" * 60)
        cvae_effects = analyze_cvae_effects(results_dir)
        print(cvae_effects.to_string())
        cvae_effects.to_csv(results_dir / 'cvae_effects_summary.csv')

    return posthoc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze trained autoencoder')
    parser.add_argument('--results-dir', type=str, default='../results/autoencoder',
                        help='Path to results directory')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='Dimensionality reduction method')

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    latent, perturbations = load_results(results_dir)
    print(f"Loaded latent representations: {latent.shape}")

    if perturbations is not None:
        print(f"Loaded perturbations: {perturbations.shape}")
        print(f"Perturbation columns: {list(perturbations.columns)}")

        # Run comparison analysis
        compare_approaches(results_dir)

    # Visualize
    visualize_latent_space(
        latent, perturbations,
        method=args.method,
        output_path=results_dir / f'latent_space_{args.method}.png'
    )
