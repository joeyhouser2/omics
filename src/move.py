"""
MOVE command runner for AML multi-omics analysis.

Usage:
    python src/move.py latent              # Latent space analysis
    python src/move.py flt3                # FLT3 association (t-test)
    python src/move.py flt3_bayes          # FLT3 association (Bayesian)
    python src/move.py npm1                # NPM1 association (t-test)
    python src/move.py npm1_bayes          # NPM1 association (Bayesian)
    python src/move.py all                 # Run all t-test analyses

Note: Association analyses can take ~45 min on a standard laptop.
      Check logs/ folder for progress.

Output:
    - results_sig_assoc.tsv: Associated feature pairs with median p-values
"""

import subprocess
import sys
from pathlib import Path


# Available tasks
TASKS = {
    'latent': 'aml_latent_analysis',
    'flt3': 'aml_flt3_ttest',
    'flt3_bayes': 'aml_flt3_bayes',
    'npm1': 'aml_npm1_ttest',
    'npm1_bayes': 'aml_npm1_bayes',
}

DATA_CONFIG = 'aml_omics'


def run_move(task_name: str, **kwargs):
    """Run a MOVE task."""
    if task_name not in TASKS:
        print(f"Unknown task: {task_name}")
        print(f"Available tasks: {list(TASKS.keys())}")
        return False

    task_config = TASKS[task_name]

    # Build command
    cmd = ['move-dl', f'data={DATA_CONFIG}', f'task={task_config}']

    # Add any override arguments
    for key, value in kwargs.items():
        cmd.append(f'{key}={value}')

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_latent_analysis(**kwargs):
    """Train VAE and analyze latent space.

    Outputs (move_data/results/latent_space/):
        - loss_curve.png: Training loss over epochs
        - reconstruction_metrics.png: Accuracy/cosine similarity per dataset
        - latent_space_tsne.png: 2D visualization of latent space
        - feature_importance.png: Impact of each feature on latent space
        - Corresponding TSV files for each plot
    """
    return run_move('latent', **kwargs)


def run_flt3_association(**kwargs):
    """Identify omics features associated with FLT3 mutation using t-test."""
    return run_move('flt3', **kwargs)


def run_npm1_association(**kwargs):
    """Identify omics features associated with NPM1 mutation using t-test."""
    return run_move('npm1', **kwargs)


def run_all(**kwargs):
    """Run all analyses."""
    print("Running all MOVE analyses...")
    print()

    success = True

    print("[1/3] Latent space analysis")
    if not run_latent_analysis(**kwargs):
        success = False
    print()

    print("[2/3] FLT3 association analysis")
    if not run_flt3_association(**kwargs):
        success = False
    print()

    print("[3/3] NPM1 association analysis")
    if not run_npm1_association(**kwargs):
        success = False
    print()

    if success:
        print("All analyses completed successfully!")
    else:
        print("Some analyses failed. Check output above.")

    return success


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Available tasks:")
        for name, config in TASKS.items():
            print(f"  {name}: {config}")
        sys.exit(1)

    task = sys.argv[1].lower()

    # Parse additional arguments (key=value format)
    kwargs = {}
    for arg in sys.argv[2:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            kwargs[key] = value

    if task == 'all':
        success = run_all(**kwargs)
    elif task in TASKS:
        success = run_move(task, **kwargs)
    else:
        print(f"Unknown task: {task}")
        print(f"Available: {list(TASKS.keys()) + ['all']}")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
