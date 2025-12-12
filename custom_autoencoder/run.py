"""Main script to train the multi-omics VAE."""
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from dataset import MultiOmicsDataset, create_data_loaders
from model import MultiOmicsVAE, MultiOmicsAE, ConditionalVAE
from trainer import VAETrainer, AETrainer, CVAETrainer


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Omics VAE')
    parser.add_argument('--data-dir', type=str, default='../move_data/raw',
                        help='Path to preprocessed data directory')
    parser.add_argument('--output-dir', type=str, default='../results/autoencoder',
                        help='Path to save results')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Latent space dimension')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[1024, 512, 256],
                        help='Hidden layer dimensions')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta parameter for KL divergence weight (beta-VAE)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--model-type', type=str, default='vae',
                        choices=['vae', 'ae', 'cvae'],
                        help='Model type: vae, ae (standard), or cvae (conditional)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Omics Autoencoder Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model type: {args.model_type.upper()}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Hidden dimensions: {args.hidden_dims}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    if args.model_type in ['vae', 'cvae']:
        print(f"Beta (KL weight): {args.beta}")
    print("=" * 60)

    # Create data loaders
    train_loader, val_loader, dataset = create_data_loaders(
        data_dir,
        batch_size=args.batch_size,
        train_split=0.8,
        random_seed=args.seed
    )

    input_dim = dataset.get_input_dim()
    feature_dims = dataset.get_feature_dims()
    n_perturbations = dataset.perturbations.shape[1] if dataset.perturbations is not None else 0

    print(f"\nInput dimension: {input_dim}")
    print(f"Feature dimensions: {feature_dims}")
    print(f"Number of perturbations: {n_perturbations}")

    # Create model based on type
    if args.model_type == 'vae':
        model = MultiOmicsVAE(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
            feature_dims=feature_dims
        )
        trainer = VAETrainer(
            model,
            learning_rate=args.lr,
            beta=args.beta,
            checkpoint_dir=output_dir / 'checkpoints'
        )
    elif args.model_type == 'cvae':
        if n_perturbations == 0:
            raise ValueError("Conditional VAE requires perturbation data!")
        model = ConditionalVAE(
            input_dim=input_dim,
            n_perturbations=n_perturbations,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
            feature_dims=feature_dims
        )
        trainer = CVAETrainer(
            model,
            learning_rate=args.lr,
            beta=args.beta,
            checkpoint_dir=output_dir / 'checkpoints'
        )
    else:  # ae
        model = MultiOmicsAE(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim,
            dropout=args.dropout
        )
        trainer = AETrainer(
            model,
            learning_rate=args.lr,
            checkpoint_dir=output_dir / 'checkpoints'
        )

    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        early_stopping_patience=20
    )

    # Extract and save latent representations
    print("\nExtracting latent representations...")
    full_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    if args.model_type == 'vae':
        z, mu, logvar = trainer.get_latent_representations(full_loader)
        latent_df = pd.DataFrame(
            mu,
            index=dataset.sample_ids,
            columns=[f'latent_{i}' for i in range(args.latent_dim)]
        )
    elif args.model_type == 'cvae':
        z, mu, logvar, c = trainer.get_latent_representations(full_loader)
        latent_df = pd.DataFrame(
            mu,
            index=dataset.sample_ids,
            columns=[f'latent_{i}' for i in range(args.latent_dim)]
        )
        # Compute and save perturbation effects
        print("\nComputing perturbation effects...")
        effects = trainer.compute_perturbation_effects(full_loader, dataset.perturbation_names)
        for name, effect in effects.items():
            effect_df = pd.DataFrame(
                effect.reshape(1, -1),
                columns=[f'feature_{i}' for i in range(len(effect))]
            )
            effect_df.to_csv(output_dir / f'perturbation_effect_{name}.csv', index=False)
            print(f"  Saved effect for {name}: mean abs effect = {np.abs(effect).mean():.4f}")
    else:
        # For standard AE
        model.eval()
        all_z = []
        with torch.no_grad():
            for batch in full_loader:
                omics_keys = [k for k in batch.keys() if k != 'perturbations']
                tensors = [batch[k] for k in sorted(omics_keys)]
                x = torch.cat(tensors, dim=1).to(trainer.device)
                z = model.encode(x)
                all_z.append(z.cpu().numpy())
        z = np.concatenate(all_z, axis=0)
        latent_df = pd.DataFrame(
            z,
            index=dataset.sample_ids,
            columns=[f'latent_{i}' for i in range(args.latent_dim)]
        )

    latent_df.to_csv(output_dir / 'latent_representations.csv')
    print(f"Saved latent representations: {latent_df.shape}")

    # Save perturbations alongside for analysis
    if dataset.perturbations is not None:
        pert_df = pd.DataFrame(
            dataset.perturbations.numpy(),
            index=dataset.sample_ids,
            columns=dataset.perturbation_names
        )
        pert_df.to_csv(output_dir / 'perturbations.csv')

    print(f"\nTraining complete! Results saved to: {output_dir}")
    print(f"  - Checkpoints: {output_dir / 'checkpoints'}")
    print(f"  - Latent representations: {output_dir / 'latent_representations.csv'}")
    if args.model_type == 'cvae':
        print(f"  - Perturbation effects: {output_dir / 'perturbation_effect_*.csv'}")


if __name__ == '__main__':
    main()
