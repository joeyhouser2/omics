"""Training utilities for multi-omics VAE."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Tuple
import json

from .model import MultiOmicsVAE, VAELoss, MultiOmicsAE, ConditionalVAE


class VAETrainer:
    """Trainer class for multi-omics VAE."""

    def __init__(
        self,
        model: MultiOmicsVAE,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        device: str = None,
        checkpoint_dir: Path = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = VAELoss(beta=beta)
        self.checkpoint_dir = checkpoint_dir or Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'val_recon': [],
            'train_kl': [],
            'val_kl': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch in train_loader:
            # Concatenate all omics data
            x = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            recon_x, mu, logvar, z = self.model(x)
            loss, recon_loss, kl_loss = self.criterion(recon_x, x, mu, logvar)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch in val_loader:
            x = self._prepare_batch(batch)
            recon_x, mu, logvar, z = self.model(x)
            loss, recon_loss, kl_loss = self.criterion(recon_x, x, mu, logvar)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare batch by concatenating omics data."""
        # Get all omics tensors (exclude perturbations for now)
        omics_keys = [k for k in batch.keys() if k != 'perturbations']
        tensors = [batch[k] for k in sorted(omics_keys)]
        x = torch.cat(tensors, dim=1).to(self.device)
        return x

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        early_stopping_patience: int = 20,
        save_best: bool = True
    ) -> Dict[str, list]:
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_recon'].append(train_recon)
            self.history['train_kl'].append(train_kl)

            # Validate
            val_loss, val_recon, val_kl = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_recon'].append(val_recon)
            self.history['val_kl'].append(val_kl)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d} | "
                      f"Train Loss: {train_loss:.4f} (recon: {train_recon:.4f}, kl: {train_kl:.4f}) | "
                      f"Val Loss: {val_loss:.4f} (recon: {val_recon:.4f}, kl: {val_kl:.4f})")

            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Save final model
        self.save_checkpoint('final_model.pt')
        self.save_history()

        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
                'feature_dims': self.model.feature_dims
            }
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

    def save_history(self):
        """Save training history to JSON."""
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    @torch.no_grad()
    def get_latent_representations(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract latent representations for all samples."""
        self.model.eval()

        all_z = []
        all_mu = []
        all_logvar = []

        for batch in data_loader:
            x = self._prepare_batch(batch)
            z, mu, logvar = self.model.get_latent(x)
            all_z.append(z.cpu().numpy())
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())

        return (
            np.concatenate(all_z, axis=0),
            np.concatenate(all_mu, axis=0),
            np.concatenate(all_logvar, axis=0)
        )


class AETrainer:
    """Trainer for standard (non-variational) autoencoder."""

    def __init__(
        self,
        model: MultiOmicsAE,
        learning_rate: float = 1e-3,
        device: str = None,
        checkpoint_dir: Path = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()
        self.checkpoint_dir = checkpoint_dir or Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = self._prepare_batch(batch)
            self.optimizer.zero_grad()
            recon_x, z = self.model(x)
            loss = self.criterion(recon_x, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            x = self._prepare_batch(batch)
            recon_x, z = self.model(x)
            loss = self.criterion(recon_x, x)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        omics_keys = [k for k in batch.keys() if k != 'perturbations']
        tensors = [batch[k] for k in sorted(omics_keys)]
        return torch.cat(tensors, dim=1).to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        early_stopping_patience: int = 20
    ) -> Dict[str, list]:
        print(f"Training AE on {self.device}")
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_ae.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return self.history


class CVAETrainer:
    """Trainer for Conditional VAE with perturbations."""

    def __init__(
        self,
        model: ConditionalVAE,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        device: str = None,
        checkpoint_dir: Path = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = VAELoss(beta=beta)
        self.checkpoint_dir = checkpoint_dir or Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_recon': [], 'val_recon': [],
            'train_kl': [], 'val_kl': []
        }

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch: separate omics data and perturbations."""
        omics_keys = [k for k in batch.keys() if k != 'perturbations']
        tensors = [batch[k] for k in sorted(omics_keys)]
        x = torch.cat(tensors, dim=1).to(self.device)
        c = batch['perturbations'].to(self.device)
        return x, c

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        self.model.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            x, c = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            recon_x, mu, logvar, z = self.model(x, c)
            loss, recon_loss, kl_loss = self.criterion(recon_x, x, mu, logvar)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in val_loader:
            x, c = self._prepare_batch(batch)
            recon_x, mu, logvar, z = self.model(x, c)
            loss, recon_loss, kl_loss = self.criterion(recon_x, x, mu, logvar)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        early_stopping_patience: int = 20,
        save_best: bool = True
    ) -> Dict[str, list]:
        print(f"Training Conditional VAE on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_recon'].append(train_recon)
            self.history['train_kl'].append(train_kl)

            val_loss, val_recon, val_kl = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_recon'].append(val_recon)
            self.history['val_kl'].append(val_kl)

            self.scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d} | "
                      f"Train: {train_loss:.4f} (recon: {train_recon:.4f}, kl: {train_kl:.4f}) | "
                      f"Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.save_checkpoint('best_cvae.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.save_checkpoint('final_cvae.pt')
        self.save_history()
        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        return self.history

    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
                'n_perturbations': self.model.n_perturbations,
                'feature_dims': self.model.feature_dims
            }
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def save_history(self):
        with open(self.checkpoint_dir / 'cvae_training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    @torch.no_grad()
    def get_latent_representations(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract latent representations with perturbations."""
        self.model.eval()
        all_z, all_mu, all_logvar, all_c = [], [], [], []

        for batch in data_loader:
            x, c = self._prepare_batch(batch)
            z, mu, logvar = self.model.get_latent(x, c)
            all_z.append(z.cpu().numpy())
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_c.append(c.cpu().numpy())

        return (
            np.concatenate(all_z, axis=0),
            np.concatenate(all_mu, axis=0),
            np.concatenate(all_logvar, axis=0),
            np.concatenate(all_c, axis=0)
        )

    @torch.no_grad()
    def compute_perturbation_effects(
        self, data_loader: DataLoader, perturbation_names: list
    ) -> Dict[str, np.ndarray]:
        """Compute average effect of each perturbation on reconstruction."""
        self.model.eval()
        n_perts = self.model.n_perturbations
        effects = {name: [] for name in perturbation_names}

        for batch in data_loader:
            x, c = self._prepare_batch(batch)
            for i, name in enumerate(perturbation_names):
                effect = self.model.compute_perturbation_effect(x, c, i)
                effects[name].append(effect.cpu().numpy())

        # Average effects across all samples
        return {
            name: np.concatenate(vals, axis=0).mean(axis=0)
            for name, vals in effects.items()
        }
