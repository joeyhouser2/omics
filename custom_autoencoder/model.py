"""Multi-omics Variational Autoencoder (VAE) model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class Encoder(nn.Module):
    """Encoder network that maps multi-omics input to latent distribution."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()

        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Latent space parameters (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network that reconstructs multi-omics data from latent space."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()

        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        h = self.decoder(z)
        return self.fc_out(h)


class MultiOmicsVAE(nn.Module):
    """Variational Autoencoder for multi-omics data integration.

    This VAE learns a joint latent representation from multiple omics
    data types (RNA-seq, methylation, CNV) and can reconstruct the
    original data from the latent space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 64,
        dropout: float = 0.2,
        feature_dims: Dict[str, int] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.feature_dims = feature_dims or {}

        # Default hidden dimensions based on input size
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, dropout)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input tensor of concatenated omics features

        Returns:
            recon_x: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Sampled latent representation
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (returns mean)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)

    def get_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get latent representation with uncertainty."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VAELoss(nn.Module):
    """Combined reconstruction and KL divergence loss for VAE."""

    def __init__(
        self,
        beta: float = 1.0,
        reconstruction_loss: str = 'mse'
    ):
        super().__init__()
        self.beta = beta  # Weight for KL divergence (beta-VAE)
        self.reconstruction_loss = reconstruction_loss

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss.

        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss
        if self.reconstruction_loss == 'mse':
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        else:
            recon_loss = F.l1_loss(recon_x, x, reduction='mean')

        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


class MultiOmicsAE(nn.Module):
    """Standard Autoencoder (non-variational) for comparison."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class ConditionalVAE(nn.Module):
    """Conditional VAE that incorporates perturbations into encoding/decoding.

    This model learns how perturbations (FLT3, NPM1, Gender) affect the
    latent representation, allowing for:
    - Perturbation-aware encoding
    - Counterfactual generation (what would sample X look like with/without mutation?)
    - Direct measurement of perturbation effects in latent space
    """

    def __init__(
        self,
        input_dim: int,
        n_perturbations: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 64,
        dropout: float = 0.2,
        feature_dims: Dict[str, int] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_perturbations = n_perturbations
        self.latent_dim = latent_dim
        self.feature_dims = feature_dims or {}

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        # Encoder takes omics + perturbations
        encoder_input_dim = input_dim + n_perturbations
        self.encoder = Encoder(encoder_input_dim, hidden_dims, latent_dim, dropout)

        # Decoder takes latent + perturbations
        decoder_input_dim = latent_dim + n_perturbations

        # Custom decoder that accepts condition
        layers = []
        prev_dim = decoder_input_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.decoder_layers = nn.Sequential(*layers)
        self.decoder_out = nn.Linear(hidden_dims[0], input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with condition."""
        xc = torch.cat([x, c], dim=1)
        return self.encoder(xc)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode with condition."""
        zc = torch.cat([z, c], dim=1)
        h = self.decoder_layers(zc)
        return self.decoder_out(h)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Omics data [batch, input_dim]
            c: Perturbations [batch, n_perturbations]

        Returns:
            recon_x, mu, logvar, z
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar, z

    def get_latent(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get latent representation."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_counterfactual(
        self, x: torch.Tensor, c_original: torch.Tensor, c_counterfactual: torch.Tensor
    ) -> torch.Tensor:
        """Generate counterfactual: encode with original, decode with new perturbation.

        This answers: "What would this sample look like if it had different perturbations?"
        """
        mu, _ = self.encode(x, c_original)
        return self.decode(mu, c_counterfactual)

    def compute_perturbation_effect(
        self, x: torch.Tensor, c: torch.Tensor, perturbation_idx: int
    ) -> torch.Tensor:
        """Compute effect of flipping a single perturbation.

        Returns the difference in reconstruction when perturbation is flipped.
        """
        # Original reconstruction
        mu, _ = self.encode(x, c)
        recon_original = self.decode(mu, c)

        # Flip the perturbation
        c_flipped = c.clone()
        c_flipped[:, perturbation_idx] = 1 - c_flipped[:, perturbation_idx]
        recon_flipped = self.decode(mu, c_flipped)

        return recon_flipped - recon_original
