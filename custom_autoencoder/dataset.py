"""Dataset class for multi-omics data loading."""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MultiOmicsDataset(Dataset):
    """PyTorch Dataset for multi-omics data.

    Loads preprocessed RNA-seq, methylation, and CNV data along with
    perturbation labels for training a multi-omics VAE.
    """

    def __init__(
        self,
        data_dir: Path,
        omics_types: List[str] = None,
        load_perturbations: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.omics_types = omics_types or ['rna_seq', 'methylation', 'cnv']
        self.load_perturbations = load_perturbations

        # Load all omics data
        self.omics_data: Dict[str, torch.Tensor] = {}
        self.feature_dims: Dict[str, int] = {}
        self.sample_ids: List[str] = None

        self._load_data()

    def _load_data(self):
        """Load all omics datasets and align samples."""
        print("Loading multi-omics data...")

        # Load sample IDs
        sample_ids_file = self.data_dir / 'sample_ids.txt'
        if sample_ids_file.exists():
            with open(sample_ids_file, 'r') as f:
                self.sample_ids = [line.strip() for line in f.readlines()]

        # Load each omics type
        for omics_type in self.omics_types:
            file_path = self.data_dir / f'{omics_type}.tsv'
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\t', index_col=0)
                self.omics_data[omics_type] = torch.tensor(
                    df.values, dtype=torch.float32
                )
                self.feature_dims[omics_type] = df.shape[1]
                print(f"  Loaded {omics_type}: {df.shape[0]} samples x {df.shape[1]} features")

                if self.sample_ids is None:
                    self.sample_ids = df.index.tolist()
            else:
                print(f"  Warning: {file_path} not found, skipping {omics_type}")

        # Load perturbations if requested
        self.perturbations = None
        self.perturbation_names = []
        if self.load_perturbations:
            pert_data = []
            for pert_file in ['flt3_status.tsv', 'npm1_status.tsv', 'gender.tsv']:
                file_path = self.data_dir / pert_file
                if file_path.exists():
                    df = pd.read_csv(file_path, sep='\t')
                    col_name = df.columns[1]
                    self.perturbation_names.append(col_name)
                    # Convert to binary
                    binary = (df[col_name].isin(['Mutated', 'Male'])).astype(int).values
                    pert_data.append(binary)

            if pert_data:
                self.perturbations = torch.tensor(
                    np.column_stack(pert_data), dtype=torch.float32
                )
                print(f"  Loaded perturbations: {self.perturbations.shape[1]} variables")

        print(f"  Total samples: {len(self)}")

    def __len__(self) -> int:
        first_key = list(self.omics_data.keys())[0]
        return self.omics_data[first_key].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with all omics data."""
        sample = {}

        # Get each omics type
        for omics_type, data in self.omics_data.items():
            sample[omics_type] = data[idx]

        # Get perturbations if available
        if self.perturbations is not None:
            sample['perturbations'] = self.perturbations[idx]

        return sample

    def get_concatenated(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get all omics data concatenated into a single tensor."""
        tensors = [self.omics_data[k] for k in self.omics_types if k in self.omics_data]
        X = torch.cat(tensors, dim=1)
        return X, self.perturbations

    def get_input_dim(self) -> int:
        """Get total input dimension (sum of all omics features)."""
        return sum(self.feature_dims.values())

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each omics type."""
        return self.feature_dims.copy()


def create_data_loaders(
    data_dir: Path,
    batch_size: int = 32,
    train_split: float = 0.8,
    random_seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders."""
    from torch.utils.data import random_split, DataLoader

    dataset = MultiOmicsDataset(data_dir)

    # Split into train/val
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    n_val = n_samples - n_train

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"Created data loaders: {n_train} train, {n_val} validation samples")

    return train_loader, val_loader, dataset
