import pandas as pd
import numpy as np
from pathlib import Path
import torch
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

#MOVE import
try:
    from move import MOVE, train_move_model
    MOVE_AVAILABLE = True
except ImportError:
    MOVE_AVAILABLE = False
    print("WARNING: move-dl package not found!")
    print("Install with: pip install move-dl")
    print("Continuing with basic setup...")
PROCESSED_DATA_DIR = Path('processed_data')
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CONFIG = {
    'latent_dim': 128,              
    'batch_size': 32,               
    'learning_rate': 1e-3,          
    'num_epochs': 100,             
    'beta': 1.0,                    
    'encoder_layers': [512, 256],   
    'decoder_layers': [256, 512],   
    'use_gpu': torch.cuda.is_available(),
    'save_every': 10,               #save checkpoint
}


class MOVEDataLoader:
    def __init__(self, data_dir: Path = PROCESSED_DATA_DIR):
        self.data_dir = data_dir

    def load_processed_data(self) -> Dict:
        print("Loading processed data...")
        rna = pd.read_csv(self.data_dir / 'rna_seq_processed.csv', index_col=0)
        meth = pd.read_csv(self.data_dir / 'methylation_processed.csv', index_col=0)
        cnv = pd.read_csv(self.data_dir / 'cnv_processed.csv', index_col=0)
        perturbations = pd.read_csv(self.data_dir / 'perturbations.csv', index_col=0)

        print(f"  RNA-seq: {rna.shape}")
        print(f"  Methylation: {meth.shape}")
        print(f"  CNV: {cnv.shape}")
        print(f"  Perturbations: {perturbations.shape}")
        assert all(rna.index == meth.index == cnv.index == perturbations.index), \
            "Sample IDs must match across all datasets!"
        data = {
            'rna': rna,
            'methylation': meth,
            'cnv': cnv,
            'perturbations': perturbations,
            'sample_ids': rna.index.tolist()}
        return data

    def create_move_format(self, data: Dict) -> Dict:
        """
        Convert pandas DataFrames to MOVE-compatible format.

        MOVE typically expects:
        - Continuous data as numpy arrays or tensors
        - Categorical/perturbation data properly encoded

        Args:
            data: Dictionary of DataFrames

        Returns:
            Dictionary in MOVE format
        """
        print("\nConverting to MOVE format...")

        # For MOVE, we typically need to create a config that describes each data type
        # Omics data: continuous
        # Perturbations: categorical (binary)

        move_data = {
            'continuous': {
                'RNA': data['rna'].values.astype(np.float32),
                'Methylation': data['methylation'].values.astype(np.float32),
                'CNV': data['cnv'].values.astype(np.float32),
            },
            'categorical': {
                'FLT3_Mutated': data['perturbations']['FLT3_Mutated'].values.astype(np.int32),
                'NPM1_Mutated': data['perturbations']['NPM1_Mutated'].values.astype(np.int32),
                'Gender_Male': data['perturbations']['Gender_Male'].values.astype(np.int32),
            },
            'sample_ids': data['sample_ids']
        }

        return move_data


def create_data_config(move_data: Dict) -> Dict:
    """
    Create MOVE data configuration.

    This tells MOVE about each data modality:
    - Name
    - Type (continuous/categorical)
    - Dimensions
    """
    config = {
        'data_types': []
    }

    # Add continuous data
    for name, values in move_data['continuous'].items():
        config['data_types'].append({
            'name': name,
            'type': 'continuous',
            'dim': values.shape[1]
        })

    # Add categorical data (perturbations)
    for name, values in move_data['categorical'].items():
        config['data_types'].append({
            'name': name,
            'type': 'categorical',
            'categories': 2,  # Binary (0 or 1)
        })

    return config


def train_move(move_data: Dict, config: Dict):
    """
    Train MOVE model.

    NOTE: This is a template. The actual MOVE API may differ.
    Check https://move-dl.readthedocs.io/ for the exact API.
    """
    print("\n" + "="*60)
    print("Training MOVE Model")
    print("="*60)

    if not MOVE_AVAILABLE:
        print("\nERROR: move-dl package not installed!")
        print("Please run: pip install move-dl")
        print("\nThis script will create a placeholder model structure.")
        print("After installing move-dl, you can use the official API.")
        return None

    # Create data configuration
    data_config = create_data_config(move_data)

    print("\nData configuration:")
    for dt in data_config['data_types']:
        print(f"  {dt}")

    # Initialize MOVE model
    # NOTE: Adjust based on actual MOVE API
    print("\nInitializing MOVE model...")
    print(f"  Latent dimension: {CONFIG['latent_dim']}")
    print(f"  Device: {'GPU' if CONFIG['use_gpu'] else 'CPU'}")

    # Placeholder for actual MOVE training
    # The real code would look something like:
    # model = MOVE(
    #     data_config=data_config,
    #     latent_dim=CONFIG['latent_dim'],
    #     encoder_layers=CONFIG['encoder_layers'],
    #     decoder_layers=CONFIG['decoder_layers'],
    # )
    #
    # Training loop would use MOVE's training function:
    # trained_model, history = train_move_model(
    #     model=model,
    #     data=move_data,
    #     batch_size=CONFIG['batch_size'],
    #     learning_rate=CONFIG['learning_rate'],
    #     num_epochs=CONFIG['num_epochs'],
    #     beta=CONFIG['beta']
    # )

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Install MOVE: pip install move-dl")
    print("2. Check documentation: https://move-dl.readthedocs.io/")
    print("3. Update this script with actual MOVE API calls")
    print("4. Alternative: Use the Google Colab tutorial:")
    print("   https://colab.research.google.com/drive/1RFWNsuGymCmppPsElBvDuA9zRbGskKmi")
    print("="*60)

    return None


def save_embeddings(embeddings: np.ndarray, sample_ids: list, output_path: Path):
    """Save latent embeddings to CSV."""
    emb_df = pd.DataFrame(
        embeddings,
        index=sample_ids,
        columns=[f'latent_{i}' for i in range(embeddings.shape[1])]
    )
    emb_df.to_csv(output_path)
    print(f"Saved embeddings: {output_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("MOVE Training Pipeline")
    print("="*60)

    # Load data
    loader = MOVEDataLoader()
    data = loader.load_processed_data()

    # Convert to MOVE format
    move_data = loader.create_move_format(data)

    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)
    print(f"Number of samples: {len(move_data['sample_ids'])}")
    print("\nContinuous modalities:")
    for name, values in move_data['continuous'].items():
        print(f"  {name}: {values.shape}")
    print("\nCategorical variables (perturbations):")
    for name, values in move_data['categorical'].items():
        n_positive = values.sum()
        print(f"  {name}: {n_positive}/{len(values)} positive ({100*n_positive/len(values):.1f}%)")

    # Train MOVE
    model = train_move(move_data, CONFIG)

    if model is not None:
        # If training succeeded, save model and generate embeddings
        print("\nGenerating latent embeddings...")
        # embeddings = model.get_embeddings(move_data)
        # save_embeddings(embeddings, move_data['sample_ids'], RESULTS_DIR / 'latent_embeddings.csv')

        print("\nSaving trained model...")
        # torch.save(model.state_dict(), MODELS_DIR / 'move_model.pth')
        print(f"Model saved to: {MODELS_DIR / 'move_model.pth'}")

    print("\n" + "="*60)
    print("Training pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
