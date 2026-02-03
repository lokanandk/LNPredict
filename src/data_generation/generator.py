"""
Main Targeted Augmentation Generator

Orchestrates multiple augmentation strategies to generate high-quality
synthetic LNP formulations from small datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from .strategies.interpolation import CompositionInterpolation
from .strategies.nearest_neighbor import NearestNeighborAugmentation  
from .strategies.perturbation import PerturbationGeneration
from .quality.assessor import QualityAssessor
from .utils import load_lnp_data, save_synthetic_data


class TargetedAugmentationGenerator:
    """
    Targeted augmentation generator for small LNP datasets.
    
    Uses multiple intelligent strategies to create diverse, high-quality
    synthetic formulations that preserve the statistical properties and
    relationships of the original dataset.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize strategy components
        self.interpolation = CompositionInterpolation(random_state)
        self.nearest_neighbor = NearestNeighborAugmentation(random_state)
        self.perturbation = PerturbationGeneration(random_state)
        self.quality_assessor = QualityAssessor()
        
    def load_data(self, json_file: str) -> pd.DataFrame:
        """Load LNP data from JSON file."""
        return load_lnp_data(json_file)
    
    def generate_synthetic_dataset(self, original_df: pd.DataFrame, 
                                 n_samples: int,
                                 strategy_weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Generate synthetic dataset using multiple strategies.
        
        Args:
            original_df: Original LNP formulation data
            n_samples: Number of synthetic samples to generate
            strategy_weights: Weights for each strategy (interpolation, nn, perturbation)
                             Defaults to {'interpolation': 0.4, 'nn': 0.3, 'perturbation': 0.3}
        
        Returns:
            DataFrame with synthetic formulations
        """
        if strategy_weights is None:
            strategy_weights = {
                'interpolation': 0.4,
                'nn': 0.3, 
                'perturbation': 0.3
            }
        
        # Validate weights
        if abs(sum(strategy_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Strategy weights must sum to 1.0")
            
        # Extract feature columns
        comp_cols = [col for col in original_df.columns if col.startswith('comp_')]
        prop_cols = [col for col in original_df.columns if col.startswith('prop_')]
        
        original_comps = original_df[comp_cols].values
        original_props = original_df[prop_cols].values
        original_targets = original_df['target'].values
        
        print(f"Generating {n_samples} synthetic samples using targeted strategies...")
        
        all_synthetic = []
        
        # Strategy 1: Composition Interpolation
        n_interp = int(strategy_weights['interpolation'] * n_samples)
        if n_interp > 0:
            print(f"1. Composition interpolation: {n_interp} samples")
            interp_samples = self.interpolation.generate_samples(
                original_comps, original_props, original_targets, 
                n_interp, comp_cols, prop_cols
            )
            all_synthetic.extend(interp_samples)
        
        # Strategy 2: Nearest Neighbor Augmentation
        n_nn = int(strategy_weights['nn'] * n_samples)
        if n_nn > 0:
            print(f"2. Nearest neighbor augmentation: {n_nn} samples")
            nn_samples = self.nearest_neighbor.generate_samples(
                original_comps, original_props, original_targets,
                n_nn, comp_cols, prop_cols
            )
            all_synthetic.extend(nn_samples)
        
        # Strategy 3: Perturbation Generation
        n_pert = n_samples - n_interp - n_nn
        if n_pert > 0:
            print(f"3. Perturbation-based generation: {n_pert} samples")
            pert_samples = self.perturbation.generate_samples(
                original_comps, original_props, original_targets,
                n_pert, comp_cols, prop_cols
            )
            all_synthetic.extend(pert_samples)
        
        # Convert to DataFrame
        synthetic_df = pd.DataFrame(all_synthetic)
        synthetic_df['id'] = [f'synthetic_{i}' for i in range(len(synthetic_df))]
        
        return synthetic_df
    
    def assess_quality(self, original_df: pd.DataFrame, 
                      synthetic_df: pd.DataFrame) -> Dict:
        """Assess quality of synthetic data."""
        return self.quality_assessor.assess_quality(original_df, synthetic_df)
    
    def save_synthetic_data(self, synthetic_df: pd.DataFrame, 
                          original_json: str, output_file: str, 
                          quality_report: Dict):
        """Save synthetic data with quality report."""
        save_synthetic_data(synthetic_df, original_json, output_file, 
                          quality_report, self.random_state)
