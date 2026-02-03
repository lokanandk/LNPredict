"""
Enhanced Data Loading with Augmentation Support
"""
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from pathlib import Path
from typing import Dict, Tuple, Optional

from src.utils.molecular_features import MultiComponentFeaturizer

class LNPDataLoader:
    """Enhanced data loader with augmentation capabilities."""
    
    def __init__(self):
        self.featurizer = MultiComponentFeaturizer()
        self.scaler = RobustScaler()
    
    def augment_data(self, molecular_features, composition_features, property_features, targets, factor=2):
        """Augment data with controlled noise injection."""
        
        augmented_mol = [molecular_features]
        augmented_comp = [composition_features]
        augmented_prop = [property_features]
        augmented_targets = [targets]
        
        for i in range(factor):
            # Controlled noise levels
            noise_mol = 0.03 if i == 0 else 0.05
            noise_comp = 0.015 if i == 0 else 0.025
            noise_prop = 0.02 if i == 0 else 0.04
            
            # Generate augmented data
            aug_mol = molecular_features + np.random.normal(0, noise_mol, molecular_features.shape)
            aug_comp = composition_features + np.random.normal(0, noise_comp, composition_features.shape)
            aug_prop = property_features + np.random.normal(0, noise_prop, property_features.shape)
            
            # Ensure compositions are valid (positive and sum to 1)
            aug_comp = np.abs(aug_comp)
            aug_comp = aug_comp / aug_comp.sum(axis=1, keepdims=True)
            
            # Clip features to reasonable bounds
            aug_mol = np.clip(aug_mol, -3, 3)
            aug_prop = np.clip(aug_prop, -3, 3)
            
            augmented_mol.append(aug_mol)
            augmented_comp.append(aug_comp)
            augmented_prop.append(aug_prop)
            augmented_targets.append(targets)
        
        return (np.vstack(augmented_mol), 
                np.vstack(augmented_comp),
                np.vstack(augmented_prop),
                np.hstack(augmented_targets))
    
    def load_and_prepare(self, data_path: str, synthetic_data: Optional[str] = None,
                        config: Dict = None, device: str = 'cuda', augment: bool = False) -> Dict:
        """Load and prepare data with optional augmentation."""
        
        if config is None:
            config = {}
        
        print(f"🔍 DEBUG: Data loader from {__file__}")
        print(f"Loading data from {data_path}")
        
        # Load main data
        features, targets, feature_names = self.featurizer.create_comprehensive_features(data_path)
        
        # Add synthetic data if provided
        if synthetic_data:
            print(f"Adding synthetic data from {synthetic_data}")
            synth_features, synth_targets, _ = self.featurizer.create_comprehensive_features(synthetic_data)
            features = np.vstack([features, synth_features])
            targets = np.hstack([targets, synth_targets])
        
        # Separate feature types
        mol_mask = [name.startswith('mol_') for name in feature_names]
        comp_mask = [name.startswith('comp_') for name in feature_names]
        prop_mask = [name.startswith('prop_') for name in feature_names]
        
        molecular_features = features[:, mol_mask]
        composition_features = features[:, comp_mask]
        property_features = features[:, prop_mask]
        
        # Scale property features
        property_features = self.scaler.fit_transform(property_features)
        
        # Prepare molecular features
        n_components = composition_features.shape[1]
        mol_feat_per_comp = molecular_features.shape[1] // n_components
        
        # Padding for attention compatibility
        if mol_feat_per_comp % 4 != 0:
            padding_needed = 4 - (mol_feat_per_comp % 4)
            mol_feat_per_comp += padding_needed
        
        total_mol_feat_needed = n_components * mol_feat_per_comp
        if molecular_features.shape[1] < total_mol_feat_needed:
            padding = total_mol_feat_needed - molecular_features.shape[1]
            molecular_features = np.pad(molecular_features, ((0, 0), (0, padding)), 'constant')
        
        molecular_features = molecular_features.reshape(-1, n_components, mol_feat_per_comp)
        
        # Split data
        indices = np.arange(len(targets))
        target_bins = self._create_stratified_bins(targets, config.get('stratify_bins', 5))
        
        if target_bins is not None:
            train_idx, test_idx = train_test_split(
                indices, test_size=config.get('test_size', 0.15), 
                random_state=config.get('random_state', 42), 
                stratify=target_bins
            )
        else:
            train_idx, test_idx = train_test_split(
                indices, test_size=config.get('test_size', 0.15),
                random_state=config.get('random_state', 42)
            )
        
        # Create data tensors
        train_data = {
            'molecular': torch.FloatTensor(molecular_features[train_idx]).to(device),
            'composition': torch.FloatTensor(composition_features[train_idx]).to(device),
            'property': torch.FloatTensor(property_features[train_idx]).to(device),
            'targets': torch.FloatTensor(targets[train_idx]).to(device)
        }
        
        val_data = {
            'molecular': torch.FloatTensor(molecular_features[test_idx]).to(device),
            'composition': torch.FloatTensor(composition_features[test_idx]).to(device),
            'property': torch.FloatTensor(property_features[test_idx]).to(device),
            'targets': torch.FloatTensor(targets[test_idx]).to(device)
        }
        
        data_config = {
            'molecular_feature_dim': mol_feat_per_comp,
            'n_components': n_components,
            'property_dim': property_features.shape[1],
            'train_data': train_data,
            'val_data': val_data,
            'scaler': self.scaler,
            'feature_names': feature_names
        }
        
        print(f"✅ Data prepared: {len(train_idx)} train, {len(test_idx)} val samples")
        return data_config
    
    def _create_stratified_bins(self, targets, n_bins=5):
        """Create bins for stratified splitting."""
        try:
            bin_edges = np.quantile(targets, np.linspace(0, 1, n_bins + 1))
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                return None
            return np.digitize(targets, bin_edges[1:-1])
        except Exception:
            return None
