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
    
    def _create_default_properties(self, n_samples: int) -> np.ndarray:
        """Create default property features when real data is missing."""
        print("⚠️ Creating default property features")
        
        # Default values for typical LNP properties
        # [particle_size, zeta_potential, encapsulation_efficiency, polydispersity_index]
        defaults = np.array([
            [100.0, -10.0, 0.8, 0.2]  # Reasonable defaults for LNPs
        ])
        
        # Repeat defaults for all samples
        default_properties = np.tile(defaults, (n_samples, 1))
        
        # Add small random noise to avoid identical values
        np.random.seed(42)  # For reproducibility
        noise_scale = 0.05
        noise = np.random.normal(0, noise_scale, default_properties.shape)
        
        # Apply noise with reasonable bounds
        default_properties[:, 0] += noise[:, 0] * 10  # particle_size ± 0.5
        default_properties[:, 1] += noise[:, 1] * 2   # zeta_potential ± 0.1
        default_properties[:, 2] += noise[:, 2] * 0.1 # encap_eff ± 0.005
        default_properties[:, 3] += noise[:, 3] * 0.02 # PDI ± 0.001
        
        # Ensure reasonable bounds
        default_properties[:, 0] = np.clip(default_properties[:, 0], 50, 200)   # particle size
        default_properties[:, 1] = np.clip(default_properties[:, 1], -50, 20)  # zeta potential
        default_properties[:, 2] = np.clip(default_properties[:, 2], 0.5, 1.0) # encap efficiency
        default_properties[:, 3] = np.clip(default_properties[:, 3], 0.1, 0.5) # PDI
        
        return default_properties
    
    def _safe_load_properties(self, features: np.ndarray, prop_mask: list, 
                             feature_names: list, n_samples: int) -> np.ndarray:
        """Safely load property features with fallbacks."""
        
        # Check if we have any property features
        prop_features = features[:, prop_mask] if any(prop_mask) else np.array([]).reshape(n_samples, 0)
        
        print(f"🔍 Property features shape: {prop_features.shape}")
        
        # If no property features or empty features, create defaults
        if prop_features.shape[1] == 0:
            print("⚠️ No property features found, using defaults")
            return self._create_default_properties(n_samples)
        
        # Check for all-zero or all-NaN features
        if np.all(prop_features == 0) or np.all(np.isnan(prop_features)):
            print("⚠️ Property features are all zeros/NaN, using defaults")
            return self._create_default_properties(n_samples)
        
        # If we have some property features but less than expected (4), pad with defaults
        if prop_features.shape[1] < 4:
            print(f"⚠️ Only {prop_features.shape[1]} property features found, padding to 4")
            defaults = self._create_default_properties(n_samples)
            
            # Use existing features for first columns, defaults for the rest
            padded_features = defaults.copy()
            padded_features[:, :prop_features.shape[1]] = prop_features
            
            return padded_features
        
        # If we have 4+ features, use first 4
        return prop_features[:, :4]
    
    def load_and_prepare(self, data_path: str, synthetic_data: Optional[str] = None,
                        config: Dict = None, device: str = 'cuda', augment: bool = False) -> Dict:
        """Load and prepare data with optional augmentation and robust property handling."""
        
        if config is None:
            config = {}
        
        print(f"🔍 DEBUG: Data loader from {__file__}")
        print(f"Loading data from {data_path}")
        
        # Load main data
        try:
            features, targets, feature_names = self.featurizer.create_comprehensive_features(data_path)
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            raise
        
        print(f"✅ Loaded features shape: {features.shape}, targets: {len(targets)}")
        
        # Add synthetic data if provided
        if synthetic_data:
            print(f"Adding synthetic data from {synthetic_data}")
            try:
                synth_features, synth_targets, _ = self.featurizer.create_comprehensive_features(synthetic_data)
                features = np.vstack([features, synth_features])
                targets = np.hstack([targets, synth_targets])
                print(f"✅ Combined features shape: {features.shape}, targets: {len(targets)}")
            except Exception as e:
                print(f"⚠️ Failed to load synthetic data: {e}, continuing without it")
        
        # Separate feature types
        mol_mask = [name.startswith('mol_') for name in feature_names]
        comp_mask = [name.startswith('comp_') for name in feature_names]
        prop_mask = [name.startswith('prop_') for name in feature_names]
        
        print(f"🔍 Feature masks - Molecular: {sum(mol_mask)}, Composition: {sum(comp_mask)}, Property: {sum(prop_mask)}")
        
        # Extract features with safety checks
        molecular_features = features[:, mol_mask] if any(mol_mask) else np.array([]).reshape(features.shape[0], 0)
        composition_features = features[:, comp_mask] if any(comp_mask) else np.array([]).reshape(features.shape[0], 0)
        
        # Safe property loading with fallbacks
        try:
            property_features = self._safe_load_properties(features, prop_mask, feature_names, features.shape[0])
            print(f"✅ Property features prepared: {property_features.shape}")
            
            # Scale property features
            property_features = self.scaler.fit_transform(property_features)
            
        except Exception as e:
            print(f"⚠️ Property feature processing failed: {e}")
            print("Creating default property features...")
            property_features = self._create_default_properties(features.shape[0])
            property_features = self.scaler.fit_transform(property_features)
        
        # Validate we have essential features
        if molecular_features.shape[1] == 0:
            raise ValueError("No molecular features found - this is required for LNPredict")
        
        if composition_features.shape[1] == 0:
            raise ValueError("No composition features found - this is required for LNPredict")
        
        # Prepare molecular features
        n_components = composition_features.shape[1]
        mol_feat_per_comp = molecular_features.shape[1] // n_components if n_components > 0 else 0
        
        print(f"🔍 Components: {n_components}, Mol features per component: {mol_feat_per_comp}")
        
        # Ensure molecular features are compatible with attention mechanism
        if mol_feat_per_comp == 0:
            print("⚠️ No molecular features per component, using minimal features")
            mol_feat_per_comp = 16  # Minimum for attention
            molecular_features = np.random.normal(0, 0.1, (features.shape[0], n_components * mol_feat_per_comp))
        
        # Padding for attention compatibility
        if mol_feat_per_comp % 4 != 0:
            padding_needed = 4 - (mol_feat_per_comp % 4)
            mol_feat_per_comp += padding_needed
        
        total_mol_feat_needed = n_components * mol_feat_per_comp
        if molecular_features.shape[1] < total_mol_feat_needed:
            padding = total_mol_feat_needed - molecular_features.shape[1]
            molecular_features = np.pad(molecular_features, ((0, 0), (0, padding)), 'constant')
        elif molecular_features.shape[1] > total_mol_feat_needed:
            # Truncate if too many features
            molecular_features = molecular_features[:, :total_mol_feat_needed]
        
        # Reshape molecular features for transformer
        try:
            molecular_features = molecular_features.reshape(-1, n_components, mol_feat_per_comp)
            print(f"✅ Molecular features reshaped: {molecular_features.shape}")
        except Exception as e:
            print(f"❌ Error reshaping molecular features: {e}")
            raise
        
        # Split data with stratification
        indices = np.arange(len(targets))
        target_bins = self._create_stratified_bins(targets, config.get('stratify_bins', 5))
        
        try:
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
        except Exception as e:
            print(f"⚠️ Stratified split failed: {e}, using random split")
            train_idx, test_idx = train_test_split(
                indices, test_size=config.get('test_size', 0.15),
                random_state=config.get('random_state', 42)
            )
        
        # Create data tensors with device handling
        try:
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
        except Exception as e:
            print(f"❌ Error creating tensors: {e}")
            raise
        
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
        print(f"📊 Final dimensions - Molecular: {mol_feat_per_comp}, Components: {n_components}, Properties: {property_features.shape[1]}")
        
        return data_config
    
    def load_for_evaluation(self, data_path: str, use_properties: bool = True) -> Dict:
        """Load data specifically for evaluation with flexible property handling."""
        
        print(f"🔍 Loading evaluation data from {data_path}")
        
        try:
            features, targets, feature_names = self.featurizer.create_comprehensive_features(data_path)
        except Exception as e:
            print(f"❌ Error loading evaluation features: {e}")
            raise
        
        # Separate feature types
        mol_mask = [name.startswith('mol_') for name in feature_names]
        comp_mask = [name.startswith('comp_') for name in feature_names]
        prop_mask = [name.startswith('prop_') for name in feature_names]
        
        molecular_features = features[:, mol_mask] if any(mol_mask) else np.array([]).reshape(features.shape[0], 0)
        composition_features = features[:, comp_mask] if any(comp_mask) else np.array([]).reshape(features.shape[0], 0)
        
        if use_properties:
            try:
                property_features = self._safe_load_properties(features, prop_mask, feature_names, features.shape[0])
                property_features = self.scaler.fit_transform(property_features)
            except Exception as e:
                print(f"⚠️ Property loading failed: {e}")
                print("Falling back to default properties")
                property_features = self._create_default_properties(features.shape[0])
                property_features = self.scaler.fit_transform(property_features)
        else:
            print("⚠️ Skipping property features for evaluation")
            property_features = self._create_default_properties(features.shape[0])
            property_features = self.scaler.fit_transform(property_features)
        
        # Process molecular features similar to training
        n_components = composition_features.shape[1]
        mol_feat_per_comp = molecular_features.shape[1] // n_components if n_components > 0 else 16
        
        if molecular_features.shape[1] == 0:
            molecular_features = np.random.normal(0, 0.1, (features.shape[0], n_components * mol_feat_per_comp))
        
        # Ensure compatibility
        if mol_feat_per_comp % 4 != 0:
            padding_needed = 4 - (mol_feat_per_comp % 4)
            mol_feat_per_comp += padding_needed
        
        total_mol_feat_needed = n_components * mol_feat_per_comp
        if molecular_features.shape[1] < total_mol_feat_needed:
            padding = total_mol_feat_needed - molecular_features.shape[1]
            molecular_features = np.pad(molecular_features, ((0, 0), (0, padding)), 'constant')
        elif molecular_features.shape[1] > total_mol_feat_needed:
            molecular_features = molecular_features[:, :total_mol_feat_needed]
        
        molecular_features = molecular_features.reshape(-1, n_components, mol_feat_per_comp)
        
        # Create evaluation dataset (no device transfer for flexibility)
        eval_data = {
            'molecular': torch.FloatTensor(molecular_features),
            'composition': torch.FloatTensor(composition_features),  
            'property': torch.FloatTensor(property_features),
            'targets': torch.FloatTensor(targets) if targets is not None and len(targets) > 0 else None,
            'molecular_feature_dim': mol_feat_per_comp,
            'n_components': n_components,
            'property_dim': property_features.shape[1]
        }
        
        print(f"✅ Evaluation data loaded: {features.shape[0]} samples")
        return eval_data
    
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
