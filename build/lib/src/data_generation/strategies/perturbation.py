"""
Perturbation-based Generation Strategy

Generates synthetic samples using controlled perturbations of existing
samples while maintaining domain constraints and relationships.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from typing import List, Dict


class PerturbationGeneration:
    """Perturbation-based generation strategy."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_samples(self, comps: np.ndarray, props: np.ndarray,
                        targets: np.ndarray, n_samples: int,
                        comp_cols: List[str], prop_cols: List[str]) -> List[Dict]:
        """Generate samples using controlled perturbations."""
        samples = []
        
        # Calculate perturbation scales
        comp_scales = np.std(comps, axis=0) * 0.1
        prop_scales = np.std(props, axis=0) * 0.15
        target_scale = np.std(targets) * 0.1
        
        for _ in range(n_samples):
            # Pick base sample
            base_idx = np.random.randint(len(comps))
            
            # Apply structured perturbations
            comp_multipliers = 1 + np.random.normal(0, 0.1, len(comps[base_idx]))
            new_comp = comps[base_idx] * comp_multipliers
            new_comp = np.maximum(new_comp, 0)
            
            # Renormalize compositions
            total_comp = np.sum(comps[base_idx])
            if np.sum(new_comp) > 0:
                new_comp = new_comp / np.sum(new_comp) * total_comp
            
            # Properties: additive perturbations
            new_props = props[base_idx] + np.random.normal(0, prop_scales)
            new_props = self._apply_domain_constraints(new_props, prop_cols)
            
            # Predict target
            combined_features = np.concatenate([new_comp, new_props])
            original_combined = np.hstack([comps, props])
            new_target = self._predict_target(combined_features, original_combined, targets)
            
            # Add target noise
            new_target += np.random.normal(0, target_scale)
            new_target = max(0, new_target)
            
            # Create sample
            sample = {}
            for i, col in enumerate(comp_cols):
                sample[col] = new_comp[i]
            for i, col in enumerate(prop_cols):
                sample[col] = new_props[i]
            sample['target'] = new_target
            
            samples.append(sample)
        
        return samples
    
    def _predict_target(self, features: np.ndarray, original_features: np.ndarray,
                       original_targets: np.ndarray) -> float:
        """Predict target from combined features."""
        model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        model.fit(original_features, original_targets)
        return model.predict(features.reshape(1, -1))[0]
    
    def _apply_domain_constraints(self, props: np.ndarray, prop_cols: List[str]) -> np.ndarray:
        """Apply domain-specific constraints to properties."""
        constrained_props = props.copy()
        
        for i, col in enumerate(prop_cols):
            if 'size' in col.lower():
                constrained_props[i] = np.clip(constrained_props[i], 10, 500)
            elif 'polydispersity' in col.lower() or 'pdi' in col.lower():
                constrained_props[i] = np.clip(constrained_props[i], 0.05, 0.8)
            elif 'encapsulation' in col.lower():
                constrained_props[i] = np.clip(constrained_props[i], 0, 100)
            elif 'zeta' in col.lower() or 'potential' in col.lower():
                constrained_props[i] = np.clip(constrained_props[i], -100, 100)
        
        return constrained_props
