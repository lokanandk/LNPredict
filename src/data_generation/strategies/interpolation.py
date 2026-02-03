"""
Composition Interpolation Strategy

Generates synthetic samples by interpolating between similar compositions
in the original dataset, preserving compositional relationships.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import GradientBoostingRegressor
from typing import List, Dict


class CompositionInterpolation:
    """Composition interpolation augmentation strategy."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_samples(self, comps: np.ndarray, props: np.ndarray, 
                        targets: np.ndarray, n_samples: int,
                        comp_cols: List[str], prop_cols: List[str]) -> List[Dict]:
        """Generate samples by interpolating between similar compositions."""
        samples = []
        
        # Find composition similarities
        comp_distances = squareform(pdist(comps, metric='cosine'))
        
        for _ in range(n_samples):
            # Pick a random base sample
            base_idx = np.random.randint(len(comps))
            
            # Find similar compositions (not identical)
            distances = comp_distances[base_idx]
            candidates = np.where((distances > 0.01) & (distances < 0.5))[0]
            
            if len(candidates) > 0:
                neighbor_idx = np.random.choice(candidates)
                
                # Interpolate between base and neighbor
                alpha = np.random.beta(2, 2)  # Concentrated around 0.5
                
                # Interpolate compositions
                new_comp = alpha * comps[base_idx] + (1 - alpha) * comps[neighbor_idx]
                
                # Predict properties
                new_props = self._predict_properties_from_composition(
                    new_comp, comps, props, prop_cols
                )
                
                # Predict target
                combined_features = np.concatenate([new_comp, new_props])
                original_combined = np.hstack([comps, props])
                new_target = self._predict_target(combined_features, original_combined, targets)
                
            else:
                # Fallback: slight perturbation
                new_comp = comps[base_idx] * (1 + np.random.normal(0, 0.05, len(comps[base_idx])))
                new_comp = np.maximum(new_comp, 0)
                
                new_props = props[base_idx] * (1 + np.random.normal(0, 0.1, len(props[base_idx])))
                new_target = targets[base_idx] * (1 + np.random.normal(0, 0.1))
            
            # Create sample dictionary
            sample = {}
            for i, col in enumerate(comp_cols):
                sample[col] = new_comp[i]
            for i, col in enumerate(prop_cols):
                sample[col] = new_props[i]
            sample['target'] = new_target
            
            samples.append(sample)
        
        return samples
    
    def _predict_properties_from_composition(self, composition: np.ndarray,
                                           original_comps: np.ndarray, 
                                           original_props: np.ndarray,
                                           prop_cols: List[str]) -> np.ndarray:
        """Predict properties from composition using trained models."""
        predicted_props = np.zeros(len(prop_cols))
        
        for i in range(len(prop_cols)):
            model = GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)
            model.fit(original_comps, original_props[:, i])
            predicted_props[i] = model.predict(composition.reshape(1, -1))[0]
        
        return predicted_props
    
    def _predict_target(self, features: np.ndarray, original_features: np.ndarray,
                       original_targets: np.ndarray) -> float:
        """Predict target from combined features."""
        model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        model.fit(original_features, original_targets)
        return model.predict(features.reshape(1, -1))[0]
