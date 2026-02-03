"""
Nearest Neighbor Augmentation Strategy

Generates synthetic samples by creating weighted combinations of
nearest neighbors in the feature space.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict


class NearestNeighborAugmentation:
    """Nearest neighbor augmentation strategy."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_samples(self, comps: np.ndarray, props: np.ndarray,
                        targets: np.ndarray, n_samples: int,
                        comp_cols: List[str], prop_cols: List[str]) -> List[Dict]:
        """Generate samples by augmenting nearest neighbors."""
        samples = []
        
        # Use k-NN to find similar samples
        nn_model = NearestNeighbors(n_neighbors=min(5, len(comps)), metric='cosine')
        combined_features = np.hstack([comps, props])
        nn_model.fit(combined_features)
        
        for _ in range(n_samples):
            # Pick random base sample
            base_idx = np.random.randint(len(comps))
            base_features = combined_features[base_idx:base_idx+1]
            
            # Find nearest neighbors
            distances, indices = nn_model.kneighbors(base_features)
            neighbor_indices = indices[0][1:]  # Exclude self
            
            if len(neighbor_indices) > 0:
                # Pick a random neighbor
                neighbor_idx = np.random.choice(neighbor_indices)
                
                # Create weighted combination
                weights = np.random.dirichlet([2, 1])  # Favor base sample
                
                new_comp = weights[0] * comps[base_idx] + weights[1] * comps[neighbor_idx]
                new_props = weights[0] * props[base_idx] + weights[1] * props[neighbor_idx]
                new_target = weights[0] * targets[base_idx] + weights[1] * targets[neighbor_idx]
                
                # Add noise for diversity
                new_comp += np.random.normal(0, np.std(comps, axis=0) * 0.02, len(new_comp))
                new_comp = np.maximum(new_comp, 0)
                
                new_props += np.random.normal(0, np.std(props, axis=0) * 0.05, len(new_props))
                new_target += np.random.normal(0, np.std(targets) * 0.05)
                
            else:
                # Fallback: perturb base sample
                new_comp = comps[base_idx] + np.random.normal(0, np.std(comps, axis=0) * 0.05, len(comps[base_idx]))
                new_comp = np.maximum(new_comp, 0)
                
                new_props = props[base_idx] + np.random.normal(0, np.std(props, axis=0) * 0.1, len(props[base_idx]))
                new_target = targets[base_idx] + np.random.normal(0, np.std(targets) * 0.1)
            
            # Create sample
            sample = {}
            for i, col in enumerate(comp_cols):
                sample[col] = new_comp[i]
            for i, col in enumerate(prop_cols):
                sample[col] = new_props[i]
            sample['target'] = new_target
            
            samples.append(sample)
        
        return samples
