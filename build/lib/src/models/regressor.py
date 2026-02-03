"""
Enhanced LNP Regressor - Clean Version
"""
import torch
from .multi_component_net import TunedMultiComponentNet

class TunedLNPRegressor:
    """Enhanced regressor wrapper."""
    
    def __init__(self, model_config, device='cuda'):
        self.device = device
        self.config = model_config
        
        print(f"🔍 DEBUG: Creating TunedLNPRegressor from {__file__}")
        print(f"🔍 DEBUG: Model config: {model_config}")
        
        # Create the enhanced multi-component model
        self.model = TunedMultiComponentNet(
            molecular_feature_dim=model_config['molecular_feature_dim'],
            n_components=model_config['n_components'],
            property_dim=model_config['property_dim'],
            hidden_dims=model_config.get('hidden_dims', [128, 64]),
            num_heads=model_config.get('num_heads', 4),
            dropout_rate=model_config.get('dropout_rate', 0.2)
        ).to(device)
    
    def prepare_data(self, molecular_features, composition_features, property_features, targets):
        """Prepare data tensors."""
        return {
            'molecular': torch.FloatTensor(molecular_features).to(self.device),
            'composition': torch.FloatTensor(composition_features).to(self.device),
            'property': torch.FloatTensor(property_features).to(self.device),
            'targets': torch.FloatTensor(targets).to(self.device)
        }
    
    def predict(self, molecular_features, composition_features, property_features):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            mol_tensor = torch.FloatTensor(molecular_features).to(self.device)
            comp_tensor = torch.FloatTensor(composition_features).to(self.device)
            prop_tensor = torch.FloatTensor(property_features).to(self.device)
            
            predictions, outputs = self.model(mol_tensor, comp_tensor, prop_tensor)
            
        return predictions.cpu().numpy(), outputs

# Model loading utilities
def load_best_model(model_path: str, device: str = 'cuda'):
    """Load the best model from checkpoint."""
    
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    model = TunedLNPRegressor(model_config, device=device)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.eval()
    
    print(f"✅ Loaded model with R² = {checkpoint['val_r2']:.4f}")
    return model, checkpoint
