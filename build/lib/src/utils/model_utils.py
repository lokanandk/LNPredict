"""
Model Utilities for Loading and Saving
"""

import torch
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np

from src.models.regressor import TunedLNPRegressor

def setup_device() -> str:
    """Setup and return the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️  Using CPU (GPU not available)")
    
    return device


def save_model_checkpoint(model, optimizer, scheduler, epoch: int, val_r2: float, 
                         val_loss: float, config: Any, model_config: Dict, 
                         filepath: Path):
    """Save comprehensive model checkpoint."""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'val_r2': val_r2,
        'val_loss': val_loss,
        'config': config,
        'model_config': model_config,
        'best_val_r2': val_r2
    }
    
    torch.save(checkpoint, filepath)


def load_best_model(model_path: str, device: str = 'cpu') -> Tuple[TunedLNPRegressor, Dict]:
    """Load the best model from checkpoint."""
    
    # Simple approach - just like your standalone code
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model
    model_config = checkpoint['model_config']
    model = TunedLNPRegressor(model_config, device=device)
    
    # Load state
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.eval()
    
    print(f"✅ Loaded model with R² = {checkpoint['val_r2']:.4f}")
    return model, checkpoint



def predict_with_saved_model(model_path: str, molecular_features, composition_features, property_features):
    """Make predictions with saved model."""
    
    model, checkpoint = load_best_model(model_path)
    
    # Prepare data
    mol_tensor = torch.FloatTensor(molecular_features).to(model.device)
    comp_tensor = torch.FloatTensor(composition_features).to(model.device) 
    prop_tensor = torch.FloatTensor(property_features).to(model.device)
    
    # Predict
    with torch.no_grad():
        predictions, model_outputs = model.model(mol_tensor, comp_tensor, prop_tensor)
    
    return predictions.cpu().numpy(), model_outputs


def evaluate_saved_model(model_path: str, molecular_features, composition_features, property_features, targets):
    """Evaluate saved model performance."""
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    predictions, _ = predict_with_saved_model(model_path, molecular_features, composition_features, property_features)
    
    metrics = {
        'r2': r2_score(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions)
    }
    
    return metrics, predictions


def count_model_parameters(model) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model) -> Dict[str, Any]:
    """Get comprehensive model summary."""
    
    total_params = count_model_parameters(model)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'non_trainable_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'model_size_mb': model_size_mb,
        'device': next(model.parameters()).device.type
    }
    
    return summary


def freeze_model_layers(model, layer_names: list):
    """Freeze specified layers in the model."""
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                print(f"Frozen layer: {name}")


def unfreeze_model_layers(model, layer_names: list):
    """Unfreeze specified layers in the model."""
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")

