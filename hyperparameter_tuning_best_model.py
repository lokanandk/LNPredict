#!/usr/bin/env python3
"""
Complete Self-Contained Hyperparameter Tuning for LNP Model with Model Saving
Goal: Break through R² < 0.5 barrier and save the best models for future use
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
import time
import itertools
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.molecular_features import MultiComponentFeaturizer

# All model classes inline to avoid import issues
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class TunedMolecularEncoder(nn.Module):
    """Tunable molecular encoder with proper dimension handling."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=64, 
                 num_heads=4, dropout=0.2):
        super().__init__()
        
        # Handle dimension compatibility
        self.num_heads = num_heads
        if input_dim % num_heads != 0:
            padding = num_heads - (input_dim % num_heads)
            self.input_projection = nn.Linear(input_dim, input_dim + padding)
            embed_dim = input_dim + padding
        else:
            self.input_projection = nn.Identity()
            embed_dim = input_dim
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward layers with residual connections
        self.layers = nn.ModuleList()
        dims = [embed_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_projection(x)
        
        # Self-attention with residual
        x_expanded = x.unsqueeze(1)
        attn_out, _ = self.attention(x_expanded, x_expanded, x_expanded)
        x = self.norm1(x + attn_out.squeeze(1))
        
        # Feed-forward with skip connections
        for layer in self.layers:
            residual = x
            x = layer(x)
            if x.shape == residual.shape:
                x = x + 0.1 * residual  # Weighted residual connection
        
        return self.output_proj(x)

class TunedMultiComponentNet(nn.Module):
    """Tunable multi-component network with enhanced architecture."""
    
    def __init__(self, molecular_feature_dim, n_components, property_dim,
                 hidden_dims=[128, 64], num_heads=4, dropout_rate=0.2):
        super().__init__()
        
        self.n_components = n_components
        output_dim = hidden_dims[-1]
        
        # Molecular encoder
        self.molecular_encoder = TunedMolecularEncoder(
            input_dim=molecular_feature_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout_rate
        )
        
        # Component attention with proper head calculation
        attn_heads = min(num_heads, output_dim)
        if output_dim % attn_heads != 0:
            attn_heads = self._find_divisor(output_dim, num_heads)
        
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=output_dim, 
            num_heads=attn_heads, 
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Enhanced property encoder
        self.property_encoder = nn.Sequential(
            nn.Linear(property_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Compositional weighting network
        self.comp_weighting = nn.Sequential(
            nn.Linear(n_components, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, n_components),
            nn.Softmax(dim=1)
        )
        
        # Enhanced fusion with multiple paths
        fusion_dim = output_dim + output_dim + n_components
        
        # Main fusion path
        self.fusion_main = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dims[0], output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Auxiliary fusion path (for ensemble-like behavior)
        self.fusion_aux = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU()
        )
        
        # Dual prediction heads
        self.prediction_head1 = nn.Linear(output_dim, 1)
        self.prediction_head2 = nn.Linear(output_dim // 2, 1)
        
        # Uncertainty and confidence heads
        self.uncertainty_head = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Feature importance predictor
        self.importance_head = nn.Sequential(
            nn.Linear(output_dim, n_components),
            nn.Softmax(dim=1)
        )
        
        param_count = sum(p.numel() for p in self.parameters())
        print(f"  Created tuned model: {param_count:,} parameters")
    
    def _find_divisor(self, n, target):
        """Find largest divisor of n that's <= target."""
        for i in range(min(target, n), 0, -1):
            if n % i == 0:
                return i
        return 1
    
    def forward(self, molecular_features, compositions, properties):
        batch_size = molecular_features.shape[0]
        
        # Encode molecular features
        mol_flat = molecular_features.reshape(-1, molecular_features.size(-1))
        mol_encoded = self.molecular_encoder(mol_flat)
        mol_encoded = mol_encoded.reshape(batch_size, self.n_components, -1)
        
        # Enhanced attention fusion
        attended_mol, attention_weights = self.attention_fusion(
            mol_encoded, mol_encoded, mol_encoded
        )
        
        # Smart compositional weighting
        enhanced_comp_weights = self.comp_weighting(compositions)
        final_mol = (attended_mol * enhanced_comp_weights.unsqueeze(-1)).sum(dim=1)
        
        # Enhanced property encoding
        prop_encoded = self.property_encoder(properties)
        
        # Multi-path fusion
        combined_features = torch.cat([final_mol, prop_encoded, compositions], dim=1)
        
        main_features = self.fusion_main(combined_features)
        aux_features = self.fusion_aux(combined_features)
        
        # Dual predictions
        pred1 = self.prediction_head1(main_features)
        pred2 = self.prediction_head2(aux_features)
        
        # Ensemble prediction with learned weighting
        ensemble_weight = torch.sigmoid(pred1.detach())  # Use pred1 confidence as weight
        final_prediction = ensemble_weight * pred1 + (1 - ensemble_weight) * pred2
        
        # Auxiliary outputs
        uncertainty = self.uncertainty_head(main_features)
        importance = self.importance_head(main_features)
        
        return final_prediction.squeeze(-1), {
            'uncertainty': uncertainty.squeeze(-1),
            'feature_importance': importance,
            'attention_weights': enhanced_comp_weights,
            'individual_predictions': [pred1.squeeze(-1), pred2.squeeze(-1)]
        }

class TunedLNPRegressor:
    """Enhanced regressor with tunable training and model saving."""
    
    def __init__(self, model_config, device='cuda'):
        self.device = device
        self.config = model_config
        
        self.model = TunedMultiComponentNet(
            molecular_feature_dim=model_config['molecular_feature_dim'],
            n_components=model_config['n_components'],
            property_dim=model_config['property_dim'],
            hidden_dims=model_config.get('hidden_dims', [128, 64]),
            num_heads=model_config.get('num_heads', 4),
            dropout_rate=model_config.get('dropout_rate', 0.2)
        ).to(device)
    
    def prepare_data(self, molecular_features, composition_features, property_features, targets):
        return {
            'molecular': torch.FloatTensor(molecular_features).to(self.device),
            'composition': torch.FloatTensor(composition_features).to(self.device),
            'property': torch.FloatTensor(property_features).to(self.device),
            'targets': torch.FloatTensor(targets).to(self.device)
        }

@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration."""
    learning_rate: float
    weight_decay: float
    dropout_rate: float
    batch_size: int
    hidden_dims: List[int]
    num_heads: int
    early_stopping_patience: int
    scheduler_type: str
    loss_weights: Tuple[float, float, float]  # MSE, Huber, L1
    
def create_stratified_bins(targets, n_bins=5):
    """Create bins for stratified splitting."""
    try:
        bin_edges = np.quantile(targets, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return None
        return np.digitize(targets, bin_edges[1:-1])
    except Exception:
        return None

class ComprehensiveHyperparameterTuner:
    """Enhanced hyperparameter tuner with model saving capabilities."""
    
    def __init__(self, data_config: Dict):
        self.data_config = data_config
        self.results = []
        self.best_config = None
        self.best_score = -np.inf
        self.best_model_path = None
        
    def create_enhanced_search_space(self) -> List[HyperparameterConfig]:
        """Create comprehensive search space focused on breaking R² barrier."""
        
        param_grid = {
            # Learning rates - wider range including higher values
            'learning_rate': [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3],
            
            # Weight decay - including no regularization and strong regularization
            'weight_decay': [0, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2],
            
            # Dropout rates - wider range to combat overfitting
            'dropout_rate': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            
            # Batch sizes
            'batch_size': [8, 16, 32, 64],
            
            # Architecture variations
            'hidden_dims': [
                [64],           # Very simple
                [128],          # Simple
                [64, 32],       # Small
                [128, 64],      # Current
                [256, 128],     # Larger
                [128, 64, 32],  # Deeper
                [256, 128, 64], # Large & deep
            ],
            
            # Attention heads
            'num_heads': [2, 4, 6, 8],
            
            # Early stopping - shorter patience to combat saturation
            'early_stopping_patience': [30, 50, 75, 100],
            
            # Scheduler types
            'scheduler_type': ['cosine', 'plateau', 'step', 'exponential', 'linear'],
            
            # Loss function weights [MSE, Huber, L1]
            'loss_weights': [
                (1.0, 0.0, 0.0),    # Pure MSE
                (0.8, 0.2, 0.0),    # MSE + Huber
                (0.6, 0.3, 0.1),    # Balanced
                (0.4, 0.4, 0.2),    # Heavy robust losses
                (0.5, 0.3, 0.2),    # Mixed
            ]
        }
        
        # Generate combinations with smart sampling
        keys = list(param_grid.keys())
        
        # First, create some high-priority combinations
        priority_configs = []
        
        # High learning rate + low regularization (for breakthrough)
        for lr in [1e-3, 3e-3, 5e-3]:
            for wd in [0, 1e-6, 1e-5]:
                for dropout in [0.1, 0.2]:
                    priority_configs.append({
                        'learning_rate': lr, 'weight_decay': wd, 'dropout_rate': dropout,
                        'batch_size': 32, 'hidden_dims': [128, 64], 'num_heads': 4,
                        'early_stopping_patience': 50, 'scheduler_type': 'cosine',
                        'loss_weights': (0.6, 0.3, 0.1)
                    })
        
        # Then add random sampling
        all_combinations = list(itertools.product(*(param_grid[k] for k in keys)))
        random.seed(42)  # For reproducibility
        random_combinations = random.sample(all_combinations, min(15, len(all_combinations)))
        
        # Combine priority and random
        all_configs = []
        
        # Add priority configurations
        for config_dict in priority_configs[:10]:  # Limit priority configs
            all_configs.append(HyperparameterConfig(**config_dict))
        
        # Add random configurations
        for combo in random_combinations:
            config = HyperparameterConfig(**dict(zip(keys, combo)))
            all_configs.append(config)
        
        print(f"Created {len(all_configs)} hyperparameter configurations")
        print(f"  - {len(priority_configs[:10])} priority configs (high LR + low reg)")
        print(f"  - {len(random_combinations)} random configs")
        
        return all_configs
    
    def train_with_config(self, model: TunedLNPRegressor, train_data, val_data, 
                         config: HyperparameterConfig, output_dir: Path, trial_num: int) -> Tuple[List[float], List[float], float]:
        """Enhanced training with anti-saturation measures and model saving."""
        
        # Create trial directory
        trial_dir = output_dir / f"trial_{trial_num}"
        trial_dir.mkdir(exist_ok=True)
        
        # Optimizer with configurable parameters
        optimizer = torch.optim.AdamW(
            model.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            eps=1e-8,
            amsgrad=True  # Better convergence
        )
        
        # Scheduler based on configuration
        if config.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2, eta_min=config.learning_rate * 0.001
            )
        elif config.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.3, patience=15, min_lr=1e-8
            )
        elif config.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100, 200, 400], gamma=0.3
            )
        elif config.scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        else:  # linear
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=500
            )
        
        # Multiple loss functions
        mse_loss = nn.MSELoss()
        huber_loss = nn.HuberLoss(delta=1.5)
        l1_loss = nn.L1Loss()
        
        train_losses, val_losses = [], []
        best_val_r2 = -np.inf
        patience_counter = 0
        
        # Anti-saturation measures
        lr_reduce_counter = 0
        stagnation_threshold = 20  # Epochs without improvement
        
        max_epochs = 600  # Prevent excessive training
        
        for epoch in range(max_epochs):
            model.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            predictions, model_outputs = model.model(
                train_data['molecular'],
                train_data['composition'],
                train_data['property']
            )
            
            # Multi-component loss with configurable weights
            mse = mse_loss(predictions, train_data['targets'])
            huber = huber_loss(predictions, train_data['targets'])
            l1 = l1_loss(predictions, train_data['targets'])
            
            # Ensemble consistency loss
            individual_preds = model_outputs.get('individual_predictions', [])
            consistency_loss = 0
            if len(individual_preds) >= 2:
                consistency_loss = mse_loss(individual_preds[0], individual_preds[1])
            
            # Uncertainty regularization
            uncertainty_reg = torch.mean(model_outputs.get('uncertainty', torch.zeros(1).to(model.device)))
            
            # Feature importance entropy (encourage diversity)
            importance = model_outputs.get('feature_importance')
            if importance is not None:
                importance_entropy = -torch.mean(torch.sum(importance * torch.log(importance + 1e-8), dim=1))
            else:
                importance_entropy = torch.tensor(0.0).to(model.device)
            
            # Combined loss with configurable weights
            w1, w2, w3 = config.loss_weights
            total_loss = (w1 * mse + w2 * huber + w3 * l1 + 
                         0.1 * consistency_loss + 
                         0.05 * uncertainty_reg + 
                         0.02 * importance_entropy)
            
            # Gradient penalty for stability
            grad_penalty = 0
            for param in model.model.parameters():
                if param.grad is not None:
                    grad_penalty += torch.norm(param.grad, p=2) ** 2
            total_loss += 1e-7 * grad_penalty
            
            total_loss.backward()
            
            # Adaptive gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Scheduler step
            if config.scheduler_type == 'plateau':
                scheduler.step(total_loss)
            else:
                scheduler.step()
            
            train_losses.append(total_loss.item())
            
            # Validation every 5 epochs for faster feedback
            if epoch % 5 == 0:
                model.model.eval()
                with torch.no_grad():
                    val_predictions, _ = model.model(
                        val_data['molecular'],
                        val_data['composition'],
                        val_data['property']
                    )
                    val_loss = mse_loss(val_predictions, val_data['targets'])
                    
                    # Calculate validation R²
                    val_r2 = r2_score(
                        val_data['targets'].cpu().numpy(),
                        val_predictions.cpu().numpy()
                    )
                    
                    val_losses.append(val_loss.item())
                    
                    # 🔥 SAVE BEST MODEL FOR THIS TRIAL
                    if val_r2 > best_val_r2 + 1e-5:  # Small improvement threshold
                        best_val_r2 = val_r2
                        patience_counter = 0
                        
                        # Save comprehensive checkpoint
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                            'val_r2': val_r2,
                            'val_loss': val_loss.item(),
                            'train_loss': total_loss.item(),
                            'config': config,
                            'model_config': model.config,
                            'best_val_r2': best_val_r2
                        }
                        
                        model_path = trial_dir / f'best_model_trial_{trial_num}.pth'
                        torch.save(checkpoint, model_path)
                        print(f"    💾 Saved best model: R²={val_r2:.4f} at epoch {epoch}")
                        
                    else:
                        patience_counter += 1
                    
                    # Early stopping with reduced patience
                    if patience_counter >= config.early_stopping_patience:
                        break
                    
                    # Anti-saturation: Reduce LR if stagnating
                    if patience_counter > stagnation_threshold and lr_reduce_counter < 3:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        lr_reduce_counter += 1
                        patience_counter = 0  # Reset patience after LR reduction
                        print(f"    Reduced LR to {optimizer.param_groups[0]['lr']:.2e}")
        
        return train_losses, val_losses, best_val_r2
    
    def run_trial(self, config: HyperparameterConfig, trial_num: int, output_dir: Path) -> Dict:
        """Run single hyperparameter trial with model saving."""
        
        print(f"\n🔬 TRIAL {trial_num}")
        print(f"  LR: {config.learning_rate:.0e}, WD: {config.weight_decay:.0e}, "
              f"Dropout: {config.dropout_rate:.2f}")
        print(f"  Architecture: {config.hidden_dims}, Heads: {config.num_heads}")
        print(f"  Scheduler: {config.scheduler_type}, Patience: {config.early_stopping_patience}")
        print(f"  Loss weights: {config.loss_weights}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        model_config = {
            'molecular_feature_dim': self.data_config['molecular_feature_dim'],
            'n_components': self.data_config['n_components'],
            'property_dim': self.data_config['property_dim'],
            'hidden_dims': config.hidden_dims,
            'num_heads': config.num_heads,
            'dropout_rate': config.dropout_rate
        }
        
        model = TunedLNPRegressor(model_config, device=device)
        
        # Training with model saving
        start_time = time.time()
        train_losses, val_losses, best_val_r2 = self.train_with_config(
            model, self.data_config['train_data'], self.data_config['val_data'], 
            config, output_dir, trial_num
        )
        training_time = time.time() - start_time
        
        # Final evaluation
        model.model.eval()
        with torch.no_grad():
            final_predictions, _ = model.model(
                self.data_config['val_data']['molecular'],
                self.data_config['val_data']['composition'],
                self.data_config['val_data']['property']
            )
            
            targets_np = self.data_config['val_data']['targets'].cpu().numpy()
            preds_np = final_predictions.cpu().numpy()
            
            final_r2 = r2_score(targets_np, preds_np)
            final_rmse = np.sqrt(mean_squared_error(targets_np, preds_np))
            final_mae = mean_absolute_error(targets_np, preds_np)
        
        trial_model_path = output_dir / f"trial_{trial_num}" / f'best_model_trial_{trial_num}.pth'
        
        result = {
            'trial_num': trial_num,
            'config': config,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'final_mae': final_mae,
            'best_val_r2': best_val_r2,
            'training_time': training_time,
            'epochs_trained': len(train_losses),
            'final_train_loss': train_losses[-1] if train_losses else np.inf,
            'final_val_loss': val_losses[-1] if val_losses else np.inf,
            'converged_early': len(val_losses) * 5 < config.early_stopping_patience * 10,
            'model_path': str(trial_model_path)  # 🔥 ADD MODEL PATH
        }
        
        breakthrough = "🎯 BREAKTHROUGH!" if final_r2 >= 0.5 else ""
        print(f"  ✅ R²: {final_r2:.4f} {breakthrough}")
        print(f"     RMSE: {final_rmse:.4f}, Time: {training_time/60:.1f}min, Epochs: {len(train_losses)}")
        print(f"     Model saved: {trial_model_path}")
        
        return result
    
    def run_comprehensive_search(self, output_dir: Path):
        """Run comprehensive hyperparameter search with model tracking."""
        
        configs = self.create_enhanced_search_space()
        
        print(f"\n🚀 COMPREHENSIVE HYPERPARAMETER SEARCH")
        print(f"Goal: Break through R² < 0.5 barrier")
        print(f"Configurations to test: {len(configs)}")
        print(f"Models will be saved to: {output_dir}")
        print("="*80)
        
        breakthrough_configs = []
        
        for i, config in enumerate(configs, 1):
            try:
                result = self.run_trial(config, i, output_dir)
                self.results.append(result)
                
                # Track breakthrough configurations
                if result['final_r2'] >= 0.5:
                    breakthrough_configs.append(result)
                
                # 🏆 UPDATE OVERALL BEST MODEL
                if result['final_r2'] > self.best_score:
                    self.best_score = result['final_r2']
                    self.best_config = config
                    self.best_model_path = result['model_path']
                    
                    # COPY BEST MODEL TO ROOT DIRECTORY
                    best_model_dest = output_dir / 'BEST_OVERALL_MODEL.pth'
                    shutil.copy2(result['model_path'], best_model_dest)
                    print(f"  🏆 NEW OVERALL BEST! R²={self.best_score:.4f} -> {best_model_dest}")
                    
            except Exception as e:
                print(f"  ❌ Trial {i} failed: {e}")
                continue
        
        return self.analyze_results(breakthrough_configs)
    
    def analyze_results(self, breakthrough_configs):
        """Analyze comprehensive results."""
        
        if not self.results:
            return {"error": "No successful trials"}
        
        sorted_results = sorted(self.results, key=lambda x: x['final_r2'], reverse=True)
        
        analysis = {
            'best_r2': sorted_results[0]['final_r2'],
            'best_config': sorted_results[0]['config'],
            'best_model_path': self.best_model_path,
            'breakthrough_achieved': len(breakthrough_configs) > 0,
            'breakthrough_configs': breakthrough_configs,
            'top_10_results': sorted_results[:10],
            'avg_r2': np.mean([r['final_r2'] for r in self.results]),
            'std_r2': np.std([r['final_r2'] for r in self.results]),
            'median_r2': np.median([r['final_r2'] for r in self.results]),
            'num_trials': len(self.results),
            'early_convergence_rate': sum(1 for r in self.results if r['converged_early']) / len(self.results),
            'avg_training_time': np.mean([r['training_time'] for r in self.results])
        }
        
        return analysis

# 🔥 MODEL LOADING UTILITIES
def load_best_model(model_path: str, device: str = 'cuda') -> Tuple[TunedLNPRegressor, Dict]:
    """Load the best model from checkpoint."""
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model
    model_config = checkpoint['model_config']
    model = TunedLNPRegressor(model_config, device=device)
    
    # Load state
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.eval()
    
    print(f"✅ Loaded model with R² = {checkpoint['val_r2']:.4f}")
    print(f"   Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"   Config: LR={checkpoint['config'].learning_rate:.0e}, "
          f"Dropout={checkpoint['config'].dropout_rate:.2f}")
    
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
    
    predictions, _ = predict_with_saved_model(model_path, molecular_features, composition_features, property_features)
    
    metrics = {
        'r2': r2_score(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions)
    }
    
    return metrics, predictions

def main():
    parser = argparse.ArgumentParser(description='Comprehensive hyperparameter tuning with model saving')
    parser.add_argument('--data', required=True, help='Path to original JSON data')
    parser.add_argument('--synthetic', help='Path to synthetic JSON data')
    parser.add_argument('--output', default='comprehensive_tuning_results', help='Output directory')
    args = parser.parse_args()
    
    print("🎯 === COMPREHENSIVE LNP HYPERPARAMETER TUNING WITH MODEL SAVING ===")
    print("Mission: Break through R² < 0.5 barrier and save best models")
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Data loading and preprocessing
    featurizer = MultiComponentFeaturizer()
    features, targets, feature_names = featurizer.create_comprehensive_features(args.data)
    
    if args.synthetic:
        synth_features, synth_targets, _ = featurizer.create_comprehensive_features(args.synthetic)
        features = np.vstack([features, synth_features])
        targets = np.hstack([targets, synth_targets])
    
    # Preprocessing
    mol_mask = [name.startswith('mol_') for name in feature_names]
    comp_mask = [name.startswith('comp_') for name in feature_names]
    prop_mask = [name.startswith('prop_') for name in feature_names]
    
    molecular_features = features[:, mol_mask]
    composition_features = features[:, comp_mask]
    property_features = features[:, prop_mask]
    
    scaler = RobustScaler()
    property_features = scaler.fit_transform(property_features)
    
    # Molecular feature preparation
    n_components = composition_features.shape[1]
    mol_feat_per_comp = molecular_features.shape[1] // n_components
    
    if mol_feat_per_comp % 4 != 0:
        padding_needed = 4 - (mol_feat_per_comp % 4)
        mol_feat_per_comp += padding_needed
    
    total_mol_feat_needed = n_components * mol_feat_per_comp
    if molecular_features.shape[1] < total_mol_feat_needed:
        padding = total_mol_feat_needed - molecular_features.shape[1]
        molecular_features = np.pad(molecular_features, ((0, 0), (0, padding)), 'constant')
    
    molecular_features = molecular_features.reshape(-1, n_components, mol_feat_per_comp)
    
    # Split
    indices = np.arange(len(targets))
    target_bins = create_stratified_bins(targets, n_bins=5)
    
    if target_bins is not None:
        train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=target_bins)
    else:
        train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    
    # Prepare data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
        'val_data': val_data
    }
    
    # Run comprehensive search with model saving
    tuner = ComprehensiveHyperparameterTuner(data_config)
    analysis = tuner.run_comprehensive_search(output_dir)
    
    # Results
    print("\n" + "="*80)
    print("🏆 === COMPREHENSIVE TUNING RESULTS ===")
    
    if analysis['breakthrough_achieved']:
        print(f"🎉 SUCCESS! BREAKTHROUGH ACHIEVED!")
        print(f"   Best R²: {analysis['best_r2']:.4f}")
        print(f"   Number of breakthrough configs: {len(analysis['breakthrough_configs'])}")
    else:
        print(f"📈 Progress made, best R²: {analysis['best_r2']:.4f}")
        if analysis['best_r2'] > 0.45:
            print("   Very close to breakthrough!")
        
    print(f"\n📊 Statistics:")
    print(f"   Average R²: {analysis['avg_r2']:.4f} ± {analysis['std_r2']:.4f}")
    print(f"   Median R²: {analysis['median_r2']:.4f}")
    print(f"   Successful trials: {analysis['num_trials']}")
    print(f"   Average training time: {analysis['avg_training_time']/60:.1f} min")
    
    best_config = analysis['best_config']
    print(f"\n🏅 Best Configuration:")
    print(f"   Learning Rate: {best_config.learning_rate:.0e}")
    print(f"   Weight Decay: {best_config.weight_decay:.0e}")
    print(f"   Dropout: {best_config.dropout_rate:.2f}")
    print(f"   Architecture: {best_config.hidden_dims}")
    print(f"   Attention Heads: {best_config.num_heads}")
    print(f"   Scheduler: {best_config.scheduler_type}")
    print(f"   Loss Weights: {best_config.loss_weights}")
    
    # 🔥 SAVE COMPREHENSIVE RESULTS WITH MODEL INFO
    with open(output_dir / 'comprehensive_results.json', 'w') as f:
        # Make serializable
        serializable_analysis = analysis.copy()
        serializable_analysis['best_model_path'] = str(analysis['best_model_path']) if analysis['best_model_path'] else None
        serializable_analysis['best_config'] = {
            'learning_rate': best_config.learning_rate,
            'weight_decay': best_config.weight_decay,
            'dropout_rate': best_config.dropout_rate,
            'hidden_dims': best_config.hidden_dims,
            'num_heads': best_config.num_heads,
            'scheduler_type': best_config.scheduler_type,
            'loss_weights': best_config.loss_weights
        }
        json.dump(serializable_analysis, f, indent=2, default=str)
    
    # Enhanced visualization
    plt.figure(figsize=(18, 12))
    
    # R² distribution
    plt.subplot(3, 3, 1)
    r2_scores = [r['final_r2'] for r in tuner.results]
    plt.hist(r2_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(analysis['best_r2'], color='red', linestyle='--', linewidth=2, label=f'Best: {analysis["best_r2"]:.3f}')
    plt.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Target: 0.5')
    plt.xlabel('R² Score')
    plt.ylabel('Frequency')
    plt.title('R² Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate vs R²
    plt.subplot(3, 3, 2)
    lrs = [r['config'].learning_rate for r in tuner.results]
    r2s = [r['final_r2'] for r in tuner.results]
    scatter = plt.scatter(lrs, r2s, c=r2s, cmap='viridis', s=60, alpha=0.7)
    plt.colorbar(scatter)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('R² Score')
    plt.title('Learning Rate vs Performance')
    plt.grid(True, alpha=0.3)
    
    # Weight decay vs R²
    plt.subplot(3, 3, 3)
    wds = [r['config'].weight_decay for r in tuner.results]
    plt.scatter(wds, r2s, c=r2s, cmap='plasma', s=60, alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Weight Decay')
    plt.ylabel('R² Score')
    plt.title('Regularization vs Performance')
    plt.grid(True, alpha=0.3)
    
    # Dropout vs R²
    plt.subplot(3, 3, 4)
    dropouts = [r['config'].dropout_rate for r in tuner.results]
    plt.scatter(dropouts, r2s, c=r2s, cmap='coolwarm', s=60, alpha=0.7)
    plt.xlabel('Dropout Rate')
    plt.ylabel('R² Score')
    plt.title('Dropout vs Performance')
    plt.grid(True, alpha=0.3)
    
    # Training time vs R²
    plt.subplot(3, 3, 5)
    times = [r['training_time']/60 for r in tuner.results]  # Convert to minutes
    plt.scatter(times, r2s, c=r2s, cmap='viridis', s=60, alpha=0.7)
    plt.xlabel('Training Time (minutes)')
    plt.ylabel('R² Score')
    plt.title('Efficiency vs Performance')
    plt.grid(True, alpha=0.3)
    
    # Top configurations
    plt.subplot(3, 3, 6)
    top_10 = analysis['top_10_results'][:10]
    trial_nums = [r['trial_num'] for r in top_10]
    top_r2s = [r['final_r2'] for r in top_10]
    colors = ['red' if r2 >= 0.5 else 'orange' if r2 >= 0.45 else 'skyblue' for r2 in top_r2s]
    bars = plt.bar(range(len(trial_nums)), top_r2s, color=colors, alpha=0.7)
    plt.xlabel('Rank')
    plt.ylabel('R² Score')
    plt.title('Top 10 Configurations')
    plt.xticks(range(len(trial_nums)), [f'#{t}' for t in trial_nums], rotation=45)
    for i, (bar, score) in enumerate(zip(bars, top_r2s)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Architecture analysis
    plt.subplot(3, 3, 7)
    arch_performance = {}
    for result in tuner.results:
        arch_key = str(result['config'].hidden_dims)
        if arch_key not in arch_performance:
            arch_performance[arch_key] = []
        arch_performance[arch_key].append(result['final_r2'])
    
    arch_names = list(arch_performance.keys())
    arch_scores = [np.mean(arch_performance[arch]) for arch in arch_names]
    arch_names_short = [arch.replace(' ', '') for arch in arch_names]
    
    bars = plt.bar(range(len(arch_names)), arch_scores, alpha=0.7, color='lightcoral')
    plt.xlabel('Architecture')
    plt.ylabel('Average R²')
    plt.title('Architecture Comparison')
    plt.xticks(range(len(arch_names)), arch_names_short, rotation=45, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Scheduler comparison
    plt.subplot(3, 3, 8)
    sched_performance = {}
    for result in tuner.results:
        sched = result['config'].scheduler_type
        if sched not in sched_performance:
            sched_performance[sched] = []
        sched_performance[sched].append(result['final_r2'])
    
    sched_names = list(sched_performance.keys())
    sched_scores = [np.mean(sched_performance[sched]) for sched in sched_names]
    
    bars = plt.bar(range(len(sched_names)), sched_scores, alpha=0.7, color='lightgreen')
    plt.xlabel('Scheduler Type')
    plt.ylabel('Average R²')
    plt.title('Scheduler Comparison')
    plt.xticks(range(len(sched_names)), sched_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Progress over trials
    plt.subplot(3, 3, 9)
    trial_numbers = [r['trial_num'] for r in tuner.results]
    cumulative_best = []
    current_best = -np.inf
    for r2 in r2_scores:
        if r2 > current_best:
            current_best = r2
        cumulative_best.append(current_best)
    
    plt.plot(trial_numbers, cumulative_best, 'b-', linewidth=2, label='Best R² So Far')
    plt.axhline(0.5, color='red', linestyle='--', label='Target R² = 0.5')
    plt.xlabel('Trial Number')
    plt.ylabel('Best R² Achieved')
    plt.title('Search Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Comprehensive hyperparameter tuning completed!")
    print(f"📁 Results saved to: {output_dir}/")
    
    # 🔥 PROVIDE MODEL USAGE INSTRUCTIONS
    if analysis['best_model_path']:
        print(f"\n🏆 === BEST MODEL SAVED ===")
        print(f"📁 Best model path: {output_dir}/BEST_OVERALL_MODEL.pth")
        print(f"🎯 Best R²: {analysis['best_r2']:.4f}")
        print(f"\n💡 To use the saved model:")
        print(f"# Import utilities")
        print(f"from your_script import load_best_model, predict_with_saved_model, evaluate_saved_model")
        print(f"")
        print(f"# Load best model")
        print(f"model, checkpoint = load_best_model('{output_dir}/BEST_OVERALL_MODEL.pth')")
        print(f"")
        print(f"# Make predictions")
        print(f"predictions, outputs = predict_with_saved_model(")
        print(f"    '{output_dir}/BEST_OVERALL_MODEL.pth',")
        print(f"    molecular_features, composition_features, property_features")
        print(f")")
        print(f"print(f'Predictions: {{predictions}}')") 
        print(f"print(f'Uncertainties: {{outputs[\"uncertainty\"]}}')") 
        print(f"")
        print(f"# Evaluate model")
        print(f"metrics, preds = evaluate_saved_model(")
        print(f"    '{output_dir}/BEST_OVERALL_MODEL.pth',")
        print(f"    test_molecular, test_composition, test_property, test_targets")
        print(f")")
        print(f"print(f'Test R²: {{metrics[\"r2\"]:.4f}}')") 
        print(f"```")
    
    if analysis['breakthrough_achieved']:
        print("\n🎉 MISSION ACCOMPLISHED! R² >= 0.5 achieved!")
        print("🔧 Use the best configuration for your production model.")
    else:
        print("\n📈 Significant progress made. Consider:")
        print("   - Extending search with more configurations")
        print("   - Trying different architectures")
        print("   - Collecting more training data")

if __name__ == "__main__":
    main()
