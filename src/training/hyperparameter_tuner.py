"""
Enhanced Hyperparameter Tuner with Breakthrough-Focused Configuration
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import itertools
import random
import time
import json
import shutil
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.models.regressor import TunedLNPRegressor

@dataclass
class HyperparameterConfig:
    """Enhanced hyperparameter configuration."""
    learning_rate: float
    weight_decay: float
    dropout_rate: float
    batch_size: int
    hidden_dims: List[int]
    num_heads: int
    early_stopping_patience: int
    scheduler_type: str
    loss_weights: Tuple[float, float, float]

class ComprehensiveHyperparameterTuner:
    """Enhanced hyperparameter tuner optimized for breakthrough performance."""
    
    def __init__(self, data_config: Dict, tuning_config: Dict):
        self.data_config = data_config
        self.tuning_config = tuning_config
        self.results = []
        self.best_config = None
        self.best_score = -np.inf
        self.best_model_path = None
        
    def create_breakthrough_search_space(self) -> List[HyperparameterConfig]:
        """Create search space optimized for R² > 0.5 breakthrough."""
        
        # FIXED: Updated parameter ranges for stability
        param_grid = {
            'learning_rate': [1e-4, 5e-4, 1e-3, 3e-3],     # Reduced from 5e-6, 1e-5
            'weight_decay': [1e-4, 3e-4, 5e-4, 1e-3],      # Stronger regularization
            'dropout_rate': [0.25, 0.3, 0.35, 0.4],        # Higher dropout
            'batch_size': [32, 64],                         # Larger batches for batch norm
            'hidden_dims': [
                [256, 128, 64],      # Deep
                [384, 192, 96],      # Balanced deep
                [512, 256, 128],     # Very deep
                [256, 256, 128],     # Wide
                [384, 256, 128]      # Wide and deep
            ],
            'num_heads': [8, 12],                           # More attention heads
            'early_stopping_patience': [25, 50, 100],      # REDUCED: Lower patience
            'scheduler_type': ['cosine', 'plateau'],        # FIXED: Removed 'onecycle'
            'loss_weights': [
                (0.8, 0.15, 0.05),   # MSE focused
                (0.7, 0.2, 0.1),     # Balanced
                (0.6, 0.25, 0.15),   # More robust losses
            ]
        }
        
        # Create breakthrough-focused priority configurations
        breakthrough_configs = []
        
        # High-performance configurations
        for lr in [5e-4, 1e-3]:  # Conservative learning rates
            for arch in [[256, 128, 64], [384, 192, 96], [512, 256, 128]]:
                for wd in [1e-4, 3e-4]:
                    for dropout in [0.3, 0.35]:
                        breakthrough_configs.append(HyperparameterConfig(
                            learning_rate=lr,
                            weight_decay=wd,
                            dropout_rate=dropout,
                            batch_size=64,  # Larger batch for stability
                            hidden_dims=arch,
                            num_heads=8,
                            early_stopping_patience=50,  # REDUCED: Lower patience
                            scheduler_type='cosine',     # FIXED: Use stable scheduler
                            loss_weights=(0.7, 0.2, 0.1)
                        ))
        
        # Add some random sampling from full grid
        keys = list(param_grid.keys())
        all_combinations = list(itertools.product(*(param_grid[k] for k in keys)))
        random.seed(42)
        max_random = min(6, len(all_combinations))  # REDUCED: Fewer random samples
        random_combinations = random.sample(all_combinations, max_random)
        
        for combo in random_combinations:
            config_dict = dict(zip(keys, combo))
            breakthrough_configs.append(HyperparameterConfig(
                learning_rate=float(config_dict['learning_rate']),
                weight_decay=float(config_dict['weight_decay']),
                dropout_rate=float(config_dict['dropout_rate']),
                batch_size=int(config_dict['batch_size']),
                hidden_dims=config_dict['hidden_dims'],
                num_heads=int(config_dict['num_heads']),
                early_stopping_patience=int(config_dict['early_stopping_patience']),
                scheduler_type=str(config_dict['scheduler_type']),
                loss_weights=config_dict['loss_weights']
            ))
        
        # Limit total configs
        max_trials = self.tuning_config.get('max_trials', 15)
        final_configs = breakthrough_configs[:max_trials]
        
        print(f"Created {len(final_configs)} breakthrough-focused configurations")
        print(f"  - Architecture focus: Deeper networks [256,128,64] to [512,256,128]")
        print(f"  - Learning rate range: 1e-4 to 3e-3 for stable convergence")  # UPDATED
        print(f"  - Enhanced regularization with higher dropout and weight decay")
        
        return final_configs
    
    def train_with_config(self, model: TunedLNPRegressor, train_data, val_data, 
                         config: HyperparameterConfig, output_dir: Path, trial_num: int) -> Tuple[List[float], List[float], float]:
        """Enhanced training with breakthrough-focused optimization."""
        
        trial_dir = output_dir / f"trial_{trial_num}"
        trial_dir.mkdir(exist_ok=True)
        
        # 🔥 ADAMW OPTIMIZER WITH PROPER WEIGHT DECAY
        optimizer = torch.optim.AdamW(
            model.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True
        )
        
        # 🔥 FIXED LEARNING RATE SCHEDULERS
        if config.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=30, T_mult=2, eta_min=config.learning_rate * 0.01
            )
        elif config.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=15, 
                min_lr=config.learning_rate * 0.001, verbose=True
            )
        else:  # Default to cosine for any unknown scheduler
            print(f"    ⚠️ Unknown scheduler '{config.scheduler_type}', using cosine")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=30, T_mult=2, eta_min=config.learning_rate * 0.01
            )
        
        # Loss functions
        mse_loss = nn.MSELoss()
        huber_loss = nn.HuberLoss(delta=1.5)
        l1_loss = nn.L1Loss()
        
        # 🔥 ENHANCED BATCH PROCESSING
        batch_size = config.batch_size
        n_train_samples = len(train_data['targets'])
        n_batches = max(1, n_train_samples // batch_size)
        
        print(f"🔍 Enhanced training: Batch size={batch_size}, Samples={n_train_samples}, Batches={n_batches}")
        
        train_losses, val_losses = [], []
        best_val_r2 = -np.inf
        patience_counter = 0
        max_epochs = 100  # REDUCED: Lower epochs for stability
        
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            model.model.train()
            
            # Enhanced batch processing with data shuffling
            epoch_losses = []
            indices = torch.randperm(n_train_samples)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_train_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Extract batch
                batch_mol = train_data['molecular'][batch_indices]
                batch_comp = train_data['composition'][batch_indices]
                batch_prop = train_data['property'][batch_indices]
                batch_targets = train_data['targets'][batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions, model_outputs = model.model(batch_mol, batch_comp, batch_prop)
                
                # 🔥 ENHANCED LOSS WITH STRONGER ENSEMBLE CONSISTENCY
                mse = mse_loss(predictions, batch_targets)
                huber = huber_loss(predictions, batch_targets)
                l1 = l1_loss(predictions, batch_targets)
                
                # Stronger ensemble consistency loss
                individual_preds = model_outputs.get('individual_predictions', [])
                consistency_loss = 0
                if len(individual_preds) >= 2:
                    consistency_loss = mse_loss(individual_preds[0], individual_preds[1])
                
                # Enhanced uncertainty regularization
                uncertainty_reg = torch.mean(model_outputs.get('uncertainty', torch.zeros(1).to(model.device)))
                
                # Feature importance diversity
                importance = model_outputs.get('feature_importance')
                if importance is not None:
                    importance_entropy = -torch.mean(torch.sum(importance * torch.log(importance + 1e-8), dim=1))
                else:
                    importance_entropy = torch.tensor(0.0).to(model.device)
                
                # 🔥 BREAKTHROUGH-FOCUSED LOSS WEIGHTING
                w1, w2, w3 = config.loss_weights
                total_loss = (w1 * mse + w2 * huber + w3 * l1 + 
                             0.15 * consistency_loss +      # Stronger consistency
                             0.08 * uncertainty_reg +       # More uncertainty reg
                             0.03 * importance_entropy)     # More diversity
                
                total_loss.backward()
                
                # ENHANCED: Stricter gradient clipping
                max_grad_norm = 0.5  # FIXED: Reduced from 1.0 for stability
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            train_losses.append(np.mean(epoch_losses))
            epoch_time = time.time() - epoch_start_time
            
            # Validation every 5 epochs for detailed monitoring
            val_loss = None  # Initialize for scheduler
            if epoch % 5 == 0:
                model.model.eval()
                with torch.no_grad():
                    val_predictions, val_outputs = model.model(
                        val_data['molecular'],
                        val_data['composition'],
                        val_data['property']
                    )
                    val_loss = mse_loss(val_predictions, val_data['targets'])
                    val_r2 = r2_score(
                        val_data['targets'].cpu().numpy(),
                        val_predictions.cpu().numpy()
                    )
                    
                    val_losses.append(val_loss.item())
                    
                    # 🔥 ENHANCED LOGGING
                    current_lr = optimizer.param_groups[0]['lr']
                    uncertainty_mean = val_outputs['uncertainty'].mean().item()
                    
                    print(f"    Epoch {epoch:3d}: Train={np.mean(epoch_losses):.4f}, "
                          f"Val R²={val_r2:.4f}, Loss={val_loss.item():.4f}, "
                          f"LR={current_lr:.2e}, Unc={uncertainty_mean:.3f}, "
                          f"Time={epoch_time:.2f}s")
                    
                    # Save best model with breakthrough detection
                    if val_r2 > best_val_r2 + 1e-5:
                        best_val_r2 = val_r2
                        patience_counter = 0
                        
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                            'val_r2': val_r2,
                            'val_loss': val_loss.item(),
                            'train_loss': np.mean(epoch_losses),
                            'config': {
                                'learning_rate': float(config.learning_rate),
                                'weight_decay': float(config.weight_decay),
                                'dropout_rate': float(config.dropout_rate),
                                'batch_size': int(config.batch_size),
                                'hidden_dims': list(config.hidden_dims),
                                'num_heads': int(config.num_heads),
                                'early_stopping_patience': int(config.early_stopping_patience),
                                'scheduler_type': str(config.scheduler_type),
                                'loss_weights': tuple(config.loss_weights)
                            },
                            'model_config': {
                                'molecular_feature_dim': int(model.config['molecular_feature_dim']),
                                'n_components': int(model.config['n_components']),
                                'property_dim': int(model.config['property_dim']),
                                'hidden_dims': list(model.config['hidden_dims']),
                                'num_heads': int(model.config['num_heads']),
                                'dropout_rate': float(model.config['dropout_rate'])
                            },
                            
                            '''
                            'config': config,
                            'model_config': model.config,
                            '''
                            'best_val_r2': best_val_r2
                        }
                        
                        model_path = trial_dir / f'best_model_trial_{trial_num}.pth'
                        torch.save(checkpoint, model_path)
                        
                        if val_r2 >= 0.5:
                            print(f"    🎉 BREAKTHROUGH! R²={val_r2:.4f} at epoch {epoch}")
                        else:
                            print(f"    💾 New best: R²={val_r2:.4f} at epoch {epoch}")
                            
                    else:
                        patience_counter += 1
                    
                    # Enhanced early stopping
                    if patience_counter >= config.early_stopping_patience:
                        print(f"    Early stopping at epoch {epoch} (patience={config.early_stopping_patience})")
                        break
            
            # 🔥 FIXED SCHEDULER STEP CALL - This is the key fix
            try:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs a metric value
                    if val_loss is not None:
                        scheduler.step(val_loss.item())
                    else:
                        # Use training loss if validation not computed this epoch
                        scheduler.step(np.mean(epoch_losses))
                else:
                    # Other schedulers (CosineAnnealingWarmRestarts, etc.) don't need metrics
                    scheduler.step()
            except Exception as scheduler_error:
                print(f"    ⚠️ Scheduler step failed: {scheduler_error}, continuing without scheduler update")
        
        return train_losses, val_losses, best_val_r2
    
    def run_trial(self, config: HyperparameterConfig, trial_num: int, output_dir: Path) -> Dict:
        """Run single hyperparameter trial with breakthrough focus."""
        
        print(f"\n🚀 BREAKTHROUGH TRIAL {trial_num}")
        print(f"  LR: {config.learning_rate:.0e}, WD: {config.weight_decay:.0e}, Dropout: {config.dropout_rate:.2f}")
        print(f"  Architecture: {config.hidden_dims}, Heads: {config.num_heads}, Batch: {config.batch_size}")
        print(f"  Scheduler: {config.scheduler_type}, Patience: {config.early_stopping_patience}")
        print(f"  Loss weights: {config.loss_weights}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create enhanced model
        model_config = {
            'molecular_feature_dim': self.data_config['molecular_feature_dim'],
            'n_components': self.data_config['n_components'],
            'property_dim': self.data_config['property_dim'],
            'hidden_dims': config.hidden_dims,
            'num_heads': config.num_heads,
            'dropout_rate': config.dropout_rate
        }
        
        model = TunedLNPRegressor(model_config, device=device)
        
        # Enhanced training
        start_time = time.time()
        train_losses, val_losses, best_val_r2 = self.train_with_config(
            model, self.data_config['train_data'], self.data_config['val_data'], 
            config, output_dir, trial_num
        )
        training_time = time.time() - start_time
        
        # Final evaluation
        model.model.eval()
        with torch.no_grad():
            final_predictions, final_outputs = model.model(
                self.data_config['val_data']['molecular'],
                self.data_config['val_data']['composition'],
                self.data_config['val_data']['property']
            )
            
            targets_np = self.data_config['val_data']['targets'].cpu().numpy()
            preds_np = final_predictions.cpu().numpy()
            
            final_r2 = r2_score(targets_np, preds_np)
            final_rmse = np.sqrt(mean_squared_error(targets_np, preds_np))
            final_mae = mean_absolute_error(targets_np, preds_np)
            
            # Enhanced metrics
            final_uncertainty = final_outputs['uncertainty'].mean().item()
            prediction_variance = final_outputs['uncertainty'].var().item()
        
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
            'final_uncertainty': final_uncertainty,
            'prediction_variance': prediction_variance,
            'breakthrough_achieved': final_r2 >= 0.5,
            'model_path': str(trial_model_path)
        }
        
        if final_r2 >= 0.5:
            print(f"  🎉 BREAKTHROUGH ACHIEVED! R²={final_r2:.4f}")
        elif final_r2 >= 0.45:
            print(f"  🔥 CLOSE TO BREAKTHROUGH! R²={final_r2:.4f}")
        else:
            print(f"  📈 R²={final_r2:.4f}, RMSE={final_rmse:.4f}")
        
        print(f"     Time: {training_time/60:.1f}min, Epochs: {len(train_losses)}")
        
        return result
    
    def run_comprehensive_search(self, output_dir: Path):
        """Run breakthrough-focused hyperparameter search."""
        
        configs = self.create_breakthrough_search_space()
        
        print(f"\n🎯 BREAKTHROUGH-FOCUSED HYPERPARAMETER SEARCH")
        print(f"Mission: Achieve consistent R² > 0.5 with enhanced architecture")
        print(f"Configurations: {len(configs)} breakthrough-optimized setups")
        print("="*80)
        
        breakthrough_configs = []
        
        for i, config in enumerate(configs, 1):
            try:
                result = self.run_trial(config, i, output_dir)
                self.results.append(result)
                
                if result['breakthrough_achieved']:
                    breakthrough_configs.append(result)
                
                if result['final_r2'] > self.best_score:
                    self.best_score = result['final_r2']
                    self.best_config = config
                    self.best_model_path = result['model_path']
                    
                    best_model_dest = output_dir / 'BEST_OVERALL_MODEL.pth'
                    shutil.copy2(result['model_path'], best_model_dest)
                    print(f"  🏆 NEW OVERALL BEST! R²={self.best_score:.4f}")
                    
            except Exception as e:
                print(f"  ❌ Trial {i} failed: {e}")
                continue
        
        return self.analyze_results(breakthrough_configs)
    
    def analyze_results(self, breakthrough_configs):
        """Analyze breakthrough results."""
        
        if not self.results:
            return {"error": "No successful trials"}
        
        sorted_results = sorted(self.results, key=lambda x: x['final_r2'], reverse=True)
        
        return {
            'best_r2': sorted_results[0]['final_r2'],
            'best_config': sorted_results[0]['config'],
            'best_model_path': self.best_model_path,
            'breakthrough_achieved': len(breakthrough_configs) > 0,
            'breakthrough_configs': breakthrough_configs,
            'breakthrough_rate': len(breakthrough_configs) / len(self.results),
            'top_10_results': sorted_results[:10],
            'avg_r2': np.mean([r['final_r2'] for r in self.results]),
            'std_r2': np.std([r['final_r2'] for r in self.results]),
            'median_r2': np.median([r['final_r2'] for r in self.results]),
            'num_trials': len(self.results),
            'avg_training_time': np.mean([r['training_time'] for r in self.results]),
            'avg_uncertainty': np.mean([r.get('final_uncertainty', 0) for r in self.results])
        }
    
    def save_results(self, output_dir: Path, analysis: Dict):
        """Save enhanced results."""
        
        serializable_analysis = analysis.copy()
        
        if 'best_model_path' in analysis:
            serializable_analysis['best_model_path'] = str(analysis['best_model_path']) if analysis['best_model_path'] else None
        
        if 'best_config' in analysis and analysis['best_config']:
            best_config = analysis['best_config']
            serializable_analysis['best_config'] = {
                'learning_rate': float(best_config.learning_rate),
                'weight_decay': float(best_config.weight_decay),
                'dropout_rate': float(best_config.dropout_rate),
                'batch_size': int(best_config.batch_size),
                'hidden_dims': best_config.hidden_dims,
                'num_heads': int(best_config.num_heads),
                'early_stopping_patience': int(best_config.early_stopping_patience),
                'scheduler_type': str(best_config.scheduler_type),
                'loss_weights': best_config.loss_weights
            }
        
        with open(output_dir / 'breakthrough_results.json', 'w') as f:
            json.dump(serializable_analysis, f, indent=2, default=str)
