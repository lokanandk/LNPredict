"""
Single Model Trainer for LNP Models
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json
import time

from src.models.regressor import TunedLNPRegressor
from src.utils.model_utils import save_model_checkpoint


class LNPTrainer:
    """Single model trainer with comprehensive logging."""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.training_history = []
        self.best_model_path = None
        self.best_score = -np.inf
    
    def train(self, data_config: Dict, output_dir: Path) -> Dict:
        """Train a single model with given configuration."""
        
        print(f"🚀 Starting model training...")
        print(f"Configuration: {self.config['training']}")
        
        # Create model
        model_config = {
            'molecular_feature_dim': data_config['molecular_feature_dim'],
            'n_components': data_config['n_components'],
            'property_dim': data_config['property_dim'],
            'hidden_dims': self.config['model']['hidden_dims'],
            'num_heads': self.config['model']['num_heads'],
            'dropout_rate': self.config['model']['dropout_rate']
        }
        
        model = TunedLNPRegressor(model_config, device=self.device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            eps=1e-8,
            amsgrad=True
        )
        
        # Setup scheduler
        scheduler = self._create_scheduler(optimizer)
        
        # Loss functions
        mse_loss = nn.MSELoss()
        huber_loss = nn.HuberLoss(delta=1.5)
        l1_loss = nn.L1Loss()
        
        # Training variables
        train_data = data_config['train_data']
        val_data = data_config['val_data']
        
        train_losses = []
        val_losses = []
        val_r2_scores = []
        
        best_val_r2 = -np.inf
        patience_counter = 0
        
        start_time = time.time()
        
        # Training loop
        epochs = self.config['training']['epochs']
        for epoch in range(epochs):
            # Training step
            model.model.train()
            optimizer.zero_grad()
            
            predictions, model_outputs = model.model(
                train_data['molecular'],
                train_data['composition'],
                train_data['property']
            )
            
            # Multi-component loss
            w1, w2, w3 = self.config['training']['loss_weights']
            mse = mse_loss(predictions, train_data['targets'])
            huber = huber_loss(predictions, train_data['targets'])
            l1 = l1_loss(predictions, train_data['targets'])
            
            # Additional losses
            consistency_loss = 0
            individual_preds = model_outputs.get('individual_predictions', [])
            if len(individual_preds) >= 2:
                consistency_loss = mse_loss(individual_preds[0], individual_preds[1])
            
            uncertainty_reg = torch.mean(model_outputs.get('uncertainty', torch.zeros(1).to(self.device)))
            
            total_loss = (w1 * mse + w2 * huber + w3 * l1 + 
                         0.1 * consistency_loss + 0.05 * uncertainty_reg)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if hasattr(scheduler, 'step') and self.config['training']['scheduler_type'] != 'plateau':
                scheduler.step()
            
            train_losses.append(total_loss.item())
            
            # Validation step
            if epoch % 5 == 0:
                model.model.eval()
                with torch.no_grad():
                    val_predictions, _ = model.model(
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
                    val_r2_scores.append(val_r2)
                    
                    # Update scheduler if plateau
                    if self.config['training']['scheduler_type'] == 'plateau':
                        scheduler.step(val_loss)
                    
                    # Save best model
                    if val_r2 > best_val_r2:
                        best_val_r2 = val_r2
                        patience_counter = 0
                        
                        checkpoint_path = output_dir / 'best_model.pth'
                        save_model_checkpoint(
                            model=model.model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            val_r2=val_r2,
                            val_loss=val_loss.item(),
                            config=self.config,
                            model_config=model_config,
                            filepath=checkpoint_path
                        )
                        self.best_model_path = checkpoint_path
                        self.best_score = val_r2
                        
                        print(f"Epoch {epoch}: New best R² = {val_r2:.4f}")
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= self.config['training']['early_stopping_patience'] // 5:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        training_time = time.time() - start_time
        
        # Prepare results
        results = {
            'best_val_r2': best_val_r2,
            'final_epoch': epoch,
            'training_time': training_time,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'best_model_path': str(self.best_model_path)
        }
        
        # Save training history
        self._save_training_history(results, output_dir)
        
        return results
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        sched_type = self.config['training']['scheduler_type']
        
        if sched_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2, eta_min=self.config['training']['learning_rate'] * 0.001
            )
        elif sched_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.3, patience=15, min_lr=1e-8
            )
        elif sched_type == 'step':
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100, 200, 400], gamma=0.3
            )
        elif sched_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        else:  # linear
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=500
            )
    
    def _save_training_history(self, results: Dict, output_dir: Path):
        """Save training history and generate plots."""
        
        # Save training history JSON
        with open(output_dir / 'training_history.json', 'w') as f:
            # Make numpy arrays JSON serializable
            serializable_results = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                  for k, v in results.items() if k != 'best_model_path'}
            json.dump(serializable_results, f, indent=2)
        
        # Generate training plots
        self._plot_training_curves(results, output_dir)
    
    def _plot_training_curves(self, results: Dict, output_dir: Path):
        """Generate training curve plots."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        ax1.plot(results['train_losses'], label='Training Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation loss
        val_epochs = list(range(0, len(results['val_losses']) * 5, 5))
        ax2.plot(val_epochs, results['val_losses'], label='Validation Loss', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Validation R²
        ax3.plot(val_epochs, results['val_r2_scores'], label='Validation R²', color='green')
        ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='R² = 0.5')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('R² Score')
        ax3.set_title('Validation R² Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Combined view
        ax4_twin = ax4.twinx()
        ax4.plot(results['train_losses'], label='Train Loss', alpha=0.7)
        ax4.plot(val_epochs, results['val_losses'], label='Val Loss', color='orange')
        ax4_twin.plot(val_epochs, results['val_r2_scores'], label='Val R²', color='green')
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4_twin.set_ylabel('R² Score')
        ax4.set_title('Training Overview')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_training_summary(self, results: Dict):
        """Print comprehensive training summary."""
        
        print("\n" + "="*60)
        print("🚀 === TRAINING SUMMARY ===")
        print("="*60)
        
        print(f"🎯 Final Results:")
        print(f"   Best Validation R²: {results['best_val_r2']:.4f}")
        print(f"   Training completed in: {results['training_time']/60:.1f} minutes")
        print(f"   Total epochs: {results['final_epoch']}")
        print(f"   Best model saved: {results['best_model_path']}")
        
        # Performance assessment
        r2 = results['best_val_r2']
        if r2 >= 0.7:
            print(f"\n🎉 EXCELLENT performance! Model achieved breakthrough results.")
        elif r2 >= 0.5:
            print(f"\n✅ GOOD performance! Model shows strong predictive capability.")
        elif r2 >= 0.3:
            print(f"\n⚠️  MODERATE performance. Consider hyperparameter tuning.")
        else:
            print(f"\n❌ POOR performance. Model needs significant improvements.")
        
        print(f"\n📊 Training Configuration:")
        train_config = self.config['training']
        print(f"   Learning Rate: {train_config['learning_rate']:.0e}")
        print(f"   Weight Decay: {train_config['weight_decay']:.0e}")
        print(f"   Batch Size: {train_config.get('batch_size', 'N/A')}")
        print(f"   Scheduler: {train_config['scheduler_type']}")
