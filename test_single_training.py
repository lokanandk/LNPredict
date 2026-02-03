#!/usr/bin/env python3
"""
Enhanced Single Configuration Test for Breakthrough Performance
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import LNPDataLoader
from models.regressor import TunedLNPRegressor
from utils.model_utils import setup_device
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import numpy as np

def augment_training_data(molecular_features, composition_features, property_features, targets, augment_factor=2):
    """Generate augmented training data using controlled noise injection."""
    
    print(f"🔄 Augmenting data with factor {augment_factor}...")
    
    augmented_mol = [molecular_features]
    augmented_comp = [composition_features]
    augmented_prop = [property_features]
    augmented_targets = [targets]
    
    for i in range(augment_factor):
        # Add controlled gaussian noise
        noise_scale_mol = 0.03 if i == 0 else 0.05  # Less noise for first augmentation
        noise_scale_comp = 0.015 if i == 0 else 0.025
        noise_scale_prop = 0.02 if i == 0 else 0.04
        
        noise_mol = molecular_features + np.random.normal(0, noise_scale_mol, molecular_features.shape)
        noise_comp = composition_features + np.random.normal(0, noise_scale_comp, composition_features.shape)
        noise_prop = property_features + np.random.normal(0, noise_scale_prop, property_features.shape)
        
        # Ensure compositions still sum to 1
        noise_comp = np.abs(noise_comp)  # Ensure positive
        noise_comp = noise_comp / noise_comp.sum(axis=1, keepdims=True)
        
        # Clip molecular features to reasonable bounds
        noise_mol = np.clip(noise_mol, -3, 3)
        noise_prop = np.clip(noise_prop, -3, 3)
        
        augmented_mol.append(noise_mol)
        augmented_comp.append(noise_comp)
        augmented_prop.append(noise_prop)
        augmented_targets.append(targets)
    
    # Combine all data
    all_mol = np.vstack(augmented_mol)
    all_comp = np.vstack(augmented_comp)
    all_prop = np.vstack(augmented_prop)
    all_targets = np.hstack(augmented_targets)
    
    print(f"✅ Data augmented: {len(targets)} → {len(all_targets)} samples")
    
    return all_mol, all_comp, all_prop, all_targets

def test_breakthrough_configuration():
    """Test breakthrough-focused configuration with data augmentation."""
    
    # 🔥 BREAKTHROUGH-OPTIMIZED CONFIGURATION
    config = {
        'learning_rate': 5e-3,        # Higher learning rate
        'weight_decay': 3e-4,         # Stronger regularization
        'dropout_rate': 0.3,          # Higher dropout
        'batch_size': 64,             # Larger batch for batch norm stability
        'hidden_dims': [256, 128, 64], # Deeper architecture
        'num_heads': 8,               # More attention heads
        'epochs': 400,                # More epochs
        'patience': 100,              # Higher patience
        'augment_data': True          # Data augmentation
    }
    
    print("🎯 === BREAKTHROUGH CONFIGURATION TEST ===")
    print(f"Config: {config}")
    print("🎯 Target: R² > 0.5 (Breakthrough threshold)")
    
    # Load and augment data
    device = setup_device()
    data_loader = LNPDataLoader()
    
    # Load base data with proper config
    data_config = data_loader.load_and_prepare(
        'data/lnp_data.json', 
        config={'test_size': 0.15, 'random_state': 42, 'stratify_bins': 5}, 
        device='cpu'  # Keep on CPU for augmentation
    )
    
    print(f"Base data: Train={len(data_config['train_data']['targets'])}, Val={len(data_config['val_data']['targets'])}")
    
    # Data augmentation
    if config['augment_data']:
        train_mol = data_config['train_data']['molecular'].cpu().numpy()
        train_comp = data_config['train_data']['composition'].cpu().numpy()
        train_prop = data_config['train_data']['property'].cpu().numpy()
        train_targets = data_config['train_data']['targets'].cpu().numpy()
        
        # Augment training data
        aug_mol, aug_comp, aug_prop, aug_targets = augment_training_data(
            train_mol, train_comp, train_prop, train_targets, augment_factor=2
        )
        
        # Update training data
        data_config['train_data'] = {
            'molecular': torch.FloatTensor(aug_mol).to(device),
            'composition': torch.FloatTensor(aug_comp).to(device),
            'property': torch.FloatTensor(aug_prop).to(device),
            'targets': torch.FloatTensor(aug_targets).to(device)
        }
        
        # Move validation data to device
        for key in data_config['val_data']:
            data_config['val_data'][key] = data_config['val_data'][key].to(device)
    
    print(f"Final data: Train={len(data_config['train_data']['targets'])}, Val={len(data_config['val_data']['targets'])}")
    
    # Create enhanced model
    model_config = {
        'molecular_feature_dim': data_config['molecular_feature_dim'],
        'n_components': data_config['n_components'],
        'property_dim': data_config['property_dim'],
        'hidden_dims': config['hidden_dims'],
        'num_heads': config['num_heads'],
        'dropout_rate': config['dropout_rate']
    }
    
    model = TunedLNPRegressor(model_config, device=device)
    
    # 🔥 ADAMW OPTIMIZER WITH PROPER CONFIGURATION
    optimizer = torch.optim.AdamW(
        model.model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True
    )
    
    # 🔥 ONECYCLELR SCHEDULER FOR BREAKTHROUGH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=config['epochs'],
        pct_start=0.1,        # 10% warmup
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # Enhanced loss functions
    mse_loss = nn.MSELoss()
    huber_loss = nn.HuberLoss(delta=1.5)
    l1_loss = nn.L1Loss()
    
    # Training setup
    batch_size = config['batch_size']
    train_data = data_config['train_data']
    val_data = data_config['val_data']
    n_train_samples = len(train_data['targets'])
    n_batches = max(1, n_train_samples // batch_size)
    
    print(f"🔍 Enhanced setup: Batch size={batch_size}, Samples={n_train_samples}, Batches={n_batches}")
    
    # Training with breakthrough monitoring
    best_val_r2 = -np.inf
    patience_counter = 0
    breakthrough_epoch = None
    
    print("\n🚀 Starting breakthrough-focused training...")
    print("Target: R² > 0.5 for breakthrough achievement\n")
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        model.model.train()
        
        # Enhanced batch processing
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
            
            # Enhanced loss calculation
            mse = mse_loss(predictions, batch_targets)
            huber = huber_loss(predictions, batch_targets)
            l1 = l1_loss(predictions, batch_targets)
            
            # Stronger ensemble consistency
            consistency_loss = 0
            individual_preds = model_outputs.get('individual_predictions', [])
            if len(individual_preds) >= 2:
                consistency_loss = mse_loss(individual_preds[0], individual_preds[1])
            
            # Enhanced uncertainty regularization
            uncertainty_reg = torch.mean(model_outputs.get('uncertainty', torch.zeros(1).to(device)))
            
            # Feature importance diversity
            importance = model_outputs.get('feature_importance')
            if importance is not None:
                importance_entropy = -torch.mean(torch.sum(importance * torch.log(importance + 1e-8), dim=1))
            else:
                importance_entropy = torch.tensor(0.0).to(device)
            
            # Breakthrough-focused loss weighting
            total_loss = (0.7 * mse + 0.2 * huber + 0.1 * l1 + 
                         0.15 * consistency_loss + 
                         0.08 * uncertainty_reg + 
                         0.03 * importance_entropy)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=2.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        
        # Detailed validation every 5 epochs
        if epoch % 5 == 0 or epoch < 10:
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
                val_rmse = np.sqrt(mean_squared_error(
                    val_data['targets'].cpu().numpy(),
                    val_predictions.cpu().numpy()
                ))
                
                # Enhanced metrics
                val_uncertainty = val_outputs['uncertainty'].mean().item()
                current_lr = optimizer.param_groups[0]['lr']
            
            # Enhanced progress reporting
            breakthrough_status = ""
            if val_r2 >= 0.5 and breakthrough_epoch is None:
                breakthrough_epoch = epoch
                breakthrough_status = " 🎉 BREAKTHROUGH ACHIEVED!"
            elif val_r2 >= 0.45:
                breakthrough_status = " 🔥 CLOSE TO BREAKTHROUGH!"
            
            print(f"Epoch {epoch:3d}: Train={np.mean(epoch_losses):.4f}, "
                  f"Val R²={val_r2:.4f}, RMSE={val_rmse:.4f}, "
                  f"Loss={val_loss.item():.4f}, LR={current_lr:.2e}, "
                  f"Unc={val_uncertainty:.3f}, Time={epoch_time:.2f}s{breakthrough_status}")
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
                print(f"    💾 New best R²: {val_r2:.4f}")
            else:
                patience_counter += 1
            
            # Enhanced early stopping with breakthrough consideration
            if patience_counter >= config['patience']:
                if best_val_r2 >= 0.5:
                    print(f"    ✅ Early stopping with breakthrough achieved!")
                else:
                    print(f"    ⏹️  Early stopping at epoch {epoch} (patience={config['patience']})")
                break
    
    # Final comprehensive evaluation
    print(f"\n{'='*60}")
    print("🎯 BREAKTHROUGH TEST RESULTS")
    print(f"{'='*60}")
    
    model.model.eval()
    with torch.no_grad():
        final_val_preds, final_outputs = model.model(
            val_data['molecular'], 
            val_data['composition'], 
            val_data['property']
        )
        final_train_preds, _ = model.model(
            train_data['molecular'][:len(val_data['targets'])],  # Sample for comparison
            train_data['composition'][:len(val_data['targets'])],
            train_data['property'][:len(val_data['targets'])]
        )
        
        val_targets_np = val_data['targets'].cpu().numpy()
        val_preds_np = final_val_preds.cpu().numpy()
        
        final_val_r2 = r2_score(val_targets_np, val_preds_np)
        final_val_rmse = np.sqrt(mean_squared_error(val_targets_np, val_preds_np))
        final_val_mae = mean_absolute_error(val_targets_np, val_preds_np)
        
        # Enhanced analytics
        prediction_std = np.std(val_preds_np)
        target_std = np.std(val_targets_np)
        uncertainty_mean = final_outputs['uncertainty'].mean().item()
        uncertainty_std = final_outputs['uncertainty'].std().item()
    
    print(f"🎯 Best Validation R²: {best_val_r2:.4f}")
    print(f"📊 Final Validation Metrics:")
    print(f"   R²: {final_val_r2:.4f}")
    print(f"   RMSE: {final_val_rmse:.4f}")
    print(f"   MAE: {final_val_mae:.4f}")
    print(f"   Prediction Std: {prediction_std:.4f}")
    print(f"   Target Std: {target_std:.4f}")
    print(f"   Avg Uncertainty: {uncertainty_mean:.4f} ± {uncertainty_std:.4f}")
    
    if breakthrough_epoch is not None:
        print(f"🎉 BREAKTHROUGH achieved at epoch {breakthrough_epoch}!")
    
    # Model output verification
    print(f"\n🔍 Model Architecture Verification:")
    print(f"   Individual predictions: {'individual_predictions' in final_outputs}")
    print(f"   Uncertainty estimation: {'uncertainty' in final_outputs}")
    print(f"   Feature importance: {'feature_importance' in final_outputs}")
    print(f"   Attention weights: {'attention_weights' in final_outputs}")
    
    # Performance assessment
    if best_val_r2 >= 0.5:
        print(f"\n🏆 SUCCESS! Breakthrough achieved with R² = {best_val_r2:.4f}")
        print("✅ Model ready for hyperparameter search with these settings!")
    elif best_val_r2 >= 0.45:
        print(f"\n📈 VERY CLOSE! R² = {best_val_r2:.4f} (need +{0.5-best_val_r2:.3f})")
        print("🔧 Try: Higher learning rate, more epochs, or architecture tweaks")
    elif best_val_r2 >= 0.35:
        print(f"\n📊 GOOD PROGRESS! R² = {best_val_r2:.4f} (need +{0.5-best_val_r2:.3f})")
        print("🔧 Recommendations: Increase model capacity or training duration")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT: R² = {best_val_r2:.4f}")
        print("🔧 Check: Data quality, preprocessing, or try different architecture")
    
    return best_val_r2, breakthrough_epoch is not None

if __name__ == "__main__":
    best_r2, achieved_breakthrough = test_breakthrough_configuration()
    
    if achieved_breakthrough:
        print(f"\n🎯 Ready for full hyperparameter search!")
    else:
        print(f"\n🔧 Optimize single config first before full search")
