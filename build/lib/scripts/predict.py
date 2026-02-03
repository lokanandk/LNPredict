#!/usr/bin/env python3
"""
Prediction Script for New Samples
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.model_utils import predict_with_saved_model
from data.data_loader import LNPDataLoader


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained LNP model')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--input_data', required=True, help='Path to input data (JSON format)')
    parser.add_argument('--output', default='predictions.csv', help='Output predictions file')
    parser.add_argument('--include_uncertainty', action='store_true', help='Include uncertainty estimates')
    args = parser.parse_args()
    
    print("🔮 === LNP MODEL PREDICTIONS ===")
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_data}")
    print(f"Output: {args.output}")
    
    # Load data
    data_loader = LNPDataLoader()
    
    # Load and prepare input data (without targets)
    try:
        features, _, feature_names = data_loader.featurizer.create_comprehensive_features(args.input_data)
        
        # Separate feature types
        mol_mask = [name.startswith('mol_') for name in feature_names]
        comp_mask = [name.startswith('comp_') for name in feature_names]
        prop_mask = [name.startswith('prop_') for name in feature_names]
        
        molecular_features = features[:, mol_mask]
        composition_features = features[:, comp_mask]
        property_features = features[:, prop_mask]
        
        # Scale property features using the same scaler (if available)
        # Note: In production, you'd want to save and load the scaler from training
        property_features = data_loader.scaler.fit_transform(property_features)
        
        # Prepare molecular features (same as training)
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
        
    except Exception as e:
        print(f"❌ Error loading input data: {e}")
        return
    
    # Make predictions
    try:
        print(f"Making predictions for {len(molecular_features)} samples...")
        
        predictions, model_outputs = predict_with_saved_model(
            args.model_path,
            molecular_features,
            composition_features,
            property_features
        )
        
        # Prepare results DataFrame
        results_df = pd.DataFrame({
            'predicted_target': predictions
        })
        
        # Add uncertainty if available and requested
        if args.include_uncertainty and 'uncertainty' in model_outputs:
            uncertainties = model_outputs['uncertainty'].cpu().numpy()
            results_df['uncertainty'] = uncertainties
            results_df['confidence'] = 1 - uncertainties  # Simple confidence measure
        
        # Add feature importance if available
        if 'feature_importance' in model_outputs:
            importance = model_outputs['feature_importance'].cpu().numpy()
            for i in range(importance.shape[1]):
                results_df[f'component_{i}_importance'] = importance[:, i]
        
        # Add sample IDs if available
        try:
            import json
            with open(args.input_data, 'r') as f:
                data = json.load(f)
            
            if 'formulations' in data:
                sample_ids = [f['id'] for f in data['formulations']]
                results_df.insert(0, 'sample_id', sample_ids[:len(predictions)])
        except:
            # Add generic IDs
            results_df.insert(0, 'sample_id', [f'sample_{i+1}' for i in range(len(predictions))])
        
        # Save results
        results_df.to_csv(args.output, index=False)
        
        # Print summary
        print(f"\n✅ Predictions completed!")
        print(f"📊 Summary:")
        print(f"   Samples processed: {len(predictions)}")
        print(f"   Mean prediction: {np.mean(predictions):.2f}")
        print(f"   Std prediction: {np.std(predictions):.2f}")
        print(f"   Min prediction: {np.min(predictions):.2f}")
        print(f"   Max prediction: {np.max(predictions):.2f}")
        
        if args.include_uncertainty and 'uncertainty' in results_df.columns:
            uncertainties = results_df['uncertainty'].values
            print(f"   Mean uncertainty: {np.mean(uncertainties):.3f}")
            print(f"   High confidence samples (unc < 0.1): {np.sum(uncertainties < 0.1)}")
        
        print(f"📁 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
