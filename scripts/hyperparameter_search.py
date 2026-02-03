#!/usr/bin/env python3
"""
Hyperparameter Search Script
"""
import argparse
import yaml
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from training.hyperparameter_tuner import ComprehensiveHyperparameterTuner
from data.data_loader import LNPDataLoader
from utils.model_utils import setup_device

def safe_float_format(value, fmt='.4f', default='N/A'):
    """Safely format numeric values, handling strings and None values."""
    try:
        return format(float(value), fmt)
    except (ValueError, TypeError, AttributeError):
        return default

def main():
    parser = argparse.ArgumentParser(description='LNP Hyperparameter Search')
    parser.add_argument('--data', required=True, help='Path to original JSON data')
    parser.add_argument('--synthetic', help='Path to synthetic JSON data')
    parser.add_argument('--output', default='outputs/hyperparam_search', help='Output directory')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    parser.add_argument('--max_trials', type=int, help='Maximum number of trials')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle config key consistency - support both naming conventions
    if args.max_trials:
        if 'hyperparameter_tuner' in config:
            config['hyperparameter_tuner']['max_trials'] = args.max_trials
        elif 'hyperparameter_search' in config:
            config['hyperparameter_search']['max_trials'] = args.max_trials
        else:
            # Create the config section if it doesn't exist
            config['hyperparameter_tuner'] = {'max_trials': args.max_trials}
    
    print("🎯 === LNP HYPERPARAMETER SEARCH ===")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    
    # Safe access to max_trials
    max_trials = (config.get('hyperparameter_tuner', {}).get('max_trials') or 
                  config.get('hyperparameter_search', {}).get('max_trials') or 
                  'N/A')
    print(f"Max trials: {max_trials}")
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = setup_device()
    
    try:
        # Load data
        data_loader = LNPDataLoader()
        data_config = data_loader.load_and_prepare(
            args.data, 
            synthetic_data=args.synthetic,
            config=config.get('data', {}),
            device=device
        )
        
        # Use the correct config section for tuner
        tuner_config = (config.get('hyperparameter_tuner', {}) or 
                       config.get('hyperparameter_search', {}))
        
        tuner = ComprehensiveHyperparameterTuner(data_config, tuner_config)
        analysis = tuner.run_comprehensive_search(output_dir)
        
        # Ensure analysis is valid before saving
        if not isinstance(analysis, dict):
            analysis = {
                'error': 'Search failed to return valid results',
                'best_r2': 0.0,
                'best_model_path': None,
                'num_trials': 0
            }
        
        # Safe handling of model path
        if 'best_model_path' in analysis and analysis['best_model_path']:
            analysis['best_model_path'] = str(analysis['best_model_path'])
        else:
            analysis['best_model_path'] = None
        
        # Save results
        tuner.save_results(output_dir, analysis)
        
        print(f"\n✅ Hyperparameter search completed!")
        print(f"📁 Results saved to: {output_dir}")
        
        # Safe result reporting
        best_r2 = analysis.get('best_r2')
        if best_r2 is not None:
            print(f"🎯 Best R²: {safe_float_format(best_r2, '.4f')}")
        else:
            print("🎯 Best R²: No successful trials")
            
        num_trials = analysis.get('num_trials', 0)
        print(f"📊 Completed trials: {num_trials}")
        
    except Exception as e:
        print(f"❌ Error during hyperparameter search: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_analysis = {
            'error': str(e),
            'best_r2': 0.0,
            'best_config': None,
            'best_model_path': None,
            'num_trials': 0,
            'avg_r2': 0.0
        }
        
        try:
            # Try to save error info
            import json
            with open(output_dir / 'error_results.json', 'w') as f:
                json.dump(error_analysis, f, indent=2)
        except:
            pass
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
