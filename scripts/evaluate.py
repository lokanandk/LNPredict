#!/usr/bin/env python3
"""
Model Evaluation Script
"""

import argparse
import yaml
from pathlib import Path
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple

# Add project root to path BEFORE imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent / 'src'))
# Now use absolute imports
from src.evaluation.evaluator import LNPEvaluator
from src.data.data_loader import LNPDataLoader
from src.utils.model_utils import setup_device

# ADD THIS CLASS DEFINITION - exactly like in your standalone code
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

def main():
    parser = argparse.ArgumentParser(description='Evaluate LNP Model')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Path to test data')
    parser.add_argument('--output', default='outputs/evaluation', help='Output directory')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    parser.add_argument('--generate_plots', action='store_true', help='Generate evaluation plots')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("📊 === LNP MODEL EVALUATION ===")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = setup_device()
    
    # Load data
    data_loader = LNPDataLoader()
    data_config = data_loader.load_and_prepare(
        args.data, 
        config=config['data'],
        device=device
    )
    
    # Create evaluator
    evaluator = LNPEvaluator(args.model_path, config['evaluation'])
    
    # Comprehensive evaluation
    results = evaluator.comprehensive_evaluation(
        data_config,
        output_dir,
        generate_plots=args.generate_plots or config['evaluation']['generate_plots']
    )
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
