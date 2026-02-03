#!/usr/bin/env python3
"""
Single Model Training Script
"""

import argparse
import yaml
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.trainer import LNPTrainer
from data.data_loader import LNPDataLoader
from utils.model_utils import setup_device


def main():
    parser = argparse.ArgumentParser(description='Train LNP Model')
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--output', default='outputs/single_model', help='Output directory')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.dropout:
        config['model']['dropout_rate'] = args.dropout
    
    print("🚀 === LNP MODEL TRAINING ===")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    
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
    
    # Create trainer
    trainer = LNPTrainer(config, device)
    
    # Train model
    results = trainer.train(data_config, output_dir)
    
    # Print results
    trainer.print_training_summary(results)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
