#!/usr/bin/env python3
"""
LNPredict Synthetic Data Generation Script

Generate high-quality synthetic LNP formulations using targeted augmentation
strategies optimized for small datasets.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.data_generation import TargetedAugmentationGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have created all the required module files.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic LNP data using targeted augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_synthetic.py --input data/lnp_data.json --n_samples 1000
  python scripts/generate_synthetic.py --input data/lnp_data.json --n_samples 5000 --output data/synthetic_large.json
  python scripts/generate_synthetic.py --input data/lnp_data.json --n_samples 500 --random_state 123
        """
    )
    
    parser.add_argument(
        "--input", "-i", 
        default="data/lnp_data.json",
        help="Input LNP dataset (JSON format)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/synthetic_lnp_data.json", 
        help="Output synthetic dataset path"
    )
    parser.add_argument(
        "--n_samples", "-n",
        type=int, 
        default=1000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--random_state", "-r",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--strategy_weights",
        type=str,
        help="Strategy weights as comma-separated values (interpolation,nn,perturbation). Example: 0.5,0.3,0.2"
    )
    
    args = parser.parse_args()
    
    # Parse strategy weights if provided
    strategy_weights = None
    if args.strategy_weights:
        try:
            weights = [float(w) for w in args.strategy_weights.split(',')]
            if len(weights) != 3:
                raise ValueError("Must provide exactly 3 weights")
            strategy_weights = {
                'interpolation': weights[0],
                'nn': weights[1], 
                'perturbation': weights[2]
            }
        except ValueError as e:
            print(f"Error parsing strategy weights: {e}")
            return 1
    
    print("="*60)
    print("🧬 LNPredict Synthetic Data Generation")
    print("="*60)
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"🎯 Samples: {args.n_samples:,}")
    print(f"🎲 Random state: {args.random_state}")
    if strategy_weights:
        print(f"⚖️  Strategy weights: {strategy_weights}")
    print("="*60)
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize generator
        generator = TargetedAugmentationGenerator(random_state=args.random_state)
        
        # Load original data
        print(f"📊 Loading original data from {args.input}...")
        original_df = generator.load_data(args.input)
        print(f"✅ Loaded {len(original_df)} original formulations")
        
        # Generate synthetic data
        print(f"\n🔬 Generating {args.n_samples:,} synthetic samples...")
        synthetic_df = generator.generate_synthetic_dataset(
            original_df, 
            args.n_samples,
            strategy_weights=strategy_weights
        )
        
        print(f"✅ Generated {len(synthetic_df):,} synthetic formulations")
        
        # Assess quality
        print(f"\n📈 Assessing synthetic data quality...")
        quality_report = generator.assess_quality(original_df, synthetic_df)
        
        # Save results
        print(f"\n💾 Saving results to {args.output}...")
        generator.save_synthetic_data(
            synthetic_df, args.input, args.output, quality_report
        )
        
        # Print final summary
        print(f"\n🎯 Generation Summary:")
        print(f"  Original samples: {len(original_df):,}")
        print(f"  Synthetic samples: {len(synthetic_df):,}")
        print(f"  Quality score: {quality_report['overall_quality']:.3f}")
        
        # Quality assessment
        quality = quality_report['overall_quality']
        if quality > 0.7:
            print("🎉 EXCELLENT quality synthetic data!")
            print("   Ready for machine learning training")
        elif quality > 0.5:
            print("✅ GOOD quality synthetic data!")
            print("   Suitable for augmenting your training set")
        elif quality > 0.3:
            print("📈 DECENT quality synthetic data!")
            print("   Consider parameter tuning for improvement")
        else:
            print("⚠️  MODERATE quality synthetic data!")
            print("   May need strategy adjustment or more original data")
        
        # Specific recommendations
        if quality_report['range_coverage'] < 0.8:
            print("💡 Tip: Increase perturbation scales for better feature coverage")
        if quality_report['diversity_score'] < 0.8:
            print("💡 Tip: Reduce similarity thresholds for more diversity")
        if quality_report['correlation_preservation'] < 0.5:
            print("💡 Tip: Check feature relationships in original data")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
