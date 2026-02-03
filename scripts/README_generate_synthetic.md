# LNPredict Synthetic Data Generation

Generate high-quality synthetic LNP (Lipid Nanoparticle) formulations using intelligent augmentation strategies optimized for small datasets.

## 🎯 Overview

The LNPredict synthetic data generation module uses **targeted augmentation** rather than full synthetic generation, making it ideal for small datasets typical in pharmaceutical research. It employs three complementary strategies:

1. **Composition Interpolation** (40%): Interpolates between similar formulations
2. **Nearest Neighbor Augmentation** (30%): Creates weighted combinations of similar samples  
3. **Perturbation Generation** (30%): Applies controlled perturbations with domain constraints

## 🚀 Quick Start

### Basic Usage

Generate 1000 synthetic samples
python scripts/generate_synthetic.py
--input data/lnp_data.json
--output data/synthetic_lnp_data.json
--n_samples 1000


### Advanced Usage

Custom strategy weights and random seed
python scripts/generate_synthetic.py
--input data/lnp_data.json
--output data/synthetic_custom.json
--n_samples 5000
--random_state 123
--strategy_weights 0.5,0.3,0.2


## 📊 Input Data Format

Your input data should be in LNPredict JSON format:

{
"component_library": {
"smiles_mapping": {
"ALC-0315": "CC[C@H](C(=O)OC[C@H]...",
"DSPC": "CCCCCCCCCCCCCCCCCC(=O)OC..."
}
},
"formulations": [
{
"id": "LNP_001",
"target": 85.2,
"components": {
"ALC-0315": {"composition_percent": 35.0},
"DSPC": {"composition_percent": 16.0}
},
"properties": {
"size": 102.3,
"polydispersity_index": 0.18,
"zeta_potential": -8.5,
"encapsulation_efficiency": 92.1
}
}
]
}


## 🎛️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input, -i` | Input LNP dataset (JSON) | `data/lnp_data.json` |
| `--output, -o` | Output synthetic dataset path | `data/synthetic_lnp_data.json` |
| `--n_samples, -n` | Number of samples to generate | `1000` |
| `--random_state, -r` | Random seed for reproducibility | `42` |
| `--strategy_weights` | Strategy weights (interpolation,nn,perturbation) | `0.4,0.3,0.3` |

## 📈 Quality Metrics

The generator evaluates synthetic data quality using:

- **Statistical Similarity**: Mann-Whitney U tests for distribution matching
- **Range Coverage**: How well synthetic data covers original data ranges
- **Correlation Preservation**: Maintenance of feature relationships
- **Diversity Score**: Uniqueness of generated samples

### Quality Interpretation

| Score | Quality | Recommendation |
|-------|---------|----------------|
| > 0.7 | Excellent | Ready for ML training |
| 0.5-0.7 | Good | Suitable for augmentation |
| 0.3-0.5 | Decent | Consider parameter tuning |
| < 0.3 | Moderate | May need strategy adjustment |

## 🔬 Generation Strategies

### 1. Composition Interpolation
- **Purpose**: Preserve compositional relationships
- **Method**: Interpolates between similar formulations using beta distribution weighting
- **Best for**: Maintaining chemical feasibility

### 2. Nearest Neighbor Augmentation  
- **Purpose**: Leverage local data structure
- **Method**: Creates weighted combinations of k-nearest neighbors
- **Best for**: Dense regions of the data space

### 3. Perturbation Generation
- **Purpose**: Controlled exploration of parameter space  
- **Method**: Applies domain-aware perturbations with constraint enforcement
- **Best for**: Exploring parameter variations

## 🛠️ Customization

### Strategy Weights

Adjust the contribution of each strategy:

More interpolation, less perturbation
python scripts/generate_synthetic.py
--strategy_weights 0.6,0.3,0.1
--n_samples 2000

text

### Domain Constraints

Modify constraints in `src/data_generation/strategies/perturbation.py`:

def _apply_domain_constraints(self, props, prop_cols):
for i, col in enumerate(prop_cols):
if 'size' in col.lower():
constrained_props[i] = np.clip(constrained_props[i], 50, 200) # Custom range
# Add more constraints...

text

## 📁 Output Files

The generator creates:

1. **Synthetic Dataset** (`synthetic_lnp_data.json`): Generated formulations
2. **Quality Report** (`synthetic_lnp_data_quality_report.json`): Detailed metrics

### Quality Report Structure

{
"avg_statistical_pvalue": 0.65,
"range_coverage": 0.82,
"correlation_preservation": 0.74,
"diversity_score": 0.91,
"overall_quality": 0.78,
"num_features_tested": 12,
"distribution_details": { ... }
}

text

## 🐍 Python API

### Basic Usage

from src.data_generation import TargetedAugmentationGenerator

Initialize generator
generator = TargetedAugmentationGenerator(random_state=42)

Load data
original_df = generator.load_data('data/lnp_data.json')

Generate synthetic data
synthetic_df = generator.generate_synthetic_dataset(
original_df,
n_samples=1000,
strategy_weights={'interpolation': 0.5, 'nn': 0.3, 'perturbation': 0.2}
)

Assess quality
quality_report = generator.assess_quality(original_df, synthetic_df)

Save results
generator.save_synthetic_data(
synthetic_df,
'data/lnp_data.json',
'data/synthetic_output.json',
quality_report
)

text

### Advanced Usage

Custom strategy initialization
from src.data_generation.strategies import CompositionInterpolation
from src.data_generation.quality import QualityAssessor

Use individual strategies
interpolation = CompositionInterpolation(random_state=42)
samples = interpolation.generate_samples(comps, props, targets, n_samples=500, ...)

Custom quality assessment
assessor = QualityAssessor()
quality = assessor.assess_quality(original_df, synthetic_df)

text

## 🔍 Troubleshooting

### Common Issues

**Low Quality Scores**:
- Try different strategy weights
- Increase sample size for better statistics
- Check input data quality and completeness

**Memory Issues**:
- Reduce `n_samples` for large datasets
- Process in batches using the Python API

**Domain Constraint Violations**:
- Modify constraints in `perturbation.py`
- Adjust perturbation scales

### Debug Mode

Add verbose logging by modifying the strategies to print intermediate steps.

## 📝 Examples

### Example 1: Small Dataset Augmentation

For datasets with < 100 samples
python scripts/generate_synthetic.py
--input data/small_dataset.json
--n_samples 2000
--strategy_weights 0.5,0.4,0.1 # More interpolation for small data

text

### Example 2: Balanced Generation

For moderate-sized datasets
python scripts/generate_synthetic.py
--input data/medium_dataset.json
--n_samples 5000
--strategy_weights 0.4,0.3,0.3 # Balanced approach

text

### Example 3: Exploration Focus

For exploring parameter space
python scripts/generate_synthetic.py
--input data/constrained_dataset.json
--n_samples 3000
--strategy_weights 0.2,0.3,0.5 # More perturbation

text

## 🔗 Integration with LNPredict

Use synthetic data with the main LNPredict training:

Generate synthetic data
python scripts/generate_synthetic.py
--input data/lnp_data.json
--output data/synthetic_lnp_data.json
--n_samples 3000

Train with both original and synthetic data
python scripts/hyperparameter_search.py
--data data/lnp_data.json
--synthetic data/synthetic_lnp_data.json
--output results/enhanced_training

text

## 🤝 Contributing

To extend the synthetic data generation:

1. **Add New Strategies**: Create new strategy classes in `src/data_generation/strategies/`
2. **Enhance Quality Metrics**: Extend `QualityAssessor` with domain-specific metrics
3. **Custom Constraints**: Modify domain constraints for your specific use case

## 📄 License

This module is part of LNPredict and follows the same license terms.

---
