# LNPredict

**A Deep Learning Framework for Lipid Nanoparticle Property Prediction**

LNPredict is a comprehensive deep learning framework for predicting lipid nanoparticle (LNP) properties using advanced neural network architectures with attention mechanisms. The framework provides automated hyperparameter tuning, comprehensive evaluation tools, and production-ready model deployment capabilities.

***

## 📋 Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Format](#data-format)
5. [Training Models](#training-models)
6. [Hyperparameter Search](#hyperparameter-search)
7. [Model Evaluation](#model-evaluation)
8. [Making Predictions](#making-predictions)
9. [Configuration](#configuration)
10. [API Reference](#api-reference)
11. [Examples](#examples)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)
14. [License](#license)
15. [Citation](#citation)

***

## ✨ Features

### 🧠 **Advanced Neural Architecture**

- **Multi-Head Attention**: Self-attention mechanisms for molecular feature processing
- **Ensemble Predictions**: Dual prediction heads with uncertainty estimation
- **Compositional Modeling**: Specialized handling of compositional data constraints
- **Feature Importance**: Built-in interpretability and attention weight analysis


### 🔧 **Training \& Optimization**

- **Automated Hyperparameter Tuning**: Comprehensive search across 25+ configurations
- **Anti-Saturation Training**: Advanced techniques to prevent validation loss plateaus
- **Multiple Loss Functions**: MSE, Huber, and L1 loss combinations with adaptive weighting
- **Model Checkpointing**: Automatic saving of best models during training


### 📊 **Evaluation \& Analysis**

- **Comprehensive Metrics**: R², RMSE, MAE, correlation analysis
- **Advanced Visualizations**: 9-panel hyperparameter analysis plots
- **Uncertainty Analysis**: Prediction confidence and error estimation
- **Performance Tracking**: Training curves and convergence monitoring


### 🚀 **Production Ready**

- **Model Deployment**: Easy loading and inference for new samples
- **Batch Processing**: Efficient prediction on large datasets
- **Configuration Management**: YAML-based configuration system
- **Extensible Architecture**: Modular design for easy customization

***

## 🔧 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: Optional but recommended for GPU acceleration
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large datasets


### Method 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/lnpredict.git
cd lnpredict

# Create virtual environment
python -m venv lnpredict_env
source lnpredict_env/bin/activate  # On Windows: lnpredict_env\Scripts\activate

# Install package
pip install -e .
```


### Method 2: Quick Install

```bash
pip install -r requirements.txt
python setup.py install
```


### Verify Installation

```bash
# Test basic functionality
python -c "from src.models.regressor import TunedLNPRegressor; print('✅ LNPredict installed successfully!')"

# Check GPU availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```


***

## 🚀 Quick Start

### 1. Prepare Your Data

Ensure your data is in the required JSON format (see [Data Format](#data-format)):

```json
{
  "component_library": {
    "smiles_mapping": {
      "ALC-0315": "CC(C)CC(C)CC(C)CC(C)CC(C)CC(C)C",
      "DSPC": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H]..."
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
```


### 2. Train Your First Model

```bash
# Train a single model with default settings
python scripts/train.py \
    --data data/lnp_data.json \
    --output outputs/my_first_model \
    --epochs 500 \
    --learning_rate 0.001
```


### 3. Run Hyperparameter Search

```bash
# Find the best hyperparameters
python scripts/hyperparameter_search.py \
    --data data/lnp_data.json \
    --output outputs/hyperparam_search \
    --max_trials 25
```


### 4. Evaluate Your Model

```bash
# Comprehensive model evaluation
python scripts/evaluate.py \
    --model_path outputs/hyperparam_search/BEST_OVERALL_MODEL.pth \
    --data data/lnp_data.json \
    --output outputs/evaluation \
    --generate_plots
```


### 5. Make Predictions

```bash
# Predict on new samples
python scripts/predict.py \
    --model_path outputs/hyperparam_search/BEST_OVERALL_MODEL.pth \
    --input_data data/new_samples.json \
    --output predictions.csv \
    --include_uncertainty
```


***

## 📊 Data Format

### Input Data Structure

LNPredict expects data in JSON format with the following structure:

```json
{
  "component_library": {
    "smiles_mapping": {
      "component_name": "SMILES_string",
      "ALC-0315": "CC[C@H](C(=O)OC[C@H](COP(=O)(O)OCC[N+](C)(C)C)OC(=O)C)",
      "DSPC": "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCC[N+](C)(C)C)OC(=O)C"
    }
  },
  "formulations": [
    {
      "id": "unique_sample_id",
      "target": 85.2,  // Target property to predict
      "components": {
        "component_name": {
          "composition_percent": 35.0,
          "smiles": "SMILES_string"  // Optional if in library
        }
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
```


### Required Fields

- **component_library.smiles_mapping**: Map of component names to SMILES strings
- **formulations[].id**: Unique identifier for each sample
- **formulations[].target**: Target property value to predict
- **formulations[].components**: Component compositions (must sum to reasonable total)
- **formulations[].properties**: Physicochemical properties


### Property Guidelines

| Property | Description | Typical Range | Units |
| :-- | :-- | :-- | :-- |
| `size` | Particle diameter | 50-200 | nm |
| `polydispersity_index` | Size uniformity | 0.05-0.5 | - |
| `zeta_potential` | Surface charge | -50 to +50 | mV |
| `encapsulation_efficiency` | Drug loading | 0-100 | % |


***

## 🏋️ Training Models

### Single Model Training

Train a model with specific hyperparameters:

```bash
python scripts/train.py \
    --data data/lnp_data.json \
    --output outputs/single_model \
    --config config/config.yaml \
    --epochs 500 \
    --learning_rate 0.001 \
    --dropout 0.2
```


### Training Options

| Parameter | Description | Default | Options |
| :-- | :-- | :-- | :-- |
| `--data` | Training data path | Required | JSON file |
| `--output` | Output directory | `outputs/single_model` | Any path |
| `--config` | Configuration file | `config/config.yaml` | YAML file |
| `--epochs` | Training epochs | 500 | Integer |
| `--learning_rate` | Learning rate | 0.001 | Float |
| `--dropout` | Dropout rate | 0.2 | 0.0-1.0 |

### Configuration File Training

Create a custom configuration:

```yaml
# custom_config.yaml
model:
  hidden_dims: [256, 128, 64]
  num_heads: 8
  dropout_rate: 0.15

training:
  epochs: 1000
  learning_rate: 0.0005
  weight_decay: 0.0001
  scheduler_type: "cosine"
  loss_weights: [0.7, 0.2, 0.1]
```

```bash
python scripts/train.py \
    --data data/lnp_data.json \
    --config custom_config.yaml \
    --output outputs/custom_model
```


### Training Outputs

After training, you'll find:

```
outputs/single_model/
├── best_model.pth              # Best model checkpoint
├── training_history.json      # Training metrics
├── training_curves.png        # Loss and R² plots
└── config_used.yaml          # Configuration used
```


### Monitoring Training

Monitor training progress in real-time:

```bash
# View training logs
tail -f outputs/single_model/training.log

# Plot training curves (if matplotlib available)
python -c "
import json
import matplotlib.pyplot as plt

with open('outputs/single_model/training_history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_losses'])
plt.title('Training Loss')
plt.subplot(1, 2, 2)
plt.plot(range(0, len(history['val_r2_scores']) * 5, 5), history['val_r2_scores'])
plt.title('Validation R²')
plt.show()
"
```


***

## 🔍 Hyperparameter Search

### Comprehensive Search

Run a comprehensive hyperparameter search to find optimal configurations:

```bash
python scripts/hyperparameter_search.py \
    --data data/lnp_data.json \
    --synthetic data/synthetic_lnp_data.json \  # Optional
    --output outputs/hyperparam_search \
    --max_trials 25
```


### Search Configuration

Customize the search space in `config/config.yaml`:

```yaml
hyperparameter_search:
  max_trials: 25
  search_space:
    learning_rate: [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
    weight_decay: [0, 1e-6, 1e-5, 1e-4, 1e-3]
    dropout_rate: [0.1, 0.15, 0.2, 0.3]
    batch_size: [16, 32, 64]
    hidden_dims:
      - [128, 64]
      - [256, 128]
      - [128, 64, 32]
    num_heads: [4, 6, 8]
    scheduler_type: ["cosine", "plateau", "step"]
    loss_weights:
      - [1.0, 0.0, 0.0]    # Pure MSE
      - [0.6, 0.3, 0.1]    # Balanced
      - [0.4, 0.4, 0.2]    # Robust losses
```


### Search Outputs

The hyperparameter search generates:

```
outputs/hyperparam_search/
├── BEST_OVERALL_MODEL.pth          # Best model across all trials
├── comprehensive_results.json      # Complete search results
├── comprehensive_analysis.png      # 9-panel analysis plots
├── trial_1/
│   └── best_model_trial_1.pth     # Best model from trial 1
├── trial_2/
│   └── best_model_trial_2.pth     # Best model from trial 2
└── ...
```


### Understanding Results

```bash
# View search summary
python -c "
import json
with open('outputs/hyperparam_search/comprehensive_results.json', 'r') as f:
    results = json.load(f)
    
print(f'Best R²: {results[\"best_r2\"]:.4f}')
print(f'Average R²: {results[\"avg_r2\"]:.4f}')
print(f'Successful trials: {results[\"num_trials\"]}')
print(f'Best config: {results[\"best_config\"]}')
"
```


### Early Stopping

The search includes intelligent early stopping:

- **Breakthrough detection**: Stops early if R² ≥ 0.5 achieved
- **Plateau detection**: Reduces learning rate if validation stagnates
- **Time management**: Prevents excessive training time per trial

***

## 📈 Model Evaluation

### Comprehensive Evaluation

Perform thorough model evaluation with visualizations:

```bash
python scripts/evaluate.py \
    --model_path outputs/hyperparam_search/BEST_OVERALL_MODEL.pth \
    --data data/lnp_data.json \
    --output outputs/evaluation \
    --generate_plots
```


### Evaluation Options

| Parameter | Description | Default |
| :-- | :-- | :-- |
| `--model_path` | Path to trained model | Required |
| `--data` | Test data path | Required |
| `--output` | Output directory | `outputs/evaluation` |
| `--generate_plots` | Generate visualization plots | False |

### Evaluation Outputs

```
outputs/evaluation/
├── predictions.csv              # Predictions with uncertainties
├── metrics.csv                 # Performance metrics
├── feature_importance.csv      # Feature importance scores
└── plots/
    ├── prediction_scatter.png  # True vs Predicted
    ├── residuals_analysis.png  # Residuals diagnostics
    ├── uncertainty_analysis.png # Uncertainty plots
    ├── feature_importance.png  # Feature importance
    └── metrics_summary.png     # Performance summary
```


### Key Metrics

| Metric | Description | Good Value | Excellent Value |
| :-- | :-- | :-- | :-- |
| **R²** | Coefficient of determination | > 0.5 | > 0.7 |
| **RMSE** | Root mean squared error | < 10% of target range | < 5% of target range |
| **MAE** | Mean absolute error | < 8% of target range | < 4% of target range |
| **Correlation** | Pearson correlation | > 0.7 | > 0.85 |

### Custom Evaluation

For advanced evaluation, use the Python API:

```python
from src.evaluation.evaluator import LNPEvaluator
from src.data.data_loader import LNPDataLoader

# Load data
data_loader = LNPDataLoader()
data_config = data_loader.load_and_prepare('data/test_data.json')

# Create evaluator
evaluator = LNPEvaluator('outputs/best_model.pth', config)

# Run evaluation
results = evaluator.comprehensive_evaluation(
    data_config, 
    'outputs/custom_eval',
    generate_plots=True
)

# Print summary
evaluator.print_evaluation_summary(results)
```


### Interpreting Results

#### **Excellent Performance (R² ≥ 0.7)**

- Model is highly predictive
- Ready for production use
- Consider publishing results


#### **Good Performance (0.5 ≤ R² < 0.7)**

- Model shows strong predictive capability
- Suitable for most applications
- Consider minor hyperparameter refinement


#### **Moderate Performance (0.3 ≤ R² < 0.5)**

- Model has some predictive value
- Recommend hyperparameter tuning
- Consider data augmentation


#### **Poor Performance (R² < 0.3)**

- Model needs significant improvement
- Check data quality and preprocessing
- Consider architecture changes

***

## 🔮 Making Predictions

### Basic Predictions

Predict properties for new LNP formulations:

```bash
python scripts/predict.py \
    --model_path outputs/hyperparam_search/BEST_OVERALL_MODEL.pth \
    --input_data data/new_samples.json \
    --output predictions.csv \
    --include_uncertainty
```


### Prediction Options

| Parameter | Description | Default |
| :-- | :-- | :-- |
| `--model_path` | Trained model path | Required |
| `--input_data` | Input data (JSON) | Required |
| `--output` | Output file (CSV) | `predictions.csv` |
| `--include_uncertainty` | Include uncertainty estimates | False |

### Input Data Format

Your prediction data should match the training format:

```json
{
  "component_library": {
    "smiles_mapping": {
      "ALC-0315": "CC[C@H](C(=O)OC[C@H]...",
      "DSPC": "CCCCCCCCCCCCCCCCCC(=O)OC..."
    }
  },
  "formulations": [
    {
      "id": "NEW_001",
      "components": {
        "ALC-0315": {"composition_percent": 40.0},
        "DSPC": {"composition_percent": 15.0},
        "Cholesterol": {"composition_percent": 20.0},
        "DMG-PEG": {"composition_percent": 25.0}
      },
      "properties": {
        "size": 105.0,
        "polydispersity_index": 0.16,
        "zeta_potential": -12.0,
        "encapsulation_efficiency": 88.0
      }
    }
  ]
}
```


### Prediction Output

The output CSV contains:

```csv
sample_id,predicted_target,uncertainty,confidence,component_0_importance,...
NEW_001,78.45,0.034,0.966,0.25,0.30,0.20,0.15,...
NEW_002,82.11,0.028,0.972,0.22,0.35,0.18,0.17,...
```


### Batch Processing

For large datasets, use the Python API for efficient batch processing:

```python
from src.utils.model_utils import predict_with_saved_model
import pandas as pd

# Load model and make predictions
predictions, outputs = predict_with_saved_model(
    'outputs/best_model.pth',
    molecular_features,
    composition_features,
    property_features
)

# Create results DataFrame
results = pd.DataFrame({
    'sample_id': sample_ids,
    'predicted_target': predictions,
    'uncertainty': outputs['uncertainty'].cpu().numpy(),
    'confidence': 1 - outputs['uncertainty'].cpu().numpy()
})

# Save results
results.to_csv('batch_predictions.csv', index=False)
```


### Quality Control

Monitor prediction quality:

```python
# Check prediction statistics
print(f"Mean prediction: {predictions.mean():.2f}")
print(f"Prediction std: {predictions.std():.2f}")
print(f"Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")

# Identify high-confidence predictions
high_confidence = outputs['uncertainty'] < 0.1
print(f"High confidence samples: {high_confidence.sum()}/{len(predictions)}")

# Flag unusual predictions
unusual = (predictions < 0) | (predictions > 100)  # Adjust based on your target
if unusual.any():
    print(f"Warning: {unusual.sum()} unusual predictions detected")
```


***

## ⚙️ Configuration

### Configuration File Structure

LNPredict uses YAML configuration files for easy customization:

```yaml
# config/config.yaml

# Model Architecture
model:
  hidden_dims: [128, 64]
  num_heads: 4
  dropout_rate: 0.2
  
# Training Configuration  
training:
  epochs: 500
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 32
  early_stopping_patience: 50
  scheduler_type: "cosine"
  loss_weights: [0.6, 0.3, 0.1]  # MSE, Huber, L1

# Data Configuration
data:
  test_size: 0.15
  random_state: 42
  stratify_bins: 5

# Hyperparameter Search
hyperparameter_search:
  max_trials: 25
  search_space:
    learning_rate: [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]
    weight_decay: [0, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
    dropout_rate: [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    batch_size: [8, 16, 32, 64]
    hidden_dims:
      - [^64]
      - [^128]
      - [64, 32]
      - [128, 64]
      - [256, 128]
      - [128, 64, 32]
      - [256, 128, 64]
    num_heads: [2, 4, 6, 8]
    early_stopping_patience: [30, 50, 75, 100]
    scheduler_type: ["cosine", "plateau", "step", "exponential", "linear"]
    loss_weights:
      - [1.0, 0.0, 0.0]
      - [0.8, 0.2, 0.0]
      - [0.6, 0.3, 0.1]
      - [0.4, 0.4, 0.2]
      - [0.5, 0.3, 0.2]

# Evaluation Configuration
evaluation:
  generate_plots: true
  save_predictions: true
  detailed_analysis: true
  plot_formats: ["png", "pdf"]
  
# Output Configuration  
output:
  save_checkpoints: true
  checkpoint_frequency: 5  # epochs
  log_level: "INFO"
```


### Model Architecture Options

| Parameter | Description | Typical Values |
| :-- | :-- | :-- |
| `hidden_dims` | Network layer sizes | `[128, 64]`, `[256, 128, 64]` |
| `num_heads` | Attention heads | `4`, `6`, `8` |
| `dropout_rate` | Dropout probability | `0.1` - `0.3` |

### Training Options

| Parameter | Description | Recommendations |
| :-- | :-- | :-- |
| `learning_rate` | Optimizer learning rate | `1e-4` to `1e-3` for stability |
| `weight_decay` | L2 regularization | `1e-5` to `1e-3` |
| `scheduler_type` | LR scheduler | `"cosine"` for smooth decay |
| `loss_weights` | Loss function weights | `[0.6, 0.3, 0.1]` balanced |

### Environment Variables

Set environment variables for advanced configuration:

```bash
# GPU settings
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Memory management

# Performance settings
export OMP_NUM_THREADS=4      # CPU threads
export MKL_NUM_THREADS=4      # Intel MKL threads

# Logging
export LNPREDICT_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
export LNPREDICT_CACHE_DIR=/tmp/lnpredict_cache  # Cache directory
```


***

## 📚 API Reference

### Core Classes

#### **TunedLNPRegressor**

Main model class for LNP property prediction.

```python
from src.models.regressor import TunedLNPRegressor

# Initialize model
model_config = {
    'molecular_feature_dim': 64,
    'n_components': 4,
    'property_dim': 4,
    'hidden_dims': [128, 64],
    'num_heads': 4,
    'dropout_rate': 0.2
}

regressor = TunedLNPRegressor(model_config, device='cuda')

# Make predictions
predictions, outputs = regressor.predict(
    molecular_features, 
    composition_features, 
    property_features
)
```


#### **LNPDataLoader**

Data loading and preprocessing utilities.

```python
from src.data.data_loader import LNPDataLoader

data_loader = LNPDataLoader()

# Load and prepare data
data_config = data_loader.load_and_prepare(
    data_path='data/lnp_data.json',
    synthetic_data='data/synthetic_data.json',  # Optional
    config=config['data'],
    device='cuda'
)
```


#### **LNPTrainer**

Single model training with comprehensive logging.

```python
from src.training.trainer import LNPTrainer

trainer = LNPTrainer(config, device='cuda')

# Train model
results = trainer.train(data_config, output_dir)

# Print results
trainer.print_training_summary(results)
```


#### **ComprehensiveHyperparameterTuner**

Automated hyperparameter optimization.

```python
from src.training.hyperparameter_tuner import ComprehensiveHyperparameterTuner

tuner = ComprehensiveHyperparameterTuner(data_config, tuning_config)

# Run hyperparameter search
analysis = tuner.run_comprehensive_search(output_dir)

print(f"Best R²: {analysis['best_r2']:.4f}")
```


#### **LNPEvaluator**

Comprehensive model evaluation with visualizations.

```python
from src.evaluation.evaluator import LNPEvaluator

evaluator = LNPEvaluator(model_path, eval_config)

# Comprehensive evaluation
results = evaluator.comprehensive_evaluation(
    data_config,
    output_dir,
    generate_plots=True
)
```


### Utility Functions

#### **Model Utilities**

```python
from src.utils.model_utils import (
    load_best_model, 
    predict_with_saved_model,
    evaluate_saved_model
)

# Load trained model
model, checkpoint = load_best_model('path/to/model.pth')

# Make predictions
predictions, outputs = predict_with_saved_model(
    'path/to/model.pth',
    molecular_features,
    composition_features, 
    property_features
)

# Evaluate model performance  
metrics, preds = evaluate_saved_model(
    'path/to/model.pth',
    molecular_features,
    composition_features,
    property_features,
    targets
)
```


### Data Processing Pipeline

```python
# Complete prediction pipeline
from src.data.data_loader import LNPDataLoader
from src.utils.model_utils import predict_with_saved_model

# 1. Load and process data
data_loader = LNPDataLoader()
features, targets, feature_names = data_loader.featurizer.create_comprehensive_features(
    'data/new_samples.json'
)

# 2. Separate feature types
mol_mask = [name.startswith('mol_') for name in feature_names]
comp_mask = [name.startswith('comp_') for name in feature_names]  
prop_mask = [name.startswith('prop_') for name in feature_names]

molecular_features = features[:, mol_mask]
composition_features = features[:, comp_mask]
property_features = features[:, prop_mask]

# 3. Make predictions
predictions, outputs = predict_with_saved_model(
    'outputs/best_model.pth',
    molecular_features,
    composition_features,
    property_features
)

print(f"Predictions: {predictions}")
print(f"Uncertainties: {outputs['uncertainty']}")
```


***

## 📖 Examples

### Example 1: Complete Training Pipeline

```python
import yaml
from pathlib import Path
from src.data.data_loader import LNPDataLoader
from src.training.trainer import LNPTrainer
from src.evaluation.evaluator import LNPEvaluator

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup paths
data_path = 'data/lnp_data.json'
output_dir = Path('outputs/complete_pipeline')
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
data_loader = LNPDataLoader()
data_config = data_loader.load_and_prepare(
    data_path, 
    config=config['data'], 
    device='cuda'
)

# Train model
trainer = LNPTrainer(config, device='cuda')
training_results = trainer.train(data_config, output_dir)

# Evaluate model
evaluator = LNPEvaluator(training_results['best_model_path'], config['evaluation'])
eval_results = evaluator.comprehensive_evaluation(
    data_config,
    output_dir / 'evaluation',
    generate_plots=True
)

print(f"Training R²: {training_results['best_val_r2']:.4f}")
print(f"Final R²: {eval_results['metrics']['r2_score']:.4f}")
```


### Example 2: Hyperparameter Optimization

```python
from src.training.hyperparameter_tuner import ComprehensiveHyperparameterTuner

# Custom hyperparameter search
tuning_config = {
    'max_trials': 10,
    'search_space': {
        'learning_rate': [1e-4, 1e-3, 5e-3],
        'dropout_rate': [0.1, 0.2, 0.3],
        'hidden_dims': [[128, 64], [256, 128]],
        'num_heads': [4, 8],
        'scheduler_type': ['cosine', 'plateau'],
        'loss_weights': [(1.0, 0.0, 0.0), (0.6, 0.3, 0.1)]
    }
}

# Run tuning
tuner = ComprehensiveHyperparameterTuner(data_config, tuning_config)
analysis = tuner.run_comprehensive_search(output_dir / 'hyperparameter_search')

# Analyze results
print(f"Best configuration achieved R²: {analysis['best_r2']:.4f}")
print(f"Best hyperparameters: {analysis['best_config']}")

# Use best model for predictions
best_model_path = analysis['best_model_path']
predictions, outputs = predict_with_saved_model(
    best_model_path,
    new_molecular_features,
    new_composition_features,
    new_property_features
)
```


### Example 3: Custom Model Architecture

```python
from src.models.regressor import TunedLNPRegressor
import torch

# Define custom model configuration
custom_config = {
    'molecular_feature_dim': 128,
    'n_components': 6,
    'property_dim': 5,
    'hidden_dims': [512, 256, 128, 64],  # Deeper network
    'num_heads': 8,                      # More attention heads
    'dropout_rate': 0.15                 # Lower dropout
}

# Create model
model = TunedLNPRegressor(custom_config, device='cuda')

# Custom training loop
optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)

criterion = torch.nn.MSELoss()
best_loss = float('inf')

for epoch in range(1000):
    model.model.train()
    optimizer.zero_grad()
    
    # Forward pass
    predictions, _ = model.model(
        train_molecular, train_composition, train_properties
    )
    
    loss = criterion(predictions, train_targets)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    # Validation
    if epoch % 10 == 0:
        model.model.eval()
        with torch.no_grad():
            val_predictions, _ = model.model(
                val_molecular, val_composition, val_properties
            )
            val_loss = criterion(val_predictions, val_targets)
            
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.model.state_dict(), 'custom_best_model.pth')
            
        print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
```


### Example 4: Uncertainty Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from src.utils.model_utils import predict_with_saved_model

# Make predictions with uncertainty
predictions, outputs = predict_with_saved_model(
    'outputs/best_model.pth',
    molecular_features,
    composition_features,
    property_features
)

# Extract uncertainty information
uncertainties = outputs['uncertainty'].cpu().numpy()
feature_importance = outputs['feature_importance'].cpu().numpy()
attention_weights = outputs['attention_weights'].cpu().numpy()

# Uncertainty analysis
print("Uncertainty Analysis:")
print(f"Mean uncertainty: {np.mean(uncertainties):.3f}")
print(f"High confidence samples (unc < 0.1): {np.sum(uncertainties < 0.1)}")
print(f"Low confidence samples (unc > 0.3): {np.sum(uncertainties > 0.3)}")

# Plot uncertainty distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(uncertainties, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.title('Uncertainty Distribution')

plt.subplot(1, 3, 2)
plt.scatter(predictions, uncertainties, alpha=0.6)
plt.xlabel('Predictions')
plt.ylabel('Uncertainty')
plt.title('Prediction vs Uncertainty')

plt.subplot(1, 3, 3)
plt.bar(range(feature_importance.shape[^1]), feature_importance.mean(axis=0))
plt.xlabel('Component Index')
plt.ylabel('Average Importance')
plt.title('Feature Importance')

plt.tight_layout()
plt.show()

# Identify high-uncertainty predictions for review
high_uncertainty_idx = uncertainties > np.percentile(uncertainties, 90)
print(f"High uncertainty samples: {high_uncertainty_idx.sum()}")
print(f"Predictions: {predictions[high_uncertainty_idx]}")
```


***

## 🚨 Troubleshooting

### Common Issues

#### **1. CUDA Out of Memory**

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

```bash
# Reduce batch size
python scripts/train.py --batch_size 16

# Use gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Train on CPU (slower but works)
export CUDA_VISIBLE_DEVICES=""
```


#### **2. Import Errors**

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solutions**:

```bash
# Install in development mode
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use absolute imports
python -m scripts.train
```


#### **3. Data Format Issues**

**Error**: `KeyError: 'components'` or similar

**Solutions**:

- Verify JSON structure matches expected format
- Check all required fields are present
- Validate SMILES strings are correct

```python
# Data validation script
import json

def validate_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check required keys
    required_keys = ['component_library', 'formulations']
    for key in required_keys:
        assert key in data, f"Missing key: {key}"
    
    # Check formulations
    for i, form in enumerate(data['formulations']):
        required_form_keys = ['id', 'target', 'components', 'properties']
        for key in required_form_keys:
            assert key in form, f"Formulation {i} missing key: {key}"
    
    print("✅ Data format validation passed!")

# Validate your data
validate_data('data/lnp_data.json')
```


#### **4. Poor Model Performance**

**Symptoms**: R² < 0.3, high validation loss

**Solutions**:

1. **Check data quality**:

```python
# Data exploration
import pandas as pd
import matplotlib.pyplot as plt

# Load and examine data
df = pd.read_json('data/lnp_data.json')

# Check target distribution
targets = [f['target'] for f in df['formulations']]
plt.hist(targets, bins=20)
plt.title('Target Distribution')
plt.show()

# Check for outliers
print(f"Target range: {min(targets)} - {max(targets)}")
print(f"Target std: {np.std(targets)}")
```

2. **Increase model complexity**:

```yaml
model:
  hidden_dims: [512, 256, 128]  # Larger network
  num_heads: 8                  # More attention
  dropout_rate: 0.1             # Less dropout
```

3. **Hyperparameter search**:

```bash
python scripts/hyperparameter_search.py --max_trials 50
```


#### **5. Training Stagnation**

**Symptoms**: Validation loss plateaus, no improvement

**Solutions**:

```yaml
training:
  learning_rate: 0.003           # Higher learning rate
  scheduler_type: "cosine"       # Better scheduler
  weight_decay: 0               # Reduce regularization
  loss_weights: [1.0, 0.0, 0.0] # Simpler loss
```


#### **6. Memory Issues During Evaluation**

**Error**: Memory exhausted during plotting

**Solutions**:

```bash
# Disable plot generation
python scripts/evaluate.py --no-plots

# Or use smaller figure sizes
export MPLBACKEND=Agg  # Non-interactive backend
```


### Performance Optimization

#### **Speed Up Training**

```bash
# Use mixed precision
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Optimize data loading
export OMP_NUM_THREADS=4

# Use compiled model (PyTorch 2.0+)
export PYTORCH_JIT_USE_NVM=1
```


#### **Reduce Memory Usage**

```python
# Gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential(model, segments=2, input=x)

# Smaller batch sizes
config['training']['batch_size'] = 16

# Clear cache periodically
torch.cuda.empty_cache()
```


### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look in `outputs/*/training.log`
2. **Validate data**: Use the validation scripts provided
3. **Try minimal examples**: Start with smaller datasets
4. **Check dependencies**: Ensure all packages are up to date
5. **Open an issue**: Provide logs, data format, and system information

**System Information for Bug Reports**:

```python
import torch
import sys
import numpy as np

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
```


***

## 🤝 Contributing

We welcome contributions to LNPredict! Here's how to get involved:

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/lnpredict.git
cd lnpredict

# Create development environment
python -m venv lnpredict_dev
source lnpredict_dev/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```


### Code Style

We follow PEP 8 with Black formatting:

```bash
# Format code
black src/ scripts/ tests/

# Check style
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```


### Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py::TestLNPModels::test_molecular_encoder
```


### Adding Features

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Write tests**: Add tests in `tests/`
4. **Implement feature**: Add code with proper documentation
5. **Run tests**: Ensure all tests pass
6. **Submit PR**: Create a pull request with clear description

### Documentation

Update documentation for new features:

```bash
# Build docs locally (if sphinx installed)
cd docs/
make html
```


### Contribution Guidelines

- **Code Quality**: Follow existing patterns and style
- **Testing**: Add tests for new functionality
- **Documentation**: Update docstrings and README
- **Compatibility**: Ensure backward compatibility
- **Performance**: Consider computational efficiency

***

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

```
Copyright (c) 2025 LNPredict Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```


***

## 📖 Citation

If you use LNPredict in your research, please cite our work:

```bibtex
@software{lnpredict2025,
  title={LNPredict: A Deep Learning Framework for Lipid Nanoparticle Property Prediction},
  author={Your Name and Contributors},
  year={2025},
  url={https://github.com/your-username/lnpredict},
  version={1.0.0}
}
```


### Academic Usage

For academic publications, please include:

- **Method Name**: LNPredict
- **Description**: Deep learning framework with attention mechanisms for LNP property prediction
- **Architecture**: Multi-component neural networks with self-attention and ensemble predictions
- **Capabilities**: Automated hyperparameter tuning, uncertainty estimation, and comprehensive evaluation

***

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **scikit-learn**: For machine learning utilities and metrics
- **RDKit**: For molecular descriptor calculation
- **Contributors**: Thanks to all contributors and users

***

## 📞 Support \& Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/lnpredict/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/lnpredict/discussions)
- **Email**: your-email@domain.com
- **Documentation**: [Full Documentation](https://lnpredict.readthedocs.io)


### Community

- **Discord**: Join our community server
- **Twitter**: Follow @LNPredict for updates
- **Newsletter**: Subscribe for major releases

***

**Happy Predicting! 🚀**

*LNPredict - Advancing LNP Design Through Deep Learning*
<span style="display:none">[^2][^3][^4][^5][^6][^7][^8]</span>

<div style="text-align: center">⁂</div>

[^1]: https://github.com/othneildrew/Best-README-Template

[^2]: https://dev.to/zand/a-comprehensive-and-user-friendly-project-readmemd-template-2ei8

[^3]: https://www.thegooddocsproject.dev/template/readme

[^4]: https://www.readme-templates.com

[^5]: https://devblogs.microsoft.com/dotnet/write-a-high-quality-readme-for-nuget-packages/

[^6]: https://docs.scholarsphere.psu.edu/guides/writing-readme/

[^7]: https://data.research.cornell.edu/data-management/sharing/readme/

[^8]: https://gitlab.com/kopino4-templates/readme-template

