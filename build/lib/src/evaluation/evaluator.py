"""
Comprehensive Model Evaluator with Visualization
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any

from src.utils.model_utils import load_best_model
from .visualizer import EvaluationVisualizer


class LNPEvaluator:
    """Comprehensive evaluator for LNP models."""
    
    def __init__(self, model_path: str, config: Dict):
        self.model_path = model_path
        self.config = config
        self.model, self.checkpoint = load_best_model(model_path)
        self.visualizer = EvaluationVisualizer(config)
    
    def comprehensive_evaluation(self, data_config: Dict, output_dir: Path, 
                               generate_plots: bool = True) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""
        
        results = {}
        
        # 1. Basic Performance Metrics
        print("📊 Computing performance metrics...")
        results['metrics'] = self._compute_metrics(data_config)
        
        # 2. Prediction Analysis
        print("🎯 Analyzing predictions...")
        results['predictions'] = self._analyze_predictions(data_config)
        
        # 3. Feature Importance Analysis
        print("🔍 Analyzing feature importance...")
        results['feature_importance'] = self._analyze_feature_importance(data_config)
        
        # 4. Uncertainty Analysis
        print("📈 Analyzing prediction uncertainty...")
        results['uncertainty'] = self._analyze_uncertainty(data_config)
        
        if generate_plots:
            # 5. Generate Visualizations
            print("📊 Generating visualizations...")
            self._generate_evaluation_plots(results, output_dir)
        
        # 6. Save Detailed Results
        self._save_evaluation_results(results, output_dir)
        
        return results
    
    def _compute_metrics(self, data_config: Dict) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        
        # Test on validation data
        val_data = data_config['val_data']
        predictions, outputs = self.model.predict(
            val_data['molecular'].cpu().numpy(),
            val_data['composition'].cpu().numpy(),
            val_data['property'].cpu().numpy()
        )
        
        targets = val_data['targets'].cpu().numpy()
        
        metrics = {
            'r2_score': r2_score(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'mse': mean_squared_error(targets, predictions),
            'explained_variance': self._explained_variance_score(targets, predictions),
            'mean_absolute_percentage_error': self._mape(targets, predictions),
            'correlation': np.corrcoef(targets, predictions)[0, 1]
        }
        
        return metrics
    
    def _analyze_predictions(self, data_config: Dict) -> Dict[str, np.ndarray]:
        """Analyze model predictions in detail."""
        
        val_data = data_config['val_data']
        predictions, outputs = self.model.predict(
            val_data['molecular'].cpu().numpy(),
            val_data['composition'].cpu().numpy(),
            val_data['property'].cpu().numpy()
        )
        
        targets = val_data['targets'].cpu().numpy()
        residuals = targets - predictions
        
        return {
            'targets': targets,
            'predictions': predictions,
            'residuals': residuals,
            'uncertainties': outputs['uncertainty'].cpu().numpy() if 'uncertainty' in outputs else None
        }
    
    def _analyze_feature_importance(self, data_config: Dict) -> Dict[str, np.ndarray]:
        """Analyze feature importance."""
        
        val_data = data_config['val_data']
        _, outputs = self.model.predict(
            val_data['molecular'].cpu().numpy(),
            val_data['composition'].cpu().numpy(),
            val_data['property'].cpu().numpy()
        )
        
        importance_scores = {}
        
        if 'feature_importance' in outputs:
            importance_scores['component_importance'] = outputs['feature_importance'].cpu().numpy().mean(axis=0)
        
        if 'attention_weights' in outputs:
            importance_scores['attention_weights'] = outputs['attention_weights'].cpu().numpy().mean(axis=0)
        
        return importance_scores
    
    def _analyze_uncertainty(self, data_config: Dict) -> Dict[str, float]:
        """Analyze prediction uncertainty."""
        
        val_data = data_config['val_data']
        predictions, outputs = self.model.predict(
            val_data['molecular'].cpu().numpy(),
            val_data['composition'].cpu().numpy(),
            val_data['property'].cpu().numpy()
        )
        
        uncertainty_stats = {}
        
        if 'uncertainty' in outputs:
            uncertainties = outputs['uncertainty'].cpu().numpy()
            uncertainty_stats = {
                'mean_uncertainty': np.mean(uncertainties),
                'std_uncertainty': np.std(uncertainties),
                'min_uncertainty': np.min(uncertainties),
                'max_uncertainty': np.max(uncertainties)
            }
        
        return uncertainty_stats
    
    def _generate_evaluation_plots(self, results: Dict, output_dir: Path):
        """Generate comprehensive evaluation plots."""
        
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Prediction vs True scatter plot
        self.visualizer.plot_prediction_scatter(
            results['predictions']['targets'],
            results['predictions']['predictions'],
            save_path=plots_dir / 'prediction_scatter.png'
        )
        
        # 2. Residuals analysis
        self.visualizer.plot_residuals_analysis(
            results['predictions']['predictions'],
            results['predictions']['residuals'],
            save_path=plots_dir / 'residuals_analysis.png'
        )
        
        # 3. Feature importance plots
        if results['feature_importance']:
            self.visualizer.plot_feature_importance(
                results['feature_importance'],
                save_path=plots_dir / 'feature_importance.png'
            )
        
        # 4. Uncertainty analysis
        if results['predictions']['uncertainties'] is not None:
            self.visualizer.plot_uncertainty_analysis(
                results['predictions']['predictions'],
                results['predictions']['uncertainties'],
                save_path=plots_dir / 'uncertainty_analysis.png'
            )
        
        # 5. Performance metrics summary
        self.visualizer.plot_metrics_summary(
            results['metrics'],
            save_path=plots_dir / 'metrics_summary.png'
        )
        
    def _save_evaluation_results(self, results: Dict, output_dir: Path):
        """Save detailed evaluation results."""
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'true_values': results['predictions']['targets'],
            'predicted_values': results['predictions']['predictions'],
            'residuals': results['predictions']['residuals']
        })
        
        if results['predictions']['uncertainties'] is not None:
            predictions_df['uncertainty'] = results['predictions']['uncertainties']
        
        predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
        
        # Save feature importance if available
        if results['feature_importance']:
            importance_df = pd.DataFrame(results['feature_importance'])
            importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    def print_evaluation_summary(self, results: Dict):
        """Print comprehensive evaluation summary."""
        
        print("\n" + "="*60)
        print("📊 === MODEL EVALUATION SUMMARY ===")
        print("="*60)
        
        metrics = results['metrics']
        print(f"🎯 Performance Metrics:")
        print(f"   R² Score: {metrics['r2_score']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   Correlation: {metrics['correlation']:.4f}")
        print(f"   MAPE: {metrics['mean_absolute_percentage_error']:.2f}%")
        
        if results['uncertainty']:
            print(f"\n🔍 Uncertainty Analysis:")
            unc = results['uncertainty']
            print(f"   Mean Uncertainty: {unc['mean_uncertainty']:.4f}")
            print(f"   Std Uncertainty: {unc['std_uncertainty']:.4f}")
        
        print(f"\n📈 Model Information:")
        print(f"   Training R²: {self.checkpoint['val_r2']:.4f}")
        print(f"   Training Epoch: {self.checkpoint['epoch']}")
        print(f"   Model Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
        # Performance assessment
        r2 = metrics['r2_score']
        if r2 >= 0.7:
            print(f"\n✅ EXCELLENT performance! Model is highly predictive.")
        elif r2 >= 0.5:
            print(f"\n✅ GOOD performance! Model shows strong predictive capability.")
        elif r2 >= 0.3:
            print(f"\n⚠️  MODERATE performance. Consider model improvements.")
        else:
            print(f"\n❌ POOR performance. Model needs significant improvements.")
    
    @staticmethod
    def _explained_variance_score(y_true, y_pred):
        """Calculate explained variance score."""
        return 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    @staticmethod
    def _mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
