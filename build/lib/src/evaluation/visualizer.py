"""
Comprehensive Visualization for Model Evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import r2_score


class EvaluationVisualizer:
    """Advanced visualizer for model evaluation."""
    
    def __init__(self, config: Dict):
        self.config = config
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               save_path: Optional[Path] = None):
        """Create prediction vs true values scatter plot."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        
        # Create scatter plot
        scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        
        # Add 1:1 line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), "b-", alpha=0.8, label=f'Fit (R² = {r2:.3f})')
        
        # Formatting
        ax.set_xlabel('True Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_title('Predicted vs True Target Values', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add R² annotation
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nN = {len(y_true)}', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_residuals_analysis(self, y_pred: np.ndarray, residuals: np.ndarray,
                               save_path: Optional[Path] = None):
        """Create comprehensive residuals analysis plot."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, s=40)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot: Residuals vs Normal Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals vs Order
        ax4.plot(range(len(residuals)), residuals, 'o-', alpha=0.6, markersize=3)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax4.set_xlabel('Observation Order')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs Order')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_feature_importance(self, importance_dict: Dict[str, np.ndarray],
                               save_path: Optional[Path] = None):
        """Plot feature importance analysis."""
        
        n_plots = len(importance_dict)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
        
        if n_plots == 1:
            axes = [axes]
        
        for idx, (name, values) in enumerate(importance_dict.items()):
            ax = axes[idx]
            
            # Create bar plot
            x_pos = np.arange(len(values))
            bars = ax.bar(x_pos, values, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Color bars by magnitude
            colors = plt.cm.viridis(values / values.max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance Score')
            ax.set_title(f'{name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            # Add value annotations on bars
            for i, v in enumerate(values):
                ax.text(i, v + 0.01 * values.max(), f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_uncertainty_analysis(self, y_pred: np.ndarray, uncertainties: np.ndarray,
                                 save_path: Optional[Path] = None):
        """Plot uncertainty analysis."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prediction vs Uncertainty
        scatter = ax1.scatter(y_pred, uncertainties, alpha=0.6, s=40, c=uncertainties, cmap='viridis')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Uncertainty')
        ax1.set_title('Prediction vs Uncertainty')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1)
        
        # 2. Uncertainty distribution
        ax2.hist(uncertainties, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=uncertainties.mean(), color='r', linestyle='--', 
                   label=f'Mean: {uncertainties.mean():.3f}')
        ax2.set_xlabel('Uncertainty')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Uncertainties')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Uncertainty percentiles
        percentiles = np.percentile(uncertainties, [10, 25, 50, 75, 90])
        ax3.bar(range(len(percentiles)), percentiles, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Percentile')
        ax3.set_ylabel('Uncertainty')
        ax3.set_title('Uncertainty Percentiles')
        ax3.set_xticks(range(len(percentiles)))
        ax3.set_xticklabels(['10th', '25th', '50th', '75th', '90th'])
        ax3.grid(True, alpha=0.3)
        
        # 4. High vs Low uncertainty predictions
        high_unc_idx = uncertainties > np.percentile(uncertainties, 75)
        low_unc_idx = uncertainties < np.percentile(uncertainties, 25)
        
        ax4.scatter(y_pred[low_unc_idx], uncertainties[low_unc_idx], 
                   alpha=0.6, s=40, color='green', label='Low Uncertainty')
        ax4.scatter(y_pred[high_unc_idx], uncertainties[high_unc_idx], 
                   alpha=0.6, s=40, color='red', label='High Uncertainty')
        ax4.set_xlabel('Predicted Values')
        ax4.set_ylabel('Uncertainty')
        ax4.set_title('High vs Low Uncertainty Predictions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_metrics_summary(self, metrics: Dict[str, float], 
                           save_path: Optional[Path] = None):
        """Create metrics summary visualization."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Key metrics bar chart
        key_metrics = ['r2_score', 'correlation', 'explained_variance']
        key_values = [metrics[k] for k in key_metrics if k in metrics]
        key_labels = [k.replace('_', ' ').title() for k in key_metrics if k in metrics]
        
        bars = ax1.bar(key_labels, key_values, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Score')
        ax1.set_title('Key Performance Metrics')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Color bars based on performance
        for bar, val in zip(bars, key_values):
            if val >= 0.7:
                bar.set_color('green')
            elif val >= 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add value annotations
        for bar, val in zip(bars, key_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Error metrics
        error_metrics = ['rmse', 'mae', 'mse']
        error_values = [metrics[k] for k in error_metrics if k in metrics]
        error_labels = [k.upper() for k in error_metrics if k in metrics]
        
        ax2.bar(error_labels, error_values, alpha=0.7, edgecolor='black', color='lightcoral')
        ax2.set_ylabel('Error')
        ax2.set_title('Error Metrics')
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance gauge (R²)
        r2 = metrics.get('r2_score', 0)
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        ax3.set_theta_zero_location('W')
        ax3.set_theta_direction(1)
        ax3.set_thetalim(0, np.pi)
        
        # Background gauge
        ax3.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
        
        # R² indicator
        r2_theta = r2 * np.pi
        ax3.fill_between(theta[theta <= r2_theta], 0, r, alpha=0.7, color='green' if r2 >= 0.7 else 'orange' if r2 >= 0.5 else 'red')
        
        ax3.set_title(f'R² Score: {r2:.3f}', pad=20)
        ax3.set_ylim(0, 1)
        ax3.set_rticks([])
        ax3.set_thetagrids([0, 45, 90, 135, 180], ['1.0', '0.75', '0.5', '0.25', '0.0'])
        
        # 4. Summary table
        ax4.axis('off')
        table_data = []
        for key, value in metrics.items():
            if isinstance(value, float):
                table_data.append([key.replace('_', ' ').title(), f'{value:.4f}'])
        
        table = ax4.table(cellText=table_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Detailed Metrics Summary', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
