"""
Quality Assessment for Synthetic Data

Evaluates the quality of synthetic data using multiple statistical
and domain-specific metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict


class QualityAssessor:
    """Quality assessor for synthetic LNP data."""
    
    def assess_quality(self, original_df: pd.DataFrame, 
                      synthetic_df: pd.DataFrame) -> Dict:
        """
        Comprehensive quality assessment for synthetic data.
        
        Args:
            original_df: Original dataset
            synthetic_df: Synthetic dataset
            
        Returns:
            Dictionary with quality metrics
        """
        numeric_cols = [col for col in original_df.select_dtypes(include=[np.number]).columns
                       if col in synthetic_df.columns and col != 'id']
        
        print(f"Assessing quality on {len(numeric_cols)} columns...")
        
        # 1. Distribution similarity
        distribution_scores = self._assess_distribution_similarity(
            original_df, synthetic_df, numeric_cols
        )
        
        # 2. Range coverage
        coverage_score = self._assess_range_coverage(
            original_df, synthetic_df, numeric_cols
        )
        
        # 3. Correlation preservation
        correlation_score = self._assess_correlation_preservation(
            original_df, synthetic_df, numeric_cols
        )
        
        # 4. Diversity score
        diversity_score = self._assess_diversity(synthetic_df)
        
        # Overall quality
        overall_quality = np.mean([
            distribution_scores['avg_pvalue'] * 0.3,
            coverage_score * 0.3,
            max(correlation_score, 0) * 0.2,
            diversity_score * 0.2
        ])
        
        quality_report = {
            'avg_statistical_pvalue': distribution_scores['avg_pvalue'],
            'range_coverage': coverage_score,
            'correlation_preservation': correlation_score,
            'diversity_score': diversity_score,
            'overall_quality': overall_quality,
            'num_features_tested': len(numeric_cols),
            'distribution_details': distribution_scores['details']
        }
        
        self._print_quality_summary(quality_report)
        return quality_report
    
    def _assess_distribution_similarity(self, original_df: pd.DataFrame,
                                      synthetic_df: pd.DataFrame,
                                      numeric_cols: list[str]) -> Dict:
        """Assess distribution similarity between original and synthetic data."""
        scores = []
        details = {}
        
        for col in numeric_cols:
            orig_vals = original_df[col].values
            synth_vals = synthetic_df[col].values
            
            # Mann-Whitney U test
            _, p_val = stats.mannwhitneyu(orig_vals, synth_vals, alternative='two-sided')
            
            # Effect size
            mean_diff = abs(np.mean(orig_vals) - np.mean(synth_vals)) / np.std(orig_vals)
            
            # Combined score
            if mean_diff < 0.3:
                score = max(p_val, 0.5)
            else:
                score = p_val
            
            scores.append(score)
            details[col] = {
                'p_value': p_val,
                'effect_size': mean_diff,
                'score': score
            }
            
            print(f"  {col}: MW p-val={p_val:.3f}, effect_size={mean_diff:.3f}, score={score:.3f}")
        
        return {
            'avg_pvalue': np.mean(scores) if scores else 0,
            'details': details
        }
    
    def _assess_range_coverage(self, original_df: pd.DataFrame,
                              synthetic_df: pd.DataFrame,
                              numeric_cols: list[str]) -> float:
        """Assess how well synthetic data covers the range of original data."""
        coverage_scores = []
        
        for col in numeric_cols:
            orig_range = original_df[col].max() - original_df[col].min()
            synth_range = synthetic_df[col].max() - synthetic_df[col].min()
            
            if orig_range > 0:
                coverage = min(synth_range / orig_range, 1.5)
                coverage_scores.append(min(coverage, 1.0))
            else:
                coverage_scores.append(1.0)
        
        return np.mean(coverage_scores)
    
    def _assess_correlation_preservation(self, original_df: pd.DataFrame,
                                       synthetic_df: pd.DataFrame,
                                       numeric_cols: list[str]) -> float:
        """Assess how well correlations are preserved."""
        if len(numeric_cols) <= 1:
            return 1.0
        
        orig_corr = original_df[numeric_cols].corr().values
        synth_corr = synthetic_df[numeric_cols].corr().values
        
        # Upper triangular mask
        mask = np.triu(np.ones_like(orig_corr, dtype=bool), k=1)
        orig_corr_vec = orig_corr[mask]
        synth_corr_vec = synth_corr[mask]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(orig_corr_vec) | np.isnan(synth_corr_vec))
        if valid_mask.sum() > 0:
            correlation = np.corrcoef(orig_corr_vec[valid_mask], synth_corr_vec[valid_mask])[0, 1]
            return correlation if not np.isnan(correlation) else 0
        
        return 0
    
    def _assess_diversity(self, synthetic_df: pd.DataFrame) -> float:
        """Assess diversity of synthetic samples."""
        numeric_df = synthetic_df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            unique_rows = len(pd.DataFrame(numeric_df.values).drop_duplicates())
            return min(unique_rows / len(synthetic_df), 1.0)
        return 0
    
    def _print_quality_summary(self, quality_report: Dict):
        """Print quality assessment summary."""
        print(f"\nQuality Assessment Summary:")
        print(f"  - Avg Statistical p-value: {quality_report['avg_statistical_pvalue']:.3f}")
        print(f"  - Range Coverage: {quality_report['range_coverage']:.3f}")
        print(f"  - Correlation Preservation: {quality_report['correlation_preservation']:.3f}")
        print(f"  - Diversity Score: {quality_report['diversity_score']:.3f}")
        print(f"  - Overall Quality: {quality_report['overall_quality']:.3f}")
