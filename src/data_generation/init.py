"""
LNPredict Synthetic Data Generation Module

Provides targeted augmentation capabilities for small LNP datasets using
multiple intelligent strategies including composition interpolation, 
nearest neighbor augmentation, and perturbation-based generation.
"""

from .generator import TargetedAugmentationGenerator
from .quality.assessor import QualityAssessor
from .utils import load_lnp_data, save_synthetic_data

__version__ = "1.0.0"
__all__ = [
    "TargetedAugmentationGenerator",
    "QualityAssessor", 
    "load_lnp_data",
    "save_synthetic_data"
]
