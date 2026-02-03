"""
Basic Tests for LNP Models
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.molecular_encoder import TunedMolecularEncoder
from models.multi_component_net import TunedMultiComponentNet
from models.regressor import TunedLNPRegressor


class TestLNPModels(unittest.TestCase):
    """Test suite for LNP models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'  # Use CPU for testing
        self.batch_size = 8
        self.molecular_feature_dim = 64
        self.n_components = 4
        self.property_dim = 4
        
    def test_molecular_encoder(self):
        """Test TunedMolecularEncoder."""
        encoder = TunedMolecularEncoder(
            input_dim=self.molecular_feature_dim,
            hidden_dims=[32, 16],
            output_dim=16,
            num_heads=2,
            dropout=0.1
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.molecular_feature_dim)
        output = encoder(x)
        
        self.assertEqual(output.shape, (self.batch_size, 16))
        self.assertFalse(torch.isnan(output).any())
    
    def test_multi_component_net(self):
        """Test TunedMultiComponentNet."""
        model = TunedMultiComponentNet(
            molecular_feature_dim=self.molecular_feature_dim,
            n_components=self.n_components,
            property_dim=self.property_dim,
            hidden_dims=[32, 16],
            num_heads=2,
            dropout_rate=0.1
        )
        
        # Create test inputs
        molecular_features = torch.randn(self.batch_size, self.n_components, self.molecular_feature_dim)
        compositions = torch.randn(self.batch_size, self.n_components)
        properties = torch.randn(self.batch_size, self.property_dim)
        
        # Test forward pass
        predictions, outputs = model(molecular_features, compositions, properties)
        
        self.assertEqual(predictions.shape, (self.batch_size,))
        self.assertFalse(torch.isnan(predictions).any())
        
        # Check auxiliary outputs
        self.assertIn('uncertainty', outputs)
        self.assertIn('feature_importance', outputs)
        self.assertIn('attention_weights', outputs)
        self.assertIn('individual_predictions', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['uncertainty'].shape, (self.batch_size,))
        self.assertEqual(outputs['feature_importance'].shape, (self.batch_size, self.n_components))
        self.assertEqual(outputs['attention_weights'].shape, (self.batch_size, self.n_components))
        self.assertEqual(len(outputs['individual_predictions']), 2)
    
    def test_regressor(self):
        """Test TunedLNPRegressor."""
        model_config = {
            'molecular_feature_dim': self.molecular_feature_dim,
            'n_components': self.n_components,
            'property_dim': self.property_dim,
            'hidden_dims': [32, 16],
            'num_heads': 2,
            'dropout_rate': 0.1
        }
        
        regressor = TunedLNPRegressor(model_config, device=self.device)
        
        # Create test data
        molecular_features = np.random.randn(self.batch_size, self.n_components, self.molecular_feature_dim)
        composition_features = np.random.randn(self.batch_size, self.n_components)
        property_features = np.random.randn(self.batch_size, self.property_dim)
        
        # Test prediction
        predictions, outputs = regressor.predict(molecular_features, composition_features, property_features)
        
        self.assertEqual(predictions.shape, (self.batch_size,))
        self.assertFalse(np.isnan(predictions).any())
    
    def test_model_parameter_count(self):
        """Test that models have reasonable parameter counts."""
        model_config = {
            'molecular_feature_dim': self.molecular_feature_dim,
            'n_components': self.n_components,
            'property_dim': self.property_dim,
            'hidden_dims': [128, 64],
            'num_heads': 4,
            'dropout_rate': 0.2
        }
        
        regressor = TunedLNPRegressor(model_config, device=self.device)
        param_count = sum(p.numel() for p in regressor.model.parameters())
        
        # Should have reasonable number of parameters
        self.assertGreater(param_count, 1000)  # At least 1K parameters
        self.assertLess(param_count, 1000000)  # Less than 1M parameters
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model_config = {
            'molecular_feature_dim': self.molecular_feature_dim,
            'n_components': self.n_components,
            'property_dim': self.property_dim,
            'hidden_dims': [32, 16],
            'num_heads': 2,
            'dropout_rate': 0.1
        }
        
        regressor = TunedLNPRegressor(model_config, device=self.device)
        
        # Create test data
        molecular_features = torch.randn(self.batch_size, self.n_components, self.molecular_feature_dim, requires_grad=True)
        composition_features = torch.randn(self.batch_size, self.n_components, requires_grad=True)
        property_features = torch.randn(self.batch_size, self.property_dim, requires_grad=True)
        targets = torch.randn(self.batch_size)
        
        # Forward pass
        predictions, _ = regressor.model(molecular_features, composition_features, property_features)
        
        # Compute loss and backward pass
        loss = torch.nn.MSELoss()(predictions, targets)
        loss.backward()
        
        # Check that gradients exist
        has_gradients = any(p.grad is not None for p in regressor.model.parameters())
        self.assertTrue(has_gradients)
        
        # Check that gradients are not zero
        total_grad_norm = sum(p.grad.norm().item() for p in regressor.model.parameters() if p.grad is not None)
        self.assertGreater(total_grad_norm, 0)


class TestModelUtilities(unittest.TestCase):
    """Test model utility functions."""
    
    def test_device_setup(self):
        """Test device setup function."""
        from utils.model_utils import setup_device
        
        device = setup_device()
        self.assertIn(device, ['cuda', 'cpu'])
    
    def test_parameter_counting(self):
        """Test parameter counting utility."""
        from utils.model_utils import count_model_parameters, get_model_summary
        
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        param_count = count_model_parameters(model)
        expected_params = (10 * 5 + 5) + (5 * 1 + 1)  # weights + biases
        self.assertEqual(param_count, expected_params)
        
        # Test model summary
        summary = get_model_summary(model)
        self.assertEqual(summary['total_parameters'], expected_params)
        self.assertEqual(summary['trainable_parameters'], expected_params)
        self.assertEqual(summary['non_trainable_parameters'], 0)


if __name__ == '__main__':
    unittest.main()
