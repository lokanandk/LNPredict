"""
Enhanced Multi-Component Network with Batch Normalization and Deeper Architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TunedMultiComponentNet(nn.Module):
    """Enhanced multi-component network with batch normalization for breakthrough performance."""
    
    def __init__(self, molecular_feature_dim, n_components, property_dim,
                 hidden_dims=[256, 128, 64], num_heads=8, dropout_rate=0.2):
        super().__init__()
        
        self.n_components = n_components
        output_dim = hidden_dims[-1]
        
        # 🔥 ENHANCED MOLECULAR ENCODER WITH BATCH NORMALIZATION
        self.molecular_encoder = nn.Sequential(
            nn.Linear(molecular_feature_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dims[1], output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # Component attention with proper head calculation
        attn_heads = min(num_heads, output_dim)
        if output_dim % attn_heads != 0:
            attn_heads = self._find_divisor(output_dim, num_heads)
        
        # Cross-component attention fusion
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=output_dim, 
            num_heads=attn_heads, 
            batch_first=True,
            dropout=dropout_rate
        )
        
        # 🔥 ENHANCED PROPERTY ENCODER WITH BATCH NORMALIZATION
        self.property_encoder = nn.Sequential(
            nn.Linear(property_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(hidden_dims[1], output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # Compositional weighting network (enhanced)
        self.comp_weighting = nn.Sequential(
            nn.Linear(n_components, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dims[1], n_components),
            nn.Softmax(dim=1)
        )
        
        # 🔥 ENHANCED FUSION WITH BATCH NORMALIZATION
        fusion_dim = output_dim + output_dim + n_components
        
        # Main fusion path (wider and deeper)
        self.fusion_main = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[0] * 2),  # Wider first layer
            nn.BatchNorm1d(hidden_dims[0] * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dims[0] * 2, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(hidden_dims[0], output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # Auxiliary fusion path (for ensemble behavior)
        self.fusion_aux = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dims[0], output_dim // 2),
            nn.BatchNorm1d(output_dim // 2),
            nn.ReLU()
        )
        
        # 🔥 DUAL PREDICTION HEADS (CRITICAL FOR ENSEMBLE)
        self.prediction_head1 = nn.Linear(output_dim, 1)
        self.prediction_head2 = nn.Linear(output_dim // 2, 1)
        
        # Enhanced uncertainty and confidence heads
        self.uncertainty_head = nn.Sequential(
            nn.Linear(output_dim, 32),  # Wider
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Feature importance predictor
        self.importance_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], n_components),
            nn.Softmax(dim=1)
        )
        
        param_count = sum(p.numel() for p in self.parameters())
        print(f"  Created enhanced model: {param_count:,} parameters")
    
    def _find_divisor(self, n, target):
        """Find largest divisor of n that's <= target."""
        for i in range(min(target, n), 0, -1):
            if n % i == 0:
                return i
        return 1
    
    def forward(self, molecular_features, compositions, properties):
        molecular_features = F.layer_norm(molecular_features, molecular_features.shape[1:])
        batch_size = molecular_features.shape[0]
        
        # Encode molecular features
        mol_flat = molecular_features.reshape(-1, molecular_features.size(-1))
        mol_encoded = self.molecular_encoder(mol_flat)
        mol_encoded = mol_encoded.reshape(batch_size, self.n_components, -1)
        
        # Enhanced attention fusion
        attended_mol, attention_weights = self.attention_fusion(
            mol_encoded, mol_encoded, mol_encoded
        )
        
        # Smart compositional weighting
        enhanced_comp_weights = self.comp_weighting(compositions)
        final_mol = (attended_mol * enhanced_comp_weights.unsqueeze(-1)).sum(dim=1)
        
        # Enhanced property encoding
        prop_encoded = self.property_encoder(properties)
        
        # Multi-path fusion
        combined_features = torch.cat([final_mol, prop_encoded, compositions], dim=1)
        
        main_features = self.fusion_main(combined_features)
        aux_features = self.fusion_aux(combined_features)
        
        # Dual predictions with ensemble weighting
        pred1 = self.prediction_head1(main_features)
        pred2 = self.prediction_head2(aux_features)
        
        # Enhanced ensemble prediction
        ensemble_weight = torch.sigmoid(pred1.detach())
        final_prediction = ensemble_weight * pred1 + (1 - ensemble_weight) * pred2
        
        # Auxiliary outputs
        uncertainty = self.uncertainty_head(main_features)
        importance = self.importance_head(main_features)
        
        return final_prediction.squeeze(-1), {
            'uncertainty': uncertainty.squeeze(-1),
            'feature_importance': importance,
            'attention_weights': enhanced_comp_weights,
            'individual_predictions': [pred1.squeeze(-1), pred2.squeeze(-1)]
        }
