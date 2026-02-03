"""
Molecular Featurization Utilities for LNP Components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import json
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from mordred import Calculator, descriptors as mordred_descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False

class MolecularFeaturizer:
    """Comprehensive molecular featurization for LNP components."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.descriptor_cache: Dict[str, Dict[str, float]] = {}
        self.available_descriptors: List[str] = [name for name, _ in Descriptors._descList]
        if MORDRED_AVAILABLE:
            self.mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES string to RDKit Mol object."""
        try:
            return Chem.MolFromSmiles(smiles)
        except:
            return None

    def calculate_rdkit_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate RDKit molecular descriptors."""
        if smiles in self.descriptor_cache:
            return self.descriptor_cache[smiles]

        mol = self.smiles_to_mol(smiles)
        descs: Dict[str, float] = {}
        if mol is None:
            descs = {name: 0.0 for name in self.available_descriptors}
        else:
            for name, func in Descriptors._descList:
                try:
                    descs[name] = float(func(mol))
                except:
                    descs[name] = 0.0

        self.descriptor_cache[smiles] = descs
        return descs

    def calculate_custom_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate custom descriptors relevant to LNP components."""
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return {}

        custom_desc: Dict[str, float] = {}
        try:
            custom_desc['mw'] = Descriptors.MolWt(mol)
            custom_desc['logp'] = Descriptors.MolLogP(mol)
            custom_desc['tpsa'] = Descriptors.TPSA(mol)
            custom_desc['hba'] = Descriptors.NumHAcceptors(mol)
            custom_desc['hbd'] = Descriptors.NumHDonors(mol)
            custom_desc['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            custom_desc['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            custom_desc['heavy_atoms'] = Descriptors.HeavyAtomCount(mol)
        except:
            for key in ['mw','logp','tpsa','hba','hbd','rotatable_bonds','aromatic_rings','heavy_atoms']:
                custom_desc.setdefault(key, 0.0)

        return custom_desc

    def featurize_component(self, smiles: str, component_name: str = "", 
                          include_mordred: bool = False) -> Dict[str, float]:
        """Comprehensive featurization of a single component."""
        features: Dict[str, float] = {}
        rdkit_desc = self.calculate_rdkit_descriptors(smiles)
        features.update({f'rdkit_{k}': v for k, v in rdkit_desc.items()})
        custom_desc = self.calculate_custom_descriptors(smiles)
        features.update({f'custom_{k}': v for k, v in custom_desc.items()})
        return features


class MultiComponentFeaturizer:
    """Featurization system for multi-component LNP formulations."""
    
    def __init__(self):
        self.mol_featurizer = MolecularFeaturizer()
        self.is_fitted = False
    
    def load_formulation_data(self, json_file: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Load formulation data and SMILES mapping."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        smiles_mapping = data['component_library']['smiles_mapping']
        
        # Convert to DataFrame
        rows = []
        for formulation in data['formulations']:
            row = {'id': formulation['id'], 'target': formulation['target']}
            
            # Add component compositions
            for comp_name, comp_data in formulation['components'].items():
                row[f'comp_{comp_name}'] = comp_data['composition_percent']
            
            # Add properties
            for prop, value in formulation['properties'].items():
                row[f'prop_{prop}'] = value
                
            rows.append(row)
        
        return pd.DataFrame(rows), smiles_mapping
    
    def create_component_features(self, smiles_mapping: Dict[str, str], 
                                include_mordred: bool = False) -> pd.DataFrame:
        """Create molecular feature matrix for all components."""
        component_features = {}
        
        for component, smiles in smiles_mapping.items():
            print(f"Featurizing component: {component}")
            features = self.mol_featurizer.featurize_component(
                smiles, component, include_mordred=include_mordred
            )
            component_features[component] = features
        
        # Convert to DataFrame
        feature_df = pd.DataFrame.from_dict(component_features, orient='index')
        feature_df.fillna(0, inplace=True)
        
        return feature_df
    
    def create_formulation_features(self, df: pd.DataFrame, component_features: pd.DataFrame) -> np.ndarray:
        """Create feature vectors for formulations using composition-weighted molecular descriptors."""
        formulation_features = []
        
        for idx, row in df.iterrows():
            # Get component columns
            comp_cols = [col for col in df.columns if col.startswith('comp_')]
            
            # Initialize formulation feature vector
            formulation_vector = np.zeros(component_features.shape[1])
            total_composition = 0
            
            # Weight molecular features by composition
            for comp_col in comp_cols:
                comp_name = comp_col.replace('comp_', '')
                composition = row[comp_col] if not pd.isna(row[comp_col]) else 0
                
                if composition > 0 and comp_name in component_features.index:
                    comp_features = component_features.loc[comp_name].values
                    formulation_vector += composition * comp_features / 100.0
                    total_composition += composition
            
            # Normalize by total composition
            if total_composition > 0:
                formulation_vector = formulation_vector / (total_composition / 100.0)
            
            formulation_features.append(formulation_vector)
        
        return np.array(formulation_features)
    
    def create_comprehensive_features(self, json_file: str, include_mordred: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create comprehensive feature matrix combining molecular and compositional features."""
        # Load data
        df, smiles_mapping = self.load_formulation_data(json_file)
        
        # Create component molecular features
        component_features = self.create_component_features(smiles_mapping, include_mordred)
        
        # Create composition-weighted molecular features
        molecular_features = self.create_formulation_features(df, component_features)
        
        # Add direct compositional features
        comp_cols = [col for col in df.columns if col.startswith('comp_')]
        composition_features = df[comp_cols].fillna(0).values
        
        # Add property features
        prop_cols = [col for col in df.columns if col.startswith('prop_')]
        property_features = df[prop_cols].fillna(0).values
        
        # Combine all features
        all_features = np.concatenate([
            molecular_features,
            composition_features,
            property_features
        ], axis=1)
        
        # Feature names
        feature_names = (
            [f'mol_{col}' for col in component_features.columns] +
            comp_cols +
            prop_cols
        )
        
        # Extract targets
        targets = df['target'].values
        
        print(f"Created feature matrix: {all_features.shape}")
        print(f"Number of samples: {len(targets)}")
        
        return all_features, targets, feature_names
