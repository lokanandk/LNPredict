"""
Utility functions for synthetic data generation.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict


def load_lnp_data(json_file: str) -> pd.DataFrame:
    """Load and preprocess LNP data from JSON format."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    rows = []
    all_components = list(data['component_library']['smiles_mapping'].keys())
    
    for formulation in data['formulations']:
        row = {'id': formulation['id'], 'target': formulation['target']}
        
        # Add composition features
        for comp in all_components:
            row[f'comp_{comp}'] = formulation['components'].get(comp, {}).get('composition_percent', 0.0)
        
        # Add property features
        for prop, value in formulation['properties'].items():
            row[f'prop_{prop}'] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows).fillna(0)


def save_synthetic_data(synthetic_df: pd.DataFrame, original_json: str, 
                       output_file: str, quality_report: Dict, random_state: int):
    """Save synthetic data in LNP JSON format with quality report."""
    with open(original_json, 'r') as f:
        original_data = json.load(f)
    
    output_data = {
        "metadata": {
            "version": "1.0",
            "description": "LNPredict synthetic LNP dataset",
            "generation_method": "targeted_augmentation",
            "synthetic_formulations": len(synthetic_df),
            "quality_metrics": quality_report,
            "random_state": random_state
        },
        "component_library": original_data['component_library'],
        "formulations": []
    }
    
    for _, row in synthetic_df.iterrows():
        formulation = {
            "id": row['id'],
            "target": float(row['target']),
            "components": {},
            "properties": {}
        }
        
        # Add components
        for col in synthetic_df.columns:
            if col.startswith('comp_') and row[col] > 0.01:
                comp_name = col.replace('comp_', '')
                if comp_name in original_data['component_library']['smiles_mapping']:
                    formulation["components"][comp_name] = {
                        "composition_percent": float(row[col]),
                        "smiles": original_data['component_library']['smiles_mapping'][comp_name]
                    }
        
        # Add properties
        for col in synthetic_df.columns:
            if col.startswith('prop_'):
                prop_name = col.replace('prop_', '')
                formulation["properties"][prop_name] = float(row[col])
        
        output_data["formulations"].append(formulation)
    
    # Save synthetic data
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save quality report
    report_file = output_file.replace('.json', '_quality_report.json')
    with open(report_file, 'w') as f:
        json.dump(quality_report, f, indent=2)
    
    print(f"\n✅ Files saved:")
    print(f"  - Synthetic dataset: {output_file}")
    print(f"  - Quality report: {report_file}")
