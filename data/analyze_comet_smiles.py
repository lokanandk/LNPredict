#!/usr/bin/env python3
"""
COMET SMILES Analyzer
Analyzes SMILES strings in COMET data to help create better template mappings.
"""

import json
import argparse
from collections import defaultdict

def analyze_comet_smiles(comet_path):
    """Analyze SMILES strings in COMET data."""

    with open(comet_path, 'r') as f:
        comet_data = json.load(f)

    smiles_by_type = defaultdict(set)
    smiles_frequency = defaultdict(int)

    print("=== COMET SMILES Analysis ===\n")

    # Collect all SMILES by component type
    for formulation_id, formulation in comet_data.items():
        for component in formulation.get("components", []):
            smiles = component["smi"]
            component_type = component["component_type"]

            smiles_by_type[component_type].add(smiles)
            smiles_frequency[smiles] += 1

    # Display results
    for component_type, smiles_set in smiles_by_type.items():
        print(f"Component Type: {component_type}")
        print("-" * 40)

        for smiles in sorted(smiles_set):
            frequency = smiles_frequency[smiles]
            print(f"  Frequency: {frequency:2d} | {smiles}")
        print()

    # Create a template mapping suggestion
    print("=== Suggested Template Mapping ===\n")
    print('    "smiles_mapping": {')

    counter_by_type = defaultdict(int)
    for component_type, smiles_set in smiles_by_type.items():
        type_prefix = component_type

        for smiles in sorted(smiles_set):
            frequency = smiles_frequency[smiles]
            component_name = f"{type_prefix}-{counter_by_type[component_type]}"
            counter_by_type[component_type] += 1

            print(f'      "{smiles}": "{component_name}",  # freq: {frequency}')

    print('    }')

    return smiles_by_type, smiles_frequency

def main():
    parser = argparse.ArgumentParser(description="Analyze SMILES in COMET data")
    parser.add_argument("comet_file", help="COMET JSON file to analyze")

    args = parser.parse_args()
    analyze_comet_smiles(args.comet_file)

if __name__ == "__main__":
    main()
