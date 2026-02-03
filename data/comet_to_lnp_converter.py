#!/usr/bin/env python3
"""
COMET → LNP Format Converter (generic IDs + updated component_library)
---------------------------------------------------------------------
• Converts COMET JSON to LNP format using generic component IDs everywhere
• Updates component_library.smiles_mapping to use generic IDs as keys
• Updates component_library.component_types to reference generic IDs

Usage: python comet_to_lnp_converter.py template.json comet.json -o output_dir
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

CATEGORY_MAP = {
    "IL": "ionizable_lipids",
    "HL": "helper_lipids",
    "CH": "helper_lipids",
    "PEG": "peg_lipids"
}

def load_json(path):
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as err:
        sys.exit(f"[ERROR] {path}: {err}")

def category_from_type(comp_type):
    return CATEGORY_MAP.get(comp_type, "unknown")

def build_global_id_map(comet_data):
    """
    Scan COMET data and assign unique generic IDs to each unique SMILES.
    Returns:
        id_map: dict mapping SMILES -> generic_id
        rev_map: dict mapping generic_id -> {'smiles': str, 'category': str}
    """
    counters = defaultdict(int)
    id_map, rev_map = {}, {}

    for form in comet_data.values():
        for comp in form["components"]:
            smi = comp["smi"]
            ctype = comp["component_type"]
            if smi in id_map:
                continue
            gen_id = f"{ctype}-{counters[ctype]}"
            counters[ctype] += 1
            id_map[smi] = gen_id
            rev_map[gen_id] = {
                "smiles": smi,
                "category": category_from_type(ctype)
            }

    return id_map, rev_map

def convert_one_formulation(form_id, form_data, label_key, id_map, rev_map):
    """Convert a single COMET formulation to LNP format."""
    total_mol = sum(c["mol"] for c in form_data["components"])
    components_out = {}

    for comp in form_data["components"]:
        smi = comp["smi"]
        gen_id = id_map[smi]
        pct = round(comp["mol"] / total_mol * 100, 1)

        components_out[gen_id] = {
            "composition_percent": pct,
            "smiles": smi,
            "category": rev_map[gen_id]["category"]
        }

    # Retain non-SMILES properties
    props_keep = [
        "volumetric_ratio", "il_rna_wt_ratio", "dataset_name", 
        "phase", "lipid_ratio", "actual_ilrna_wt_ratio", "NP_ratio"
    ]
    props_out = {k: form_data[k] for k in props_keep if k in form_data}

    return {
        "id": str(form_id),
        "components": components_out,
        "properties": props_out,
        "target": form_data["labels"][label_key]
    }

def build_component_library(rev_map, original_library):
    """
    Build updated component_library with generic IDs.
    """
    # Create smiles_mapping with generic IDs as keys
    smiles_mapping_updated = {
        gen_id: data["smiles"] 
        for gen_id, data in rev_map.items()
    }
    
    # Group generic IDs by category for component_types
    component_types_updated = defaultdict(list)
    for gen_id, data in rev_map.items():
        category = data["category"]
        component_types_updated[category].append(gen_id)
    
    # Sort each category list for consistent output
    for category in component_types_updated:
        component_types_updated[category].sort()
    
    return {
        "smiles_mapping": smiles_mapping_updated,
        "component_types": dict(component_types_updated)
    }

def convert_all(template_path, comet_path, out_dir):
    """Main conversion function."""
    tmpl = load_json(template_path)
    comet = load_json(comet_path)

    id_map, rev_map = build_global_id_map(comet)

    # Find all label types
    label_types = set()
    for f in comet.values():
        label_types.update(f.get("labels", {}))

    print(f"Found labels: {', '.join(label_types)}")
    print(f"Generated {len(rev_map)} unique component IDs:")
    for gen_id in sorted(rev_map.keys()):
        print(f"  {gen_id} -> {rev_map[gen_id]['smiles'][:50]}...")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for lbl in label_types:
        formulations = [
            convert_one_formulation(fid, fdat, lbl, id_map, rev_map)
            for fid, fdat in comet.items()
            if lbl in fdat.get("labels", {})
        ]

        # Build metadata
        meta = tmpl.get("metadata", {}).copy()
        meta["total_formulations"] = len(formulations)
        meta["components"] = sorted(rev_map.keys())

        # Build updated component_library
        updated_component_library = build_component_library(
            rev_map, 
            tmpl.get("component_library", {})
        )

        out_data = {
            "metadata": meta,
            "component_library": updated_component_library,
            "formulations": formulations
        }

        out_file = Path(out_dir) / f"{lbl}.json"
        with open(out_file, "w") as fh:
            json.dump(out_data, fh, indent=2)

        print(f"  → wrote {out_file} ({len(formulations)} formulations)")

def main():
    """Command line interface."""
    ap = argparse.ArgumentParser(
        description="Convert COMET JSON to LNP JSON with generic component IDs"
    )
    ap.add_argument("template", help="Template JSON file")
    ap.add_argument("comet_input", help="COMET-style input JSON file")
    ap.add_argument("-o", "--output_dir", default=".", 
                    help="Output directory (default: current)")
    
    args = ap.parse_args()
    convert_all(args.template, args.comet_input, args.output_dir)
    print("\nConversion completed!")

if __name__ == "__main__":
    main()
