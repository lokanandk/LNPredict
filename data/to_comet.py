#!/usr/bin/env python3
import argparse
import json
import copy
import os

def load_template(template_path):
    with open(template_path, "r") as f:
        return json.load(f)

def load_comet(comet_path):
    with open(comet_path, "r") as f:
        return json.load(f)

def build_smiles_lookup(component_library):
    """Create a SMILES → component_name lookup dict"""
    return {v: k for k, v in component_library["smiles_mapping"].items()}

def transform_formulation(fid, data, target_key, smiles_to_name, component_library):
    """Convert one COMET entry to target format"""
    formulation = {
        "id": f"CH-{fid}",
        "components": {},
        "properties": {},  # placeholder if you want to add size, PDI, etc.
        "target": data["labels"][target_key]
    }

    total_mol = sum(c["mol"] for c in data["components"])
    if total_mol == 0:
        total_mol = 1  # avoid division by zero

    for comp in data["components"]:
        smi = comp["smi"]
        mol = comp["mol"]
        comp_name = smiles_to_name.get(smi, comp["component_type"])  # fallback
        
        # find category
        category = None
        for cat, members in component_library["component_types"].items():
            if comp_name in members:
                category = cat
                break
        
        formulation["components"][comp_name] = {
            "composition_percent": round((mol / total_mol) * 100, 2),
            "smiles": smi,
            "category": category if category else comp["component_type"]
        }

    return formulation

def convert(template_json, comet_json, output_prefix):
    component_library = template_json["component_library"]
    metadata_template = template_json["metadata"]

    smiles_to_name = build_smiles_lookup(component_library)

    # collect all unique label names
    all_labels = set()
    for v in comet_json.values():
        all_labels.update(v["labels"].keys())

    # make output dir if needed
    os.makedirs("converted_jsons", exist_ok=True)

    for label in all_labels:
        out = {
            "metadata": copy.deepcopy(metadata_template),
            "component_library": component_library,
            "formulations": []
        }

        for fid, entry in comet_json.items():
            if label in entry["labels"]:
                f = transform_formulation(fid, entry, label, smiles_to_name, component_library)
                out["formulations"].append(f)

        out["metadata"]["total_formulations"] = len(out["formulations"])

        outpath = os.path.join("converted_jsons", f"{output_prefix}_{label}.json")
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)

        print(f"✅ Wrote {outpath}")

def main():
    parser = argparse.ArgumentParser(description="Convert COMET JSON to template JSON format.")
    parser.add_argument("template", help="Path to template JSON (desired format)")
    parser.add_argument("comet", help="Path to COMET JSON (input data)")
    parser.add_argument("--prefix", default="converted", help="Output prefix for JSON files")
    args = parser.parse_args()

    template_json = load_template(args.template)
    comet_json = load_comet(args.comet)

    convert(template_json, comet_json, args.prefix)

if __name__ == "__main__":
    main()
