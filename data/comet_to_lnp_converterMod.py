#!/usr/bin/env python3
"""
COMET → LNP Format Converter  (uses generic component IDs)
----------------------------------------------------------
• Takes a COMET-style JSON file and a template file (only used
  for smiles_mapping/category lookup).
• Outputs one JSON file per label found in the COMET data.
• Every component gets a synthetic ID like “IL-0”, “HL-1”, etc.
  Those IDs are recorded in metadata["components"].
Usage
-----
python comet_to_lnp_converter.py template.json comet.json -o out_dir
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# ────────────────────────────────────────────────────────────── helpers


def load_json(path):
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as err:
        sys.exit(f"[ERROR] {path}: {err}")


CATEGORY_MAP = {"IL": "ionizable_lipids",
                "HL": "helper_lipids",
                "CH": "helper_lipids",
                "PEG": "peg_lipids"}


def category_from_type(comp_type):
    return CATEGORY_MAP.get(comp_type, "unknown")


# ────────────────────────────────────────────────── main converter logic


def build_global_id_map(comet_data):
    """
    Scan the whole COMET dict, assign a unique generic ID to each
    (component_type, exact_smiles) pair.
    Returns:
        id_map  dict  (smiles → gen_id)
        rev_map dict  (gen_id → {'smiles':…, 'category':…})
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
            rev_map[gen_id] = {"smiles": smi,
                               "category": category_from_type(ctype)}

    return id_map, rev_map


def convert_one_formulation(form_id, form_data, label_key,
                            id_map, rev_map):
    """
    Create a single formulation entry in the desired output format.
    """
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

    # retain non-SMILES properties the user requested
    props_keep = ["volumetric_ratio", "il_rna_wt_ratio",
                  "dataset_name", "phase", "lipid_ratio",
                  "actual_ilrna_wt_ratio", "NP_ratio"]

    props_out = {k: form_data[k] for k in props_keep if k in form_data}

    return {
        "id": str(form_id),
        "components": components_out,
        "properties": props_out,
        "target": form_data["labels"][label_key]
    }


def convert_all(template_path, comet_path, out_dir):
    tmpl = load_json(template_path)        # still useful for category list
    comet = load_json(comet_path)

    id_map, rev_map = build_global_id_map(comet)

    # ─── determine all label types ───────────────────────────
    label_types = set()
    for f in comet.values():
        label_types.update(f.get("labels", {}))

    print(f"Found labels: {', '.join(label_types)}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for lbl in label_types:
        formulations = [convert_one_formulation(fid, fdat, lbl,
                         id_map, rev_map)
                        for fid, fdat in comet.items()
                        if lbl in fdat.get("labels", {})]

        # metadata
        meta = tmpl.get("metadata", {}).copy()
        meta["total_formulations"] = len(formulations)
        meta["components"] = sorted(rev_map.keys())

        out_data = {
            "metadata": meta,
            # retain original component_library but it is optional
            "component_library": tmpl.get("component_library", {}),
            "formulations": formulations
        }

        out_file = Path(out_dir) / f"{lbl}.json"
        with open(out_file, "w") as fh:
            json.dump(out_data, fh, indent=2)

        print(f"  → wrote {out_file}  ({len(formulations)} formulations)")


# ────────────────────────────────────────────────────────── CLI wrapper


def main():
    ap = argparse.ArgumentParser(
        description="Convert COMET JSON to LNP JSON (generic IDs).")
    ap.add_argument("template", help="template JSON (desired structure)")
    ap.add_argument("comet_input", help="COMET-style JSON to convert")
    ap.add_argument("-o", "--output_dir", default=".",
                    help="directory for output files (default: current)")

    args = ap.parse_args()
    convert_all(args.template, args.comet_input, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
