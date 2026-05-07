import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

# ================= CONFIGURATION =================
# Default configuration if none is provided via CLI
DEFAULT_ARTIFACTS = {
    "marker": 1.0,
    "prec": 1.0,
    "dust": 1.0,
    "fold": 1.0,
    "blur": 1.0,
    "stitch": 1.0,
}
# =================================================

def generate_all_manifests(source_path, output_path, artifacts_config):
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        print(f"Error: Source directory not found at {source_path}")
        return

    os.makedirs(output_path, exist_ok=True)

    # 1. Gather valid images
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif")
    all_files = sorted(
        [
            f
            for f in os.listdir(source_path)
            if f.lower().endswith(valid_extensions) and os.path.isfile(source_path / f)
        ]
    )

    actual_count = len(all_files)
    total_support = actual_count
    results_summary = []

    print(f"--- Found {actual_count} physical files in folder ---")
    print(f"--- Calculations based on FIXED SUPPORT: {total_support} ---\n")

    # 2. Loop through each artifact
    for name, percentage in artifacts_config.items():
        # Deterministic seed based on artifact name
        seed_value = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (10**8)
        random.seed(seed_value)

        # USE ROUNDING: e.g., 3662 * 0.3 = 1098.6 -> 1099
        num_to_select_target = int(round(total_support * percentage))

        # Safety check against physical folder limits
        final_selection_count = min(num_to_select_target, actual_count)
        selected_files = random.sample(all_files, final_selection_count)

        # Filename formatting
        perc_str = str(percentage).replace(".", "")
        output_filename = f"manifest_stress_{name}_{perc_str}.json"
        out_file_path = output_path / output_filename

        # 3. Build JSON
        manifest_data = {
            "artifact_type": name,
            "total_support": total_support,
            "selected_counts": final_selection_count,
            "selection_percentage": percentage,
            "tagged_files": {filename: percentage for filename in selected_files},
        }

        # 4. Save
        with open(out_file_path, "w") as f:
            json.dump(manifest_data, f, indent=4)

        results_summary.append(
            {
                "name": name,
                "target": num_to_select_target,
                "actual": final_selection_count,
                "status": "OK" if num_to_select_target <= actual_count else "CAPPED",
            }
        )

    # 5. Final Summary Table
    print("-" * 55)
    print(f"{'Artifact':<10} | {'Target':<10} | {'Actual':<10} | {'Status'}")
    print("-" * 55)
    for res in results_summary:
        print(
            f"{res['name']:<10} | {res['target']:<10} | {res['actual']:<10} | {res['status']}"
        )
    print("-" * 55)
    print(f"Manifests saved successfully to {output_path}")

def parse_artifacts_arg(arg_str):
    if not arg_str:
        return DEFAULT_ARTIFACTS
    
    custom_artifacts = {}
    pairs = arg_str.split()
    for pair in pairs:
        try:
            name, pct = pair.split(':')
            custom_artifacts[name] = float(pct)
        except ValueError:
            print(f"Error parsing artifact config '{pair}'. Expected format 'name:fraction' (e.g., 'dust:0.5').")
            sys.exit(1)
    return custom_artifacts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic JSON manifests for artifact generation.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the source directory containing clean images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the generated JSON manifests.")
    parser.add_argument("--artifacts", type=str, default="", help="Optional. Space-separated list of artifact:fraction pairs. Example: 'dust:1.0 blur:0.5'. Defaults to all artifacts at 1.0.")
    
    args = parser.parse_args()
    
    artifacts_config = parse_artifacts_arg(args.artifacts)
    generate_all_manifests(args.input_dir, args.output_dir, artifacts_config)