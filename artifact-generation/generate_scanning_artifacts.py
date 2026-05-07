import argparse
import cv2
import numpy as np
import random
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
from dataclasses import asdict
from typing import List, Tuple

# --- Robust Import Safety Net ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.utils import set_seed_from_string

# Note: Ensure scanning_artefacts.py is in the same directory or accessible in your path
try:
    from scanning_artefacts import ArtifactGenerator, BlurConfig, StitchConfig
except ImportError:
    print("Error: Could not import 'scanning_artefacts'. Ensure the module is in your directory.")
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
SELECTION_SEED = 42  # Global seed modifier for this specific augmentation
CROP_MARGIN = 20     # Artifacts entirely within this border will be rejected

def get_deterministic_blurs(size: Tuple[int, int]) -> List[BlurConfig]:
    """Generates random partial strip artifacts tailored for 300x300 @ 1mmp."""
    blurs = []
    w, h = size

    # 1 or 2 blurred regions per image
    num_blurs = random.randint(1, 2)

    for _ in range(num_blurs):
        # We use a retry loop. If the generated blur is entirely inside the 20px margin, 
        # it rejects it and rolls again (up to 20 times).
        for _attempt in range(20):
            orientation = random.choice(["vertical", "horizontal"])

            if orientation == "vertical":
                # STRIP WIDTH: 10% to 40% of image width (mimics a sensor band)
                bw = random.randint(int(w * 0.10), int(w * 0.70))
                # STRIP HEIGHT: 50% to 100% (Can end inside image or go full length)
                bh = random.randint(int(h * 0.50), h)
                
                # POSITION
                bx = random.randint(0, w - bw)
                by = random.randint(0, h - bh)  # Can start at top, or middle

            else:  # horizontal
                # STRIP HEIGHT: 10% to 40% of image height
                bh = random.randint(int(h * 0.10), int(h * 0.70))
                # STRIP WIDTH: 50% to 100%
                bw = random.randint(int(w * 0.50), w)

                # POSITION
                bx = random.randint(0, w - bw)
                by = random.randint(0, h - bh)

            # --- Margin Protection Check ---
            # If the blur's max dimension doesn't cross the margin, it's entirely inside it.
            entirely_left = (bx + bw <= CROP_MARGIN)
            entirely_right = (bx >= w - CROP_MARGIN)
            entirely_top = (by + bh <= CROP_MARGIN)
            entirely_bottom = (by >= h - CROP_MARGIN)

            if not (entirely_left or entirely_right or entirely_top or entirely_bottom):
                # INTENSITY: Kept low for 1mmp resolution
                b_int = random.choice([3, 5, 7])

                # FEATHERING: Soft edges for realistic optical focus loss
                min_dim = min(bw, bh)
                f_base = random.choice([int(min_dim * 0.2), int(min_dim * 0.4)])
                b_feather = f_base | 1

                blurs.append(BlurConfig(bx, by, bw, bh, b_int, b_feather))
                break # Success! Exit the retry loop

    return blurs

def get_deterministic_stitches(size: Tuple[int, int]) -> List[StitchConfig]:
    """
    Stitch Logic: 1 Line, Shift -5 to 5 px (Subtle 1MPP mechanical error).
    """
    stitches = []
    w, h = size

    orientation = random.choice(["vertical", "horizontal"])

    if orientation == "vertical":
        # Calculate safe bounds so the stitch line is NEVER in the 20px margin
        min_pos = max(CROP_MARGIN + 1, int(w * 0.2))
        max_pos = min(w - CROP_MARGIN - 1, int(w * 0.8))
        
        # Fallback if the image is bizarrely small
        if min_pos >= max_pos: 
            min_pos = max_pos = w // 2 
        
        pos = random.randint(min_pos, max_pos)

        # MAIN SHIFT (Horizontal): Reduced to 2-6 pixels (positive or negative)
        mag_x = random.randint(2, 6)
        shift_x = mag_x * random.choice([-1, 1])

        # MINOR SHIFT (Vertical sliding): Very slight, 0-2 pixels
        shift_y = random.randint(-2, 2)

        stitches.append(StitchConfig(pos, shift_x, shift_y, "vertical"))

    else:  # Horizontal
        # Calculate safe bounds so the stitch line is NEVER in the 20px margin
        min_pos = max(CROP_MARGIN + 1, int(h * 0.2))
        max_pos = min(h - CROP_MARGIN - 1, int(h * 0.8))
        
        if min_pos >= max_pos: 
            min_pos = max_pos = h // 2
        
        pos = random.randint(min_pos, max_pos)

        # MINOR SHIFT (Horizontal sliding): Very slight
        shift_x = random.randint(-2, 2)

        # MAIN SHIFT (Vertical): Reduced to 2-6 pixels
        mag_y = random.randint(2, 6)
        shift_y = mag_y * random.choice([-1, 1])

        stitches.append(StitchConfig(pos, shift_x, shift_y, "horizontal"))

    return stitches

def run_pipeline(input_dir: str, output_dir: str, json_path: str, artifact_type: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print(f"Loading manifest from: {json_path}")
    if not os.path.exists(json_path):
        print(f"Error: JSON file {json_path} does not exist.")
        sys.exit(1)

    with open(json_path, 'r') as f:
        external_manifest = json.load(f)
    
    files_to_process = external_manifest.get("tagged_files", {})
    print(f"Total Selected for Processing: {len(files_to_process)}")

    output_path.mkdir(parents=True, exist_ok=True)

    reproduction_log = {}
    engine = ArtifactGenerator()

    for rel_path_str in tqdm(files_to_process.keys(), desc=f"Processing {artifact_type.capitalize()}"):
        assigned_ratio = files_to_process[rel_path_str] if isinstance(files_to_process, dict) else None

        src_full_path = input_path / rel_path_str
        dst_full_path = output_path / rel_path_str
        
        if not src_full_path.exists():
            potential_paths = list(input_path.rglob(rel_path_str))
            if potential_paths:
                src_full_path = potential_paths[0]
            else:
                tqdm.write(f"[WARNING] File not found: {src_full_path}")
                continue

        dst_full_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(src_full_path))
        if img is None: 
            tqdm.write(f"[WARNING] Unreadable image: {src_full_path}")
            continue
        h, w = img.shape[:2]

        # Deterministic seed using our shared utility
        file_seed = f"{rel_path_str}_{SELECTION_SEED}"
        set_seed_from_string(file_seed)

        current_blurs = []
        current_stitches = []

        if artifact_type == "blur":
            current_blurs = get_deterministic_blurs((w, h))
        elif artifact_type == "stitch":
            current_stitches = get_deterministic_stitches((w, h))
        
        modified_img = engine.generate_patch(
            large_image=img,
            origin=(0, 0),
            output_size=(w, h),
            stitches=current_stitches,
            blurs=current_blurs,
        )

        cv2.imwrite(str(dst_full_path), modified_img)

        # Log exact configurations used for reproduction
        reproduction_log[rel_path_str] = {
            "entry_point_ratio": assigned_ratio,
            "artifact_type": artifact_type,
            "stitches": [asdict(s) for s in current_stitches],
            "blurs": [asdict(b) for b in current_blurs],
        }

    log_path = output_path / f"{artifact_type}_reproduction_log.json"
    with open(log_path, "w") as f:
        json.dump(reproduction_log, f, indent=2)

    print(f"Generation complete. Results and reproduction log saved in {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Procedurally generate Blur or Stitch Artifacts on Clean Tissue")
    parser.add_argument("--input_source", type=str, required=True, help="Path to the input folder of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument("--json_filter", type=str, required=True, help="Path to JSON file containing list of images to process")
    parser.add_argument("--artifact_type", type=str, required=True, choices=["blur", "stitch"], help="Type of artifact to generate ('blur' or 'stitch')")

    args = parser.parse_args()

    run_pipeline(args.input_source, args.output_dir, args.json_filter, args.artifact_type)

if __name__ == "__main__":
    main()