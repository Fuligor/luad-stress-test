import argparse
import os
import sys
import json
import random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Robust Import Safety Net ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from utils.utils import set_seed_from_string

def subtractive_blend(image, mask, ink_color_bgr, alpha_map):
    """
    Applies color using a subtractive model.
    Result = Image * (1 - Alpha * (1 - InkColor))
    This ensures the ink always acts as a darken filter.
    """
    # Normalize image and ink color to 0.0-1.0 range
    img_norm = image.astype(np.float32) / 255.0
    ink_norm = ink_color_bgr / 255.0

    # Expand alpha map to 3 channels
    alpha_3c = np.stack([alpha_map] * 3, axis=-1)

    # Subtractive blending formula
    # The ink filters the light: more ink = less original light passes through
    transmission_factor = 1.0 - (alpha_3c * (1.0 - ink_norm))
    result_norm = img_norm * transmission_factor

    return np.clip(result_norm * 255.0, 0, 255).astype(np.uint8)

def generate_marker_artifact(image, identifier):
    # Fix seed based on image identifier (filename) using shared utility
    set_seed_from_string(identifier)

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # --- 1. Heavy Ink Logic ---
    # 30% chance for a dense, heavy marker
    heavy_mode = random.random() < 0.3

    if heavy_mode:
        # Darker, more opaque pigments
        color_anchors = {
            "deep_forest": (40, 60, 20),   # Very dark green
            "charcoal": (30, 30, 35),      # Almost black
            "dark_magenta": (60, 0, 180),  # Deep purple-red
        }
        max_alpha = random.uniform(0.92, 0.99)  # Nearly opaque core
        alpha_power = 0.3  # Broader, denser center
        heavy_pigment_factor = random.uniform(0.5, 0.7)  # Significantly darken the color
    else:
        # Standard markers
        color_anchors = {
            "pink_red": (88, 0, 235),
            "muted_green": (98, 116, 0),
            "dark_grey": (53, 45, 54),
        }
        max_alpha = random.uniform(0.6, 0.85)
        alpha_power = 0.6
        heavy_pigment_factor = 1.0

    # --- 2. Color Selection & Jitter ---
    choice = random.choice(list(color_anchors.keys()))
    base_bgr = np.array(color_anchors[choice], dtype=np.float32)

    # Apply heavy pigment darkening
    base_bgr *= heavy_pigment_factor

    # Add random channel jitter
    marker_color_bgr = np.clip(base_bgr + np.random.normal(0, 15, 3), 0, 255).astype(np.float32)

    # --- 3. Geometry Generation (Edge-Anchored) ---
    origin = random.choice(["top", "bottom", "left", "right", "corner"] * 2 + ["double_corner"])

    pts_list = []
    if origin == "double_corner":
        # Creates two separate marker strokes in opposite corners
        pts_list.append(np.array([[0, 0], [w // 3, 0], [0, h // 3]], np.int32))
        pts_list.append(np.array([[w, h], [w - w // 3, h], [w, h - h // 3]], np.int32))
    elif origin == "corner":
        pts_list.append(
            np.array([
                [0, 0],
                [int(w * random.uniform(0.4, 0.9)), 0],
                [0, int(h * random.uniform(0.4, 0.9))],
            ], np.int32)
        )
    else:
        if origin in ["left", "right"]:
            x_f = 0 if origin == "left" else w
            x_v = int(w * random.uniform(0.3, 0.8))
            x_v = x_v if origin == "left" else w - x_v
            pts_list.append(
                np.array([
                    [x_f, 0], [x_v, 0], [x_v + random.randint(-40, 40), h], [x_f, h]
                ], np.int32)
            )
        else:
            y_f = 0 if origin == "top" else h
            y_v = int(h * random.uniform(0.3, 0.8))
            y_v = y_v if origin == "top" else h - y_v
            pts_list.append(
                np.array([
                    [0, y_f], [0, y_v], [w, y_v + random.randint(-40, 40)], [w, y_f]
                ], np.int32)
            )

    cv2.fillPoly(mask, pts_list, 255)

    # --- 4. Alpha Map Generation ---
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_map, dist_map, 0, 1.0, cv2.NORM_MINMAX)

    # Less blur in heavy mode for sharper edges
    blur_kernel = (31, 31) if heavy_mode else (51, 51)
    bleed_mask = cv2.GaussianBlur(mask.astype(np.float32), blur_kernel, 0) / 255.0

    # Combine distance, power, and bleed to get final alpha
    alpha_map = np.power(dist_map, alpha_power) * bleed_mask * max_alpha
    np.clip(alpha_map, 0, 1.0, out=alpha_map)  # Ensure valid range

    # --- 5. Apply Subtractive Blending ---
    final_img = subtractive_blend(image, mask, marker_color_bgr, alpha_map)

    return final_img, (mask > 0).astype(np.uint8) * 255

def process_from_json(input_dir, output_dir, json_path):
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    tagged_files = data.get("tagged_files", {})
    
    for filename in tqdm(tagged_files.keys(), desc="Processing JSON", unit="img"):
        in_p = os.path.join(input_dir, filename)
        if not os.path.exists(in_p):
            continue

        img = cv2.imread(in_p)
        if img is None:
            continue

        marked, mask = generate_marker_artifact(img, filename)

        cv2.imwrite(os.path.join(output_dir, filename), marked)
        cv2.imwrite(os.path.join(mask_dir, filename), mask)

def process_from_parquet(parquet_path, output_dir):
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    filtered_df = df[(df["source"] == "clean")]

    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing Parquet", unit="img"):
        filename = row["filename"]
        img_bytes = np.frombuffer(row["image"], np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is not None:
            marked, mask = generate_marker_artifact(img, filename)
            cv2.imwrite(os.path.join(output_dir, filename), marked)
            cv2.imwrite(os.path.join(mask_dir, filename), mask)

def main():
    parser = argparse.ArgumentParser(description="Procedurally generate Marker Artifacts on Clean Tissue")
    parser.add_argument("--input_source", type=str, required=True, help="Path to Parquet file OR Folder of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument("--json_filter", type=str, default=None, help="Path to JSON file containing list of images to process (Required for Folder mode)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_source.endswith(".parquet"):
        process_from_parquet(args.input_source, args.output_dir)
    else:
        if not args.json_filter:
            print("Error: JSON mode requires a --json_filter to be provided when input_source is a directory.")
            sys.exit(1)
        process_from_json(args.input_source, args.output_dir, args.json_filter)
        
    print(f"Generation complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()