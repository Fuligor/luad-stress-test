import argparse
import cv2
import numpy as np
import random
import sys
import os
import json
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

# --- Robust Import Safety Net ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.utils import set_seed_from_string

# --- Configuration ---
@dataclass
class DustConfig:
    dust_type: str  # 'smudge' or 'particulate'
    intensity: float  # 0.0-1.0 (Opacity multiplier)
    color: Tuple[int, int, int]  # BGR Color
    min_size: int  # Radius/Scale
    max_size: int  # Radius/Scale
    count: int  # Number of artifacts per image

class DustGenerator:
    """
    Advanced generator for creating textured, organic dust artifacts.
    """

    def apply_dust(
        self, image: np.ndarray, configs: List[DustConfig]
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        output = image.copy()
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for config in configs:
            if config.dust_type == "smudge":
                overlay, layer_mask = self._generate_textured_smudge(h, w, config)
            else:
                overlay, layer_mask = self._generate_particulate(h, w, config)

            # --- Feature: Tissue Defocus ---
            # If the mask takes up a large area, blur the underlying tissue.
            # This simulates the camera focusing on the dust (cover slip)
            # and the tissue (slide) going out of focus.
            mask_coverage = np.mean(layer_mask)
            if mask_coverage > 0.15:  # If >15% of image is covered by artifact intensity
                # Create a blurred version of the current state
                blur_k = random.choice([15, 21, 31])
                blurred_tissue = cv2.GaussianBlur(output, (blur_k, blur_k), 0)

                # Create a weight map for where to apply the blur.
                # We dilate the dust mask so the blur extends slightly beyond the dust itself.
                blur_weight = cv2.GaussianBlur(layer_mask, (51, 51), 0)
                blur_weight = np.clip(blur_weight * 1.5, 0, 1.0)
                blur_weight_3c = cv2.merge([blur_weight, blur_weight, blur_weight])

                # Blend blurred tissue with sharp tissue
                output = output.astype(np.float32)
                blurred_tissue = blurred_tissue.astype(np.float32)
                output = (
                    output * (1.0 - blur_weight_3c) + blurred_tissue * blur_weight_3c
                )
                output = np.clip(output, 0, 255).astype(np.uint8)

            # --- Compositing ---
            # Threshold low values to capture the faint edges of the smudge in the mask
            _, binary_layer_mask = cv2.threshold(
                layer_mask, 0.05, 255, cv2.THRESH_BINARY
            )
            combined_mask = cv2.bitwise_or(
                combined_mask, binary_layer_mask.astype(np.uint8)
            )

            # Blend
            alpha_3c = cv2.merge([layer_mask, layer_mask, layer_mask])
            output = output.astype(np.float32)
            overlay = overlay.astype(np.float32)

            # Standard Alpha Blend
            output = output * (1.0 - alpha_3c) + overlay * alpha_3c
            output = np.clip(output, 0, 255).astype(np.uint8)

        return output, combined_mask

    def _generate_textured_smudge(
        self, h: int, w: int, config: DustConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates organic, cloud-like smudges with grit texture and directional streaks.
        Includes logic for 'dense core' artifacts.
        """

        # --- Feature: Dense/Dark Artifact Mode ---
        # Sometimes (e.g. 20% chance), make the artifact much darker and more opaque
        is_dense = random.random() < 0.2

        current_color = config.color
        if is_dense:
            # Darken the color significantly - almost black/very dark brown
            current_color = tuple(max(0, c - 70) for c in current_color)

        # 1. Overlay Color
        overlay = np.full((h, w, 3), current_color, dtype=np.uint8)

        # 2. Mask Layer (Float)
        mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(config.count):
            artifact_created = False

            for attempt in range(10):
                # Temporary accumulator for THIS single artifact attempt
                this_artifact = np.zeros((h, w), dtype=np.float32)

                # Base size - Calculate first to ensure visibility
                r_base = random.randint(config.min_size, config.max_size)

                # --- Position & Orientation ---
                # Ensure the artifact overlaps the image.
                # Allow center to be off-screen up to (radius - buffer) to ensure at least some overlap
                margin = int(r_base * 0.6)
                cx = random.randint(-margin, w + margin)
                cy = random.randint(-margin, h + margin)

                # Decide on a "Smear Direction" for this artifact
                # This makes the blobs elongated (oval) and aligned, like a wipe mark.
                smear_angle = random.randint(0, 360)
                elongation_factor = random.uniform(
                    1.5, 3.0
                )  # How oval is it? (1.0 = circle, 3.0 = long cigar)

                # --- Feature: Dense Core Construction ---
                if is_dense:
                    # NEW LOGIC: Smooth Gradient Core
                    core_axes = (
                        int(r_base * 0.2),
                        int(r_base * 0.2 / elongation_factor),
                    )
                    temp_core = np.zeros((h, w), dtype=np.float32)

                    cv2.ellipse(
                        temp_core, (cx, cy), core_axes, smear_angle, 0, 360, 1.0, -1
                    )

                    # Apply massive blur to turn the shape into a gradient
                    sigma = r_base * 0.5
                    ksize = int(sigma * 4) | 1
                    temp_core = cv2.GaussianBlur(temp_core, (ksize, ksize), sigma)

                    # Normalize and Boost
                    c_max = temp_core.max()
                    if c_max > 0:
                        temp_core = (temp_core / c_max) * 1.3

                    this_artifact = cv2.add(this_artifact, temp_core)

                # --- Build Cloud from Puffs ---
                num_puffs = random.randint(15, 30)

                for i in range(num_puffs):
                    # Distribute puffs along the smear axis slightly more than across it
                    dist = random.gauss(0, r_base * 0.5)
                    # Offset perpendicular to smear
                    perp = random.gauss(0, r_base * 0.2)

                    # Convert smear-space to image-space (simple rotation math)
                    rad = np.deg2rad(smear_angle)
                    off_x = dist * np.cos(rad) - perp * np.sin(rad)
                    off_y = dist * np.sin(rad) + perp * np.cos(rad)

                    puff_center = (int(cx + off_x), int(cy + off_y))

                    # Elongated dimensions (OVALS)
                    major_axis = random.randint(int(r_base * 0.2), int(r_base * 0.5))
                    minor_axis = int(major_axis / elongation_factor)

                    # Jitter the angle slightly so they aren't perfectly parallel
                    puff_angle = smear_angle + random.randint(-15, 15)

                    # Puff Intensity
                    puff_intensity = random.uniform(0.1, 0.4)

                    # Draw Puff
                    temp_layer = np.zeros((h, w), dtype=np.float32)
                    cv2.ellipse(
                        temp_layer,
                        puff_center,
                        (major_axis, minor_axis),
                        puff_angle,
                        0,
                        360,
                        puff_intensity,
                        -1,
                    )
                    this_artifact = cv2.add(this_artifact, temp_layer)

                # --- Visibility Check ---
                # Calculate sum of pixels in this artifact to see if it exists onscreen
                if np.sum(this_artifact) > 100:  # Arbitrary threshold ensuring visibility
                    mask = cv2.add(mask, this_artifact)
                    artifact_created = True
                    break  # Success, exit retry loop

            if not artifact_created:
                print("Warning: Failed to generate visible artifact after retries.")

        # 3. Add "Grit" (Noise)
        # We generate a noise map and multiply it with the mask.
        # This breaks up the smooth gradients, making it look like particulate dust.
        noise = np.random.normal(1.0, 0.3, (h, w)).astype(np.float32)
        # Clip noise to avoid extreme bright spots or negatives
        noise = np.clip(noise, 0.5, 1.5)

        # Apply noise only where mask exists
        mask = mask * noise

        # 4. Blur
        # We use two blurs: one small one for the grit, one large one for the out-of-focus shape.
        # This preserves some of the "texture" while still looking blurry.

        # First, a slight blur to soften the sharp noise pixels
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Second, the heavy defocus blur
        blur_radius = int(config.max_size * 0.4) | 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

        # 5. Normalize
        mask = np.clip(mask, 0, 1.0)

        if is_dense:
            # For dense artifacts, ignore the requested transparency intensity
            # and force it to be nearly opaque (0.95-1.0) so tissue is hidden.
            mask = mask * random.uniform(0.85, 0.95)
        else:
            mask = mask * config.intensity

        return overlay, mask

    def _generate_particulate(
        self, h: int, w: int, config: DustConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates small, sharp, jagged debris.
        """
        overlay = np.full((h, w, 3), config.color, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(config.count):
            cx, cy = random.randint(0, w), random.randint(0, h)
            radius = random.randint(config.min_size, config.max_size)

            pts = []
            num_vertices = random.randint(5, 12)
            for _ in range(num_vertices):
                ang = random.uniform(0, 2 * np.pi)
                r = random.uniform(radius * 0.5, radius * 1.5)
                pts.append([cx + r * np.cos(ang), cy + r * np.sin(ang)])

            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1.0)

        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        return overlay, mask

# --- Pipeline Class ---
class DustAugmentor:
    def __init__(self, input_dir: str, output_dir: str, json_list_path: str = None):
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir)
        self.json_list_path = Path(json_list_path) if json_list_path else None
        self.generator = DustGenerator()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)

    def _get_random_configs(self):
        configs = []
        dirt_color = (
            random.randint(60, 100),
            random.randint(60, 100),
            random.randint(70, 110),
        )
        configs.append(
            DustConfig(
                dust_type="smudge",
                intensity=random.uniform(0.4, 0.8),
                color=dirt_color,
                min_size=120,
                max_size=300,
                count=random.randint(1, 2),
            )
        )
        return configs

    def process(self):
        """Processes images from the input folder based on the JSON list."""
        if not self.json_list_path or not self.json_list_path.exists():
            print("Error: JSON file not found.")
            return

        with open(self.json_list_path, "r") as f:
            data = json.load(f)
        files_to_process = list(data.get("tagged_files", {}).keys())

        for filename in tqdm(files_to_process, desc="Processing JSON"):
            src_path = self.input_dir / filename
            if not src_path.exists():
                continue
            image = cv2.imread(str(src_path))
            if image is None:
                continue

            # --- Set deterministic seed before ANY random calls ---
            set_seed_from_string(filename)
            
            # Now these will be deterministic per filename
            configs = self._get_random_configs()
            aug_img, mask = self.generator.apply_dust(image, configs)
            
            cv2.imwrite(str(self.output_dir / filename), aug_img)
            cv2.imwrite(str(self.output_dir / "masks" / filename), mask)

    def process_from_parquet(self, parquet_path):
        """Processes images directly from a Parquet file."""
        print(f"Loading parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        filtered_df = df[(df["source"] == "clean")]

        for _, row in tqdm(
            filtered_df.iterrows(), total=len(filtered_df), desc="Processing Parquet"
        ):
            filename = row["filename"]
            img_bytes = np.frombuffer(row["image"], np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                # --- Set deterministic seed before ANY random calls ---
                set_seed_from_string(filename)
                
                # Now these will be deterministic per filename
                configs = self._get_random_configs()
                aug_img, mask = self.generator.apply_dust(img, configs)
                
                cv2.imwrite(str(self.output_dir / filename), aug_img)
                cv2.imwrite(str(self.output_dir / "masks" / filename), mask)

def main():
    parser = argparse.ArgumentParser(description="Procedurally generate Dust Artifacts on Clean Tissue")
    parser.add_argument("--input_source", type=str, required=True, help="Path to Parquet file OR Folder of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument("--json_filter", type=str, default=None, help="Path to JSON file containing list of images to process (Required for Folder mode)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_source.endswith(".parquet"):
        augmentor = DustAugmentor(input_dir=None, output_dir=args.output_dir)
        augmentor.process_from_parquet(args.input_source)
    else:
        if not args.json_filter:
            print("Error: JSON mode requires a --json_filter to be provided when input_source is a directory.")
            sys.exit(1)
        augmentor = DustAugmentor(
            input_dir=args.input_source, 
            output_dir=args.output_dir, 
            json_list_path=args.json_filter
        )
        augmentor.process()
        
    print(f"Generation complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()