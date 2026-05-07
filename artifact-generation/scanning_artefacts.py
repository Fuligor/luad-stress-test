import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

MAX_EXPECTED_SHIFT = 30


# --- Configuration Classes ---
@dataclass
class BlurConfig:
    x: int
    y: int
    width: int
    height: int
    intensity: int  # The strength of the optical blur (kernel size)
    feathering: int  # How smooth the edge of the blur area is (kernel size)


@dataclass
class StitchConfig:
    position: int
    shift_x: int  # Pixels to shift horizontally at this line
    shift_y: int  # Pixels to shift vertically at this line
    orientation: str  # 'vertical' or 'horizontal'


# --- Core Generator Class ---
class ArtifactGenerator:
    def generate_patch(
        self,
        large_image: np.ndarray,
        origin: Tuple[int, int],
        output_size: Tuple[int, int],
        stitches: List[StitchConfig] = [],
        blurs: List[BlurConfig] = [],
    ) -> np.ndarray:
        # --- 1. SETUP PADDING & BUFFERS ---
        # We need two types of padding:
        # A. 'pad': The area where stitches/shifts are active (visible area + stitch margin)
        # B. 'safety': Extra buffer source data so we NEVER run out of pixels when shifting

        pad = 0
        safety = 0

        if stitches:
            pad = 50  # Space for the stitch logic to operate
            safety = 100  # Extra "fuel" for the shift so we don't hit black pixels

            # Create a source image that is MUCH larger than needed
            # We use REFLECT_101 to generate believable "fake" tissue data
            working_img = cv2.copyMakeBorder(
                large_image,
                pad + safety,
                pad + safety,
                pad + safety,
                pad + safety,
                cv2.BORDER_REFLECT_101,
            )
        else:
            working_img = large_image.copy()

        # The canvas we paint onto corresponds to the 'pad' size, not the full 'safety' size
        # Dimensions: Original + 2*pad
        work_h = large_image.shape[0] + 2 * pad
        work_w = large_image.shape[1] + 2 * pad

        # Initialize Canvas
        if len(working_img.shape) == 3:
            output = np.zeros(
                (work_h, work_w, working_img.shape[2]), dtype=working_img.dtype
            )
        else:
            output = np.zeros((work_h, work_w), dtype=working_img.dtype)

        # --- 2. PROCESS STITCHES ---
        # Sort stitches
        v_stitches = sorted(
            [s for s in stitches if s.orientation == "vertical"],
            key=lambda s: s.position,
        )
        h_stitches = sorted(
            [s for s in stitches if s.orientation == "horizontal"],
            key=lambda s: s.position,
        )

        # Adjust stitch positions to align with our 'output' canvas (which has 'pad' border)
        adj_v_stitches = [
            StitchConfig(s.position + pad, s.shift_x, s.shift_y, s.orientation)
            for s in v_stitches
        ]
        adj_h_stitches = [
            StitchConfig(s.position + pad, s.shift_x, s.shift_y, s.orientation)
            for s in h_stitches
        ]

        # Define grid based on 'output' dimensions
        x_bounds = [0] + [s.position for s in adj_v_stitches] + [work_w]
        y_bounds = [0] + [s.position for s in adj_h_stitches] + [work_h]

        for i in range(len(y_bounds) - 1):
            y0, y1 = y_bounds[i], y_bounds[i + 1]
            if y0 >= y1:
                continue

            acc_shift_x_h = sum(s.shift_x for s in adj_h_stitches[:i]) if i > 0 else 0
            acc_shift_y_h = sum(s.shift_y for s in adj_h_stitches[:i]) if i > 0 else 0

            for j in range(len(x_bounds) - 1):
                x0, x1 = x_bounds[j], x_bounds[j + 1]
                if x0 >= x1:
                    continue

                acc_shift_x_v = (
                    sum(s.shift_x for s in adj_v_stitches[:j]) if j > 0 else 0
                )
                acc_shift_y_v = (
                    sum(s.shift_y for s in adj_v_stitches[:j]) if j > 0 else 0
                )

                # --- COORDINATE MAPPING ---
                # Target: We want to fill output[y0:y1, x0:x1]
                # Source: We calculate the shifted coordinates

                # Base Source (relative to output canvas 0,0)
                src_x_base = x0 + acc_shift_x_h + acc_shift_x_v
                src_y_base = y0 + acc_shift_y_h + acc_shift_y_v

                # Safety Offset: Map to the heavily padded 'working_img'
                # working_img starts at -safety relative to output
                src_x = src_x_base + safety
                src_y = src_y_base + safety

                cell_w = x1 - x0
                cell_h = y1 - y0

                # Robust Bounds Check
                # Even with safety, we check to be nice, but this should effectively never fail now.
                src_h, src_w = working_img.shape[:2]

                if (
                    src_x < 0
                    or src_x + cell_w > src_w
                    or src_y < 0
                    or src_y + cell_h > src_h
                ):
                    print(
                        f"Warning: Shift out of bounds despite safety buffer! ({src_x}, {src_y})"
                    )
                    continue

                output[y0:y1, x0:x1] = working_img[
                    src_y : src_y + cell_h, src_x : src_x + cell_w
                ]

        # --- 3. CROP BACK TO ORIGINAL SIZE ---
        if pad > 0:
            final_output = output[
                pad : pad + output_size[1], pad : pad + output_size[0]
            ]
        else:
            final_output = output

        # --- 4. APPLY BLURS ---
        for blur in blurs:
            final_output = self._apply_smooth_blur(final_output, blur)

        return final_output

    def _apply_smooth_blur(self, image: np.ndarray, config: BlurConfig) -> np.ndarray:
        """
        Applies blur blended smoothly using a feathered mask.
        """
        h, w = image.shape[:2]

        # 1. Create the blurry version of the whole image
        k_intensity = (
            config.intensity if config.intensity % 2 == 1 else config.intensity + 1
        )
        blurred_img = cv2.GaussianBlur(image, (k_intensity, k_intensity), 0)

        # 2. Create the mask (alpha channel)
        # Start with black mask
        mask = np.zeros((h, w), dtype=np.float32)

        # Draw white rectangle where blur should be
        x_end = min(config.x + config.width, w)
        y_end = min(config.y + config.height, h)
        x_start = max(config.x, 0)
        y_start = max(config.y, 0)

        if x_start >= x_end or y_start >= y_end:
            return image

        cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), (1.0), -1)

        # 3. Feather the mask edges
        # The feathering intensity determines how smooth the transition is
        k_feather = (
            config.feathering if config.feathering % 2 == 1 else config.feathering + 1
        )
        # Use a very large kernel for smooth transitions
        mask_blurred = cv2.GaussianBlur(mask, (k_feather, k_feather), 0)

        # 4. Alpha blend: Final = Original * (1-mask) + Blurry * mask
        # Expand mask to 3 channels for easy multiplication with RGB image
        mask_3c = cv2.merge([mask_blurred, mask_blurred, mask_blurred])

        image = image.astype(np.float32)
        blurred_img = blurred_img.astype(np.float32)

        blended = image * (1.0 - mask_3c) + blurred_img * mask_3c

        return blended.astype(np.uint8)


# --- Randomizer / Pipeline Class ---
class DatasetAugmentor:
    def __init__(
        self, input_dir: str, output_dir: str, target_size: Tuple[int, int] = (756, 756)
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.generator = ArtifactGenerator()
        self.annotations: List[Dict[str, Any]] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- SAFETY PARAMETERS ---
        # We need to know the max possible shift to calculate safe margins.
        # Based on _random_stitches logic:
        # Max shift per line is 35 pixels.
        # We might have up to 2 lines per axis (though current logic usually does 1).
        # Let's be conservative and assume a cumulative shift of ~100px is possible.
        self.MAX_SHIFT_PER_AXIS = 100

    def process_dataset(self, patches_per_source: int = 5):
        valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        image_files = [
            f for f in self.input_dir.iterdir() if f.suffix.lower() in valid_extensions
        ]

        if not image_files:
            print(f"No images found in {self.input_dir}")
            return

        print(f"Found {len(image_files)} source images.")
        patch_count_global = 0

        for src_path in image_files:
            print(f"Processing source: {src_path.name}...")
            large_img = cv2.imread(str(src_path))

            if large_img is None:
                continue

            h_src, w_src = large_img.shape[:2]

            # --- CALCULATE SAFE ZONE ---
            # We need the crop to stay within bounds even if we shift:
            # 1. Left/Up (negative shift): Origin must be > MAX_SHIFT
            # 2. Right/Down (positive shift): Origin must be < Width - Target - MAX_SHIFT

            min_safe_x = self.MAX_SHIFT_PER_AXIS
            min_safe_y = self.MAX_SHIFT_PER_AXIS

            max_safe_x = w_src - self.target_size[0] - self.MAX_SHIFT_PER_AXIS
            max_safe_y = h_src - self.target_size[1] - self.MAX_SHIFT_PER_AXIS

            # Check if image is big enough to contain a safe zone
            if max_safe_x <= min_safe_x or max_safe_y <= min_safe_y:
                print("  -> Skipping: Image too small for safe cropping.")
                continue

            for i in range(patches_per_source):
                # Pick origin strictly from the safe zone
                origin_x = random.randint(min_safe_x, max_safe_x)
                origin_y = random.randint(min_safe_y, max_safe_y)

                # Generate Random Artifact Configs
                stitches = self._random_stitches(self.target_size)
                blurs = self._random_blurs(self.target_size)

                try:
                    # Pass the validated origin
                    result_patch = self.generator.generate_patch(
                        large_image=large_img,
                        origin=(origin_x, origin_y),
                        output_size=self.target_size,
                        stitches=stitches,
                        blurs=blurs,
                    )

                    filename = f"{src_path.stem}_p{i:02d}_{patch_count_global}.png"
                    cv2.imwrite(str(self.output_dir / filename), result_patch)

                    self._record_annotation(filename, stitches, blurs)
                    patch_count_global += 1

                except Exception as e:
                    print(f"  -> Failed patch {i}: {e}")

        self.save_annotations_to_json()
        print(f"\nFinished. Generated {patch_count_global} patches.")

    def _record_annotation(
        self, filename: str, stitches: List[StitchConfig], blurs: List[BlurConfig]
    ):
        entry = {
            "filename": filename,
            "width": self.target_size[0],
            "height": self.target_size[1],
            "artifacts": {
                "stitches": [asdict(s) for s in stitches],
                "blurs": [asdict(b) for b in blurs],
            },
        }
        self.annotations.append(entry)

    def save_annotations_to_json(self, filename: str = "dataset_annotations.json"):
        with open(self.output_dir / filename, "w") as f:
            json.dump(self.annotations, f, indent=2)

    def _random_stitches(self, size: Tuple[int, int]) -> List[StitchConfig]:
        stitches = []
        w, h = size
        shift_range = list(range(-35, -10)) + list(range(10, 36))

        if random.random() < 0.4:
            pos = random.randint(int(w * 0.15), int(w * 0.85))
            sx = random.choice(shift_range + [0])
            sy = random.choice(shift_range)
            stitches.append(StitchConfig(pos, sx, sy, "vertical"))

        if random.random() < 0.4:
            pos = random.randint(int(h * 0.15), int(h * 0.85))
            sx = random.choice(shift_range)
            sy = random.choice(shift_range + [0])
            stitches.append(StitchConfig(pos, sx, sy, "horizontal"))
        return stitches

    def _random_blurs(self, size: Tuple[int, int]) -> List[BlurConfig]:
        blurs = []
        w, h = size
        if random.random() < 0.6:
            num_blurs = random.randint(1, 2)
            for _ in range(num_blurs):
                bw = random.randint(int(w * 0.3), int(w * 0.9))
                bh = random.randint(int(h * 0.3), int(h * 0.9))
                bx = random.randint(int(-bw * 0.2), int(w - bw * 0.5))
                by = random.randint(int(-bh * 0.2), int(h - bh * 0.5))
                b_int = random.choice([25, 35, 45, 55, 75])
                min_dim = min(bw, bh)
                b_feather = random.choice(
                    [
                        int(min_dim * 0.3) | 1,
                        int(min_dim * 0.5) | 1,
                        int(min_dim * 0.7) | 1,
                    ]
                )
                blurs.append(BlurConfig(bx, by, bw, bh, b_int, b_feather))
        return blurs


# --- Execution Block ---
if __name__ == "__main__":
    input_folder = "../../data/clean-full-tiles-sample"
    output_folder = "scanning_artefacts_improved"

    # --- Setup Dummy Data (so the script runs out of the box) ---
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print("Creating dummy source images...")
        for i in range(2):
            # Create very large dummy images (3000x3000)
            dummy = np.zeros((3000, 3000, 3), dtype=np.uint8)
            # Background Texture
            cv2.randn(dummy, (140, 130, 150), (40, 40, 40))
            # Dense Grid to make multidirectional shifts obvious
            for k in range(0, 3000, 50):
                cv2.line(dummy, (k, 0), (k, 3000), (100, 100, 100), 1)
                cv2.line(dummy, (0, k), (3000, k), (100, 100, 100), 1)
            # Random biological-looking shapes
            for _ in range(50):
                center = (random.randint(0, 3000), random.randint(0, 3000))
                axes = (random.randint(50, 200), random.randint(50, 200))
                angle = random.randint(0, 360)
                color = (
                    random.randint(100, 220),
                    random.randint(50, 150),
                    random.randint(180, 255),
                )
                cv2.ellipse(dummy, center, axes, angle, 0, 360, color, -1)
            cv2.imwrite(os.path.join(input_folder, f"wsi_large_{i:02d}.jpg"), dummy)

    # --- Run Augmentor ---
    augmentor = DatasetAugmentor(
        input_dir=input_folder, output_dir=output_folder, target_size=(756, 756)
    )

    augmentor.process_dataset(patches_per_source=10)
    print(f"\nDone. Output in '{output_folder}'")
