import argparse
import os
import io
import sys
import json
import random
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from diffusers import AutoPipelineForInpainting
from tqdm.auto import tqdm

# --- Robust Import Safety Net ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.utils import set_seed_from_string

# --- CONFIGURATION ---
DEFAULT_PROMPT = "Histopathological image of lung tissue, hematoxylin and eosin staining, with a fold artifact visible."
NEGATIVE_PROMPT = "blur, low quality, distortion, text, watermark"

def process_mask_for_inference(pil_mask, target_size=(512, 512)):
    """
    Prepares the mask to match the 'Soft Edge' style seen during training.
    """
    # 1. Resize (Nearest is fine here because we will blur later)
    mask_resized = pil_mask.resize(target_size, Image.Resampling.NEAREST)
    
    # 2. Convert to Tensor
    mask_tensor = transforms.ToTensor()(mask_resized) # [1, H, W]
    
    # 3. Dilation (Optional but Recommended) - Adds a safety buffer
    #    If your masks are very precise, you can skip this. 
    #    But if you see 'clean tissue' borders inside the artifact, keep this.
    #    kernel_size=15 is ~7 pixels expansion.
    dilated = torch.nn.functional.max_pool2d(mask_tensor.unsqueeze(0), kernel_size=15, stride=1, padding=7)
    dilated = dilated.squeeze(0)
    
    # 4. BLUR (CRITICAL) - Matches your training data's BILINEAR look
    #    Without this, you get dotted lines/aliasing.
    blurred = gaussian_blur(dilated, kernel_size=(11, 11), sigma=2.0)
    
    # 5. Return PIL Image
    return transforms.ToPILImage()(blurred)

def get_bezier_curve(p0, p1, p2, p3, num_points=100):
    """Generates points for a Bezier curve (for procedural masks)."""
    t = np.linspace(0, 1, num_points)
    x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
    y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
    return list(zip(x, y))

def generate_procedural_mask(size=(512, 512)):
    """Creates a random organic-looking line mask if no real masks are available."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    
    x0, y0 = random.randint(0, size[0]), random.randint(0, size[1])
    x3, y3 = random.randint(0, size[0]), random.randint(0, size[1])
    x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
    x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])
    
    points = get_bezier_curve((x0,y0), (x1,y1), (x2,y2), (x3,y3))
    width = random.randint(15, 40)
    draw.line(points, fill=255, width=width, joint="curve")
    
    return mask

def load_mask_bank(parquet_path):
    """Extracts real artifact masks from the dataset."""
    print(f"Loading mask bank from {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
        masks = []
        # Filter for rows that actually have an artifact
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Bank"):
            m_bytes = row['mask']
            if m_bytes is None: 
                continue
            
            # Basic check without full load if possible, or just try loading
            try:
                with Image.open(io.BytesIO(m_bytes)) as m_img:
                    if np.array(m_img.convert("L")).max() > 0:
                        masks.append(m_bytes)
            except Exception:
                pass
        
        print(f"Found {len(masks)} real artifact masks.")
        return masks
    except Exception as e:
        print(f"Warning: Could not load mask bank ({e}). Will use procedural generation.")
        return []

def load_filter_list(json_path):
    """Parses the JSON file to get a set of allowed filenames."""
    if not json_path or not os.path.exists(json_path):
        return None
    
    print(f"Loading filter list from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract keys from "tagged_files"
        if "tagged_files" in data:
            allowed_files = set(data["tagged_files"].keys())
            print(f"Filter active: Processing {len(allowed_files)} specific images.")
            return allowed_files
        else:
            print("Warning: JSON found but 'tagged_files' key is missing. Processing ALL images.")
            return None
    except Exception as e:
        print(f"Error reading JSON filter: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate Artifacts on Clean Tissue")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved model folder")
    parser.add_argument("--input_source", type=str, required=True, help="Path to Parquet file OR Folder of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument("--parquet_for_masks", type=str, default=None, help="Path to parquet file to borrow masks from")
    parser.add_argument("--json_filter", type=str, default=None, help="Path to JSON file containing list of images to process")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_images", type=int, default=None, help="Limit total generation count (optional)")
    
    args = parser.parse_args()
    
    # 1. Setup Model
    print(f"Loading model from {args.checkpoint}...")
    pipeline = AutoPipelineForInpainting.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(args.device)
    pipeline.set_progress_bar_config(disable=True)

    # 2. Setup Mask Bank
    mask_bank = []
    # Determine where to get masks from
    mask_source = args.parquet_for_masks if args.parquet_for_masks else (args.input_source if args.input_source.endswith(".parquet") else None)
    
    if mask_source:
        mask_bank = load_mask_bank(mask_source)
    else:
        print("No parquet file provided for masks. Using procedural generation.")

    # 3. Load JSON Filter
    allowed_filenames = load_filter_list(args.json_filter)

    # 4. Input Data Generator
    def image_generator():
        # A. PARQUET MODE (Ignores JSON filter usually, unless filenames match exactly)
        if args.input_source.endswith(".parquet"):
            print("Reading input parquet...")
            df = pd.read_parquet(args.input_source)
            if 'split' in df.columns:
                df = df[df['split'] == 'test']
            
            count = 0
            for idx, row in df.iterrows():
                # Check if image is clean
                if np.array(Image.open(io.BytesIO(row['mask'])).convert("L")).max() > 0:
                    continue # Skip existing artifacts
                
                fname = row.get('filename', f"sample_{idx}.png")
                
                # Apply JSON Filter if active
                if allowed_filenames is not None and fname not in allowed_filenames:
                    continue

                if args.num_images and count >= args.num_images: 
                    break
                
                img = Image.open(io.BytesIO(row['image'])).convert("RGB")
                yield fname, img
                count += 1

        # B. FOLDER MODE (Primary use case for JSON filter)
        elif os.path.isdir(args.input_source):
            files = [f for f in os.listdir(args.input_source) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
            
            # Apply Filter immediately
            if allowed_filenames is not None:
                original_count = len(files)
                files = [f for f in files if f in allowed_filenames]
                print(f"Filtered {original_count} files down to {len(files)} target images.")
            
            count = 0
            for f in files:
                if args.num_images and count >= args.num_images: 
                    break
                path = os.path.join(args.input_source, f)
                try:
                    img = Image.open(path).convert("RGB")
                    yield f, img
                    count += 1
                except Exception as e:
                    print(f"Skipping {f}: {e}")

    # 5. Processing Loop
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True) 

    print("Starting generation...")
    
    processed_count = 0
    for filename, original_img in tqdm(image_generator()):
        original_w, original_h = original_img.size
        
        # --- DETERMINISTIC SEEDING PER IMAGE ---
        # 1. Fix Python's random & np.random based on filename
        set_seed_from_string(filename)
        
        # 2. Use our fixed python random to generate a seed for PyTorch
        local_torch_seed = random.randint(0, 2**31)
        generator = torch.Generator(device=args.device).manual_seed(local_torch_seed)

        # Prepare Input (Resize to 512x512 for Stable Diffusion)
        input_img_512 = original_img.resize((512, 512), Image.Resampling.BILINEAR)
        
        # Select Mask
        if mask_bank:
            mask_bytes = random.choice(mask_bank)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            mask_img = mask_img.resize((512, 512), Image.Resampling.NEAREST)
            #mask_img = process_mask_for_inference(mask_img, target_size=(512, 512))
        else:
            mask_img = generate_procedural_mask((512, 512))

        # Run Inference
        with torch.autocast(args.device):
            result_512 = pipeline(
                prompt=DEFAULT_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=input_img_512,
                mask_image=mask_img,
                num_inference_steps=30,
                guidance_scale=7.5,
                strength=0.9, # for model to see a bit of underlying tissue
                generator=generator # Inject the deterministic PyTorch generator here
            ).images[0]

        # Resize back to original dimensions
        result_final = result_512.resize((original_w, original_h), Image.Resampling.LANCZOS)
        mask_final = mask_img.resize((original_w, original_h), Image.Resampling.NEAREST)

        # Save
        base_name = os.path.splitext(filename)[0]
        save_path_img = os.path.join(args.output_dir, f"{filename}")
        save_path_mask = os.path.join(args.output_dir, "masks", f"{base_name}_mask.png")
        
        result_final.save(save_path_img)
        mask_final.save(save_path_mask)
        processed_count += 1

    print(f"Generation complete. Processed {processed_count} images into {args.output_dir}.")

if __name__ == "__main__":
    main()