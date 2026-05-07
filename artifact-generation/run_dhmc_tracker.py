import os
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import timedelta

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_missing_files(manifest_path, output_dir):
    """
    Returns a list of files that are missing in the output_dir based on the manifest.
    """
    if not manifest_path.exists():
        return []
    
    try:
        with open(manifest_path, 'r') as f:
            data = json.load(f)
            # Get filenames from the 'tagged_files' key
            expected_files = data.get("tagged_files", {}).keys()
    except Exception as e:
        print(f"Error reading manifest {manifest_path}: {e}")
        return []

    missing = []
    for filename in expected_files:
        if not (output_dir / filename).exists():
            missing.append(filename)
    
    return missing

def run_generation(args):
    manifest_root = Path(args.manifest_root)
    input_images_root = Path(args.input_root)
    output_images_root = Path(args.output_root)

    if not manifest_root.exists():
        print(f"Error: Manifest directory not found: {manifest_root}")
        return

    slide_dirs = sorted([d for d in manifest_root.iterdir() if d.is_dir() and d.name.startswith("DHMC")])
    total_slides = len(slide_dirs)
    
    print("--- BATCH GENERATION START ---")
    print(f"Target Script: {args.script}")
    print(f"Total slides found in manifests: {total_slides}")
    print("-" * 50)

    start_time_total = time.time()
    processed_count = 0

    for slide_dir in slide_dirs:
        slide_id = slide_dir.name
        manifest_file = slide_dir / "manifest.json"
        input_dir = input_images_root / slide_id
        output_dir = output_images_root / slide_id
        
        if not manifest_file.exists() or not input_dir.exists():
            print(f"Skipping {slide_id}: Manifest or Input directory missing.")
            continue

        # --- PROGRESS CHECK LOGIC ---
        missing_files = get_missing_files(manifest_file, output_dir)
        
        if len(missing_files) == 0:
            print(f"Skipping {slide_id}: Already fully processed (all files exist).")
            # Increase the counter for accurate progress tracking, but don't measure start time
            processed_count += 1
            continue
        else:
            print(f"Processing {slide_id}: Missing {len(missing_files)} files.")

        output_dir.mkdir(parents=True, exist_ok=True)
        processed_count += 1
        start_time_slide = time.time()
        
        print(f"[{processed_count}/{total_slides}] Running processing for: {slide_id}...")

        # Base command that applies to all scripts
        command = [
            "uv", "run", "python", args.script,
            "--input_source", str(input_dir),
            "--output_dir", str(output_dir),
            "--json_filter", str(manifest_file)
        ]

        # Add optional stable diffusion arguments if provided
        if args.checkpoint:
            command.extend(["--checkpoint", args.checkpoint])
        if args.parquet_path:
            command.extend(["--parquet_for_masks", args.parquet_path])
        if args.artifact_type:
            command.extend(["--artifact_type", args.artifact_type])

        try:
            # We use subprocess.run to execute the python script in a separate process
            subprocess.run(command, check=True)
            
            end_time_slide = time.time()
            duration_slide = end_time_slide - start_time_slide
            
            elapsed_total = end_time_slide - start_time_total
            # Calculate average time based only on actively processed slides to keep ETA accurate
            avg_time = elapsed_total / processed_count 
            remaining_slides = total_slides - processed_count
            eta = avg_time * remaining_slides
            
            progress_bar = "#" * int((processed_count / total_slides) * 20) + "-" * (20 - int((processed_count / total_slides) * 20))
            
            print(f"   Done in: {format_time(duration_slide)}")
            print(f"   Progress: [{progress_bar}] {int((processed_count/total_slides)*100)}%")
            print(f"   Elapsed: {format_time(elapsed_total)} | ETA: {format_time(eta)}")
            print("-" * 50)

        except subprocess.CalledProcessError:
            print(f"Error processing {slide_id}. Skipping to next.")
            continue

    total_duration = time.time() - start_time_total
    print(f"\nCOMPLETED! Total execution time: {format_time(total_duration)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal tracking wrapper for long-running artifact generation.")
    
    # Required routing arguments
    parser.add_argument("--script", type=str, required=True, help="Path to the artifact generator script to run (e.g., generate_folds.py)")
    parser.add_argument("--manifest_root", type=str, required=True, help="Path to the root directory containing DHMC manifests")
    parser.add_argument("--input_root", type=str, required=True, help="Path to the root directory of input images")
    parser.add_argument("--output_root", type=str, required=True, help="Path to the root directory where outputs will be saved")
    
    # Optional arguments used by specific scripts (Stable Diffusion vs OpenCV)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (required for folds/precipitates)")
    parser.add_argument("--parquet_path", type=str, default=None, help="Path to parquet file for masks (required for folds/precipitates)")
    parser.add_argument("--artifact_type", type=str, default=None, help="Type of artifact (required for scanning_artifacts.py)")

    args = parser.parse_args()
    run_generation(args)