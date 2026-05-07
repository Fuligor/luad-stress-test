#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default variables
INPUT_DIR=""
OUTPUT_DIR=""
FOLDS_CKPT=""
FOLDS_PARQUET=""
PREC_CKPT=""
PREC_PARQUET=""

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_dir) INPUT_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --folds_ckpt) FOLDS_CKPT="$2"; shift 2 ;;
        --folds_parquet) FOLDS_PARQUET="$2"; shift 2 ;;
        --prec_ckpt) PREC_CKPT="$2"; shift 2 ;;
        --prec_parquet) PREC_PARQUET="$2"; shift 2 ;;
        *) 
            echo "Error: Unknown parameter passed: $1"
            echo "Usage: ./run_tcga_generation_pipeline.sh --input_dir <dir> --output_dir <dir> --folds_ckpt <path> --folds_parquet <path> --prec_ckpt <path> --prec_parquet <path>"
            exit 1 
            ;;
    esac
done

# Basic validation
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --input_dir and --output_dir are required."
    exit 1
fi

if [ -z "$FOLDS_CKPT" ] || [ -z "$FOLDS_PARQUET" ] || [ -z "$PREC_CKPT" ] || [ -z "$PREC_PARQUET" ]; then
    echo "Warning: Deep learning assets (checkpoints/parquets) are not fully provided. Folds/Precipitates steps might fail if executed."
fi

MANIFESTS_DIR="$OUTPUT_DIR/manifests"

echo "================================================="
echo "STARTING MASTER ARTIFACT GENERATION PIPELINE"
echo "Input Directory : $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "================================================="

#echo "Syncing dependencies using uv..."
#uv sync

# ---------------------------------------------------------
# STEP 1: Generate Manifests
# ---------------------------------------------------------
echo ""
echo "[STEP 1/7] Generating Manifests..."
python helpers/generate_manifests.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$MANIFESTS_DIR"

# ---------------------------------------------------------
# STEP 2: Generate Folds
# ---------------------------------------------------------
echo ""
echo "[STEP 2/7] Generating Fold Artifacts..."
python generate_folds.py \
    --checkpoint "$FOLDS_CKPT" \
    --input_source "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR/fold" \
    --parquet_for_masks "$FOLDS_PARQUET" \
    --json_filter "$MANIFESTS_DIR/manifest_stress_fold_10.json"

# ---------------------------------------------------------
# STEP 3: Generate Precipitates
# ---------------------------------------------------------
echo ""
echo "[STEP 3/7] Generating Precipitate Artifacts..."
python generate_precipitates.py \
    --checkpoint "$PREC_CKPT" \
    --input_source "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR/prec" \
    --parquet_for_masks "$PREC_PARQUET" \
    --json_filter "$MANIFESTS_DIR/manifest_stress_prec_10.json"

# ---------------------------------------------------------
# STEP 4: Generate Markers
# ---------------------------------------------------------
echo ""
echo "[STEP 4/7] Generating Marker Artifacts..."
python generate_marker.py \
    --input_source "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR/marker" \
    --json_filter "$MANIFESTS_DIR/manifest_stress_marker_10.json"

# ---------------------------------------------------------
# STEP 5: Generate Dust
# ---------------------------------------------------------
echo ""
echo "[STEP 5/7] Generating Dust Artifacts..."
python generate_dust.py \
    --input_source "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR/dust" \
    --json_filter "$MANIFESTS_DIR/manifest_stress_dust_10.json"

# ---------------------------------------------------------
# STEP 6: Generate Scanning Artifacts (Blur)
# ---------------------------------------------------------
echo ""
echo "[STEP 6/7] Generating Scanning Artifacts (Blur)..."
python generate_scanning_artifacts.py \
    --input_source "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR/blur" \
    --json_filter "$MANIFESTS_DIR/manifest_stress_blur_10.json" \
    --artifact_type "blur"

# ---------------------------------------------------------
# STEP 7: Generate Scanning Artifacts (Stitch)
# ---------------------------------------------------------
echo ""
echo "[STEP 7/7] Generating Scanning Artifacts (Stitch)..."
python generate_scanning_artifacts.py \
    --input_source "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR/stitch" \
    --json_filter "$MANIFESTS_DIR/manifest_stress_stitch_10.json" \
    --artifact_type "stitch"

echo ""
echo "================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "All artifacts generated in: $OUTPUT_DIR"
echo "================================================="