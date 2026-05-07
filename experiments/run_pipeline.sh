#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default paths
INPUT_DATA_DIR="./data/raw_results"
RESULTS_CSV="./results/LUAD_C.csv"
OUTPUT_DIR="./results/charts"
DO_PROCESSING=true

# --- Bulletproof Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --csv) 
            RESULTS_CSV="$2"
            DO_PROCESSING=false
            shift 2
            ;;
        --data) 
            INPUT_DATA_DIR="$2"
            shift 2
            ;;
        --out) 
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *) 
            echo "Error: Unknown parameter passed: $1"
            echo "Usage: ./run_pipeline.sh [--data <dir>] [--csv <file>] [--out <dir>]"
            exit 1 
            ;;
    esac
done

echo "Starting evaluation pipeline..."

# If we are skipping processing, make absolutely sure the CSV exists AND isn't empty!
if [ "$DO_PROCESSING" = false ]; then
    if [ ! -s "$RESULTS_CSV" ]; then
        echo "Error: The file '$RESULTS_CSV' does not exist or is empty!"
        exit 1
    fi
    echo "Existing, valid CSV file detected. Skipping bootstrap processing..."
fi

# Ensure output directories exist safely
echo "Ensuring output directories exist..."
mkdir -p "$(dirname "$RESULTS_CSV")"
mkdir -p "$OUTPUT_DIR"

# 1. Ensure dependencies are installed via uv
echo "Syncing dependencies using uv..."
uv sync

# 2. Process results and run bootstrap (if needed)
if [ "$DO_PROCESSING" = true ]; then
    echo "Processing results and calculating bootstrap metrics..."
    uv run python process_results.py \
        --base-path "$INPUT_DATA_DIR" \
        --output "$RESULTS_CSV"
fi

# 3. Generate Charts
echo "Generating radar charts..."
uv run python generate_radar.py --input "$RESULTS_CSV" --output-dir "$OUTPUT_DIR"

echo "Generating tornado charts..."
uv run python generate_tornado.py --input "$RESULTS_CSV" --output-dir "$OUTPUT_DIR"

echo "Generating cohort transition charts..."
uv run python generate_transitions.py --input "$RESULTS_CSV" --output-dir "$OUTPUT_DIR"

echo "Generating latent space profiles..."
uv run python generate_latent_profiles.py --input "$RESULTS_CSV" --output-dir "$OUTPUT_DIR"

echo "Generating overconfidence analysis charts..."
uv run python generate_overconfidence.py --input "$RESULTS_CSV" --output-dir "$OUTPUT_DIR"

echo "Pipeline finished successfully!"