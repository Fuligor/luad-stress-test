# LUAD Stress Test

**Research use only — Not for clinical or diagnostic use.** This software is intended for research and development purposes only. It has not been validated for clinical decision making, diagnostics, or any regulated medical use. Do not use in patient care or production medical environments.

A comprehensive pipeline for analyzing histopathology patches in Lung Adenocarcinoma (LUAD) datasets using deep learning models. The project provides automated dataset generation, model inference, and prediction post-processing across multiple datasets and encoder architectures.

This repository was developed for the research work titled: "Beyond Clean Data: The LUAD-C Benchmark for Stress-Testing and Feature Stability in Computational Pathology".


## Overview

This project implements a three-stage analysis pipeline:

1. **Prepare**: Generate patches from raw Whole Slide Images (WSI) or patch datasets
2. **Predict**: Run inference on patches using multiple pre-trained encoder models
3. **Postprocess**: Smooth predictions and compute final analysis results

Supported datasets:
- **dhmc**: Digitized Histology Medical Collection
- **anorak**: Anorak dataset
- **tcga**: The Cancer Genome Atlas LUAD subset

Supported artifact types (image quality assessment):
- blur, dust, fold, marker, prec (precipitates), stitch

Classification models support 7 LUAD histologic patterns:
- NTU (Normal tissue), ACC (Acinar), CRB (Cribriform), LEP (Lepidic), 
- MIP (Micropapillary), PAP (Papillary), SOL (Solid)

## Installation

### Requirements

- Python ≥ 3.12
- CUDA 12.4 (for GPU) or CPU-only installation

### Setup

Clone the repository and install with dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd luad-stress-test

# Install with CPU backend
uv sync --extra cpu

# OR install with CUDA backend
uv sync --extra cuda
```

### Environment Configuration

Create a `.env` file in the project root to configure data paths:

```env
# Example paths - adjust to your environment
DATA_RAW_PATH=/path/to/raw/data
DATA_PROCESSED_PATH=/path/to/processed/data
MODEL_PATH=/path/to/models
```

## Usage

### CLI Command

The main entry point is the `luad-stress-test` command:

```bash
luad-stress-test COMMAND [OPTIONS]
```

### Pipeline Command

Run the complete analysis pipeline or individual steps:

```bash
luad-stress-test pipeline \
  --step <step> \
  --model-name <model> \
  --dataset-name <dataset> \
  --artifact-name <artifact> \
  --write-embedings <bool>
```

#### Parameters

**`--step`** (required)
- `prepare`: Only generate patches from raw data
- `predict`: Only run model inference
- `postprocess`: Only smooth and compute results
- `full`: Run the complete pipeline (default: `full`)

**`--model-name`** (default: `all`)
- Specific model name or `all` to process all available models
- Available encoders: EfficientNet, GigaPath, ResNet18, Swin, UNI, ViRChow2, ViT-B/16

**`--dataset-name`** (default: `all`)
- `dhmc`, `anorak`, `tcga`, or `all` to process all datasets

**`--artifact-name`** (optional, default: `all`)
- `blur`, `dust`, `fold`, `marker`, `prec`, `stitch`, or `all`
- When `None`: process patches without artifact filtering

**`--write-embedings`** (optional)
- Setting to `True` saves intermediate embedding vectors during prediction
- Useful for downstream analysis or visualization

### Examples

#### Run complete pipeline on all datasets
```bash
luad-stress-test pipeline --step full
```

#### Prepare datasets only
```bash
luad-stress-test pipeline --step prepare --dataset-name all
```

#### Run inference with specific model on DHMC dataset
```bash
luad-stress-test pipeline \
  --step predict \
  --model-name efficientnet \
  --dataset-name dhmc
```

#### Process all models with Anorak dataset, no artifact filtering
```bash
luad-stress-test pipeline \
  --step predict \
  --dataset-name anorak
```

#### Run postprocessing (smooth + compute results) on DHMC only
```bash
luad-stress-test pipeline \
  --step postprocess \
  --dataset-name dhmc
```

#### Save embeddings during prediction
```bash
luad-stress-test pipeline \
  --step predict \
  --model-name uni \
  --dataset-name tcga \
  --write-embedings True
```

#### Process all artifacts separately for Anorak dataset
```bash
luad-stress-test pipeline \
  --step predict \
  --dataset-name anorak \
  --artifact-name all
```

## Project Structure

```
src/luad_stress_test/
├── predict/                    # Model inference pipeline
│   ├── models/                 # Encoder architectures
│   │   └── encoders/           # Pre-trained model implementations
│   ├── dataset/                # Patch dataset loaders
│   └── metrics/                # Performance evaluation
├── preprocessing/              # Data preparation
│   ├── wsi/                    # WSI tiling and tissue extraction
│   ├── anorak/                 # Anorak dataset processing
│   └── luad_c/                 # LUAD-C dataset processing
├── prediction_postprocessing/  # Results analysis
│   ├── smooth_predictions.py   # Apply spatial smoothing
│   ├── assess_predominant_pattern.py  # Classify predominant histology
│   └── compute_results.py      # Generate final metrics
└── utils.py                    # Dataset and label definitions
```

## Pipeline Architecture

### Stage 1: Preparation

Generates standardized patches from raw histopathology data:
- **DHMC**: Extracts patches from WSI using configurable tiling strategy
- **Anorak**: Processes image/mask pairs into patch format
- **TCGA**: Converts LUAD patches to standardized format

### Stage 2: Prediction

For each (model, dataset, artifact) combination:
1. Loads pre-trained encoder model
2. Creates patch dataset with optional artifact filtering
3. Runs batch inference on GPU/CPU
4. Saves predictions and optional embeddings

### Stage 3: Postprocessing

On DHMC dataset only (configurable):
1. **Smoothing**: Applies spatial filtering to neighboring patch predictions
2. **Pattern Assessment**: Determines predominant histologic pattern per slide
3. **Results Computation**: Aggregates metrics and generates summary statistics

## Configuration

### Dataset Paths

The `PathManager` class manages all data directory locations. Configure via environment or code:

- `PathManager.dhmc_raw()`: Raw WSI files
- `PathManager.anorak_raw()`: Anorak raw images/masks
- `PathManager.dhmc_processed()`: Output patch directories
- `PathManager.model_dir()`: Pre-trained model checkpoints

### Model Configuration

Each encoder model has specific configuration:
- Input size: 384×384 pixels (default)
- Patch normalization: ImageNet pre-training statistics
- Target magnification: 0.9 microns per pixel (MPP)

## License

See [LICENSE.md](LICENSE.md)

## Authors

- Zuzanna Krawczyk-Borysiak
- Adam Krawczyk
- Małgorzata Sokół
- Zaneta Swiderska-Chadaj