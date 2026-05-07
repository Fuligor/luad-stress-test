# Histopathology Artifact Generation Pipeline
This module contains a modular pipeline for procedurally and generatively applying realistic artifacts to histopathological patches (384x384px, 09 MPP). 

## Repository Structure
### Core Generators
Standalone scripts that apply specific artifacts to images based on JSON manifests:

* `generate_dust.py`: Procedurally generates organic smudges and particulate dust.

* `generate_marker.py`: Procedurally applies edge-anchored marker ink (heavy/standard).

* `generate_folds.py`: Applies generative tissue folds using Stable Diffusion (diffusers).

* `generate_precipitates.py`: Applies generative staining precipitates using Stable Diffusion.

* `generate_scanning_artifacts.py`: Wraps the core logic to apply focus blur and stitching errors.

* `scanning_artefacts.py`: Core mathematical/OpenCV logic for scanning anomalies.

### Orchestration & Tracking
* `run_luad_c_generation_pipeline.sh`: The master bash script. Orchestrates the full, sequential generation pipeline for flat cohorts (e.g., TCGA-LUAD). It generates manifests and runs all artifact scripts back-to-back.

* `run_dhmc_tracker.py`: A universal python wrapper for processing nested WSI cohorts (like DHMC). It provides ETA tracking, progress bars, and automatically skips existing files to easily recover machine timeouts.

### Support Directories
* `helpers/`: Contains helper scripts, including manifest generators (`generate_manifests.py`, `generate_manifests_dhmc.py`) that handle fixed-support sampling and deterministic dataset splitting.

* `models_training/`: Contains the scripts and configurations used to fine-tune the Stable Diffusion UNet models for the generative artifacts.

* `utils/`: Shared utilities, including deterministic seed generators to ensure spatial consistency across different artifacts.

* `project.toml`: The uv configuration file managing all deep learning and image processing dependencies.

### Model weights
Model weights for folds and precitipates generation along with artifacts masks (stored in parquet files) are accesible for download under following [link](https://dataverse.harvard.edu/previewurl.xhtml?token=438b2fa7-3999-4739-87f1-c3a31f4de838)