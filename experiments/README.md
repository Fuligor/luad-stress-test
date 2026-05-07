# Conference Results Pipeline

This module contains the evaluation pipeline for processing preliminary models predictions under artifacts stress, running statistical bootstrap analysis, and generating the final results tables for the conference paper.

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python environment management.

1. Install `uv`:
   ```bash
   curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh


## Setup & Installation

```
uv venv
source .venv/bin/activate
uv sync
```

## Running Pipeline
Whole pipeline:
```
./run_pipeline.sh --data /path/to/raw_results --out ./results/charts
```
Charts only generation on the basics of bootstrap csv:
```
./run_pipeline.sh --csv results/LUAD_C.csv
```

### Generating bootstrap table with results only
```
uv run python process_results.py \
    --base-path /path/to/your/results/data \
    --output LUAD_C.csv
```