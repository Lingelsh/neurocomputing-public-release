# NeuroComputing (MVP Release)

This repository contains the Minimal Viable Product (MVP) release of the core modeling, inference, and data preparation code for the NeuroComputing project.

## MVP Scope

This release is intended for reproducibility and data availability compliance. It includes:
- **Core Library**: `src/neurobaseline/` (model structures, data processing, metrics).
- **Configuration**: A template `configs/global.json` for customized runs.
- **Inference Example**: `examples/run_inference.py` (a starting point for running model inference).
- **Data Prep Tool**: `scripts/prepare_dataset.py` (standardized preprocessing for your custom datasets).

## Quickstart

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Use the provided script to format your raw data into the expected JSONL structure:
   ```bash
   python scripts/prepare_dataset.py --input <path_to_data.csv> --output <data_dir>
   ```

3. **Run Inference**:
   Configure `configs/global.json` and run the inference template:
   ```bash
   python examples/run_inference.py
   ```

## Security & Privacy Note
This code has been sanitized for public release. Internal paths, sensitive tokens, and environment-specific experiment configurations have been removed.
