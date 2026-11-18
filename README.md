# NeuroComputing

This repository contains the core evaluation and modeling code for the NeuroComputing project.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure data paths in `configs/global.json` to point to your local dataset directory.

## Running Inference

We have provided an inference template in `examples/run_inference.py`. 
You can run it as follows:

```bash
python examples/run_inference.py
```

Please customize the dataset loading and model paths within the script to match your specific requirements.

## Data Structure

This code expects datasets to be in JSONL or Parquet format with the following columns:
- `idx`: Unique identifier
- `func`: Code snippet
- `target`: 0 or 1

## Sanitization
This codebase has been cleaned for public release. Internal paths and dataset names have been removed.
