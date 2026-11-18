# NeuroComputing

This repository contains the core evaluation and modeling code for the NeuroComputing project.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure data paths in `configs/` to point to your local dataset directory.

## Data Structure

This code expects datasets to be in JSONL or Parquet format with the following columns:
- `idx`: Unique identifier
- `func`: Code snippet
- `target`: 0 or 1

## Sanitization
This codebase has been cleaned for public release. Internal paths and dataset names have been removed.
