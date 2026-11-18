# CFVul

This repository provides the training and evaluation code for CFVul and the frozen scores and dataset metadata used in the paper.

## Recompute the paper metrics

Install the dependencies with Python 3.11 or newer. The metric scripts run on CPU and use the supplied scores directly.

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_metrics.py > paper-metrics.csv
python scripts/reproduce_metrics.py --transformations > transformation-metrics.json
```

To verify the two CFVul outputs and their CodeBERT comparison on PrimeVul:

```sh
python scripts/reproduce_metrics.py --dataset primevul --model codebert --model cfa --model cfa_raw
```

See the [data documentation](reproduction/README.md) for dataset provenance, file schemas, and evaluation subsets.

## Train on PrimeVul

Download the original official PrimeVul files and the CodeBERT model revision recorded in [the checkpoint metadata](reproduction/protocol/primevul-checkpoints.json).

```sh
python scripts/prepare_dataset.py --input path/to/official/primevul --output data/primevul
export PYTHONPATH=src
python -m neurobaseline train --config configs/primevul-codebert.json --model path/to/codebert --local-files-only
python -m neurobaseline train --config configs/primevul-cfa.json --model path/to/codebert --local-files-only
python examples/run_inference.py --run runs/primevul-cfa/cfa/seed13 --split test --device cuda
```

The training commands require a CUDA environment. The inference example scores a run produced by training. For exact verification of the reported results, use the frozen scores shipped with this repository.

## Licensing

The repository's [MIT license](LICENSE) covers its code and documentation. Obtain the original datasets and pretrained models from their upstream maintainers under their respective terms.
