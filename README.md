# CFVul

Code and evaluation data for *Mitigating fix reversal: Counterfactual constraints for function-level vulnerability detection*. This repository includes the CFVul implementation, PrimeVul training configurations, and the data needed to recompute the reported evaluation metrics.

## Recompute evaluation metrics

Use Python 3.11 or newer. The frozen score tables can be evaluated on CPU without downloading datasets or model weights.

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_metrics.py > paper-metrics.csv
python scripts/reproduce_metrics.py --transformations > transformation-metrics.json
```

The [data documentation](reproduction/README.md) describes the upstream datasets, paper splits, score columns, and code-transformation conditions.

## Train on PrimeVul

Download the official original-release PrimeVul splits from the [upstream repository](https://github.com/DLVulDet/PrimeVul#-primevul-dataset) and the CodeBERT revision recorded in the [checkpoint metadata](reproduction/protocol/primevul-checkpoints.json). The training configurations target CUDA.

```sh
python scripts/prepare_dataset.py --input path/to/official/primevul --output data/primevul
export PYTHONPATH=src
python -m neurobaseline train --config configs/primevul-codebert.json --model path/to/codebert --local-files-only
python -m neurobaseline train --config configs/primevul-cfa.json --model path/to/codebert --local-files-only
python -m neurobaseline score --run runs/primevul-cfa/cfa/seed13 --split test --device cuda
```

## License

The repository code and documentation are available under the [MIT license](LICENSE). Obtain the datasets and pretrained model from their upstream maintainers under their respective terms.
