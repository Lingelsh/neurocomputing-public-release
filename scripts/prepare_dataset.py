"""Make the official PrimeVul splits available to the training CLI."""

import argparse
from pathlib import Path

SPLITS = ("train", "valid", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory with primevul_{train,valid,test}.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="New directory for the CLI's split links")
    args = parser.parse_args()
    sources = {split: (args.input / f"primevul_{split}.jsonl").resolve() for split in SPLITS}
    for source in sources.values():
        if not source.is_file():
            parser.error(f"Missing official split: {source}")
    args.output.mkdir(parents=True)
    for split, source in sources.items():
        (args.output / f"{split}.jsonl").symlink_to(source)


if __name__ == "__main__":
    main()
