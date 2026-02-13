#!/usr/bin/env python3
"""
Download NoveltyBench dataset for offline use.

This script downloads the NoveltyBench dataset from HuggingFace and saves it
locally for use in environments without internet access.

Usage:
	python experiments/download_dataset.py --output-dir /path/to/save/dataset

Example NoveltyBench invocation using the downloaded dataset:

```bash
python -m experiments.noveltybench.run_baseline_noveltybench_experiment \
  --model /path/to/model \
  --dataset-path /path/to/saved/dataset \
  --scorer-type local
```
"""

# Standard library imports
import argparse
import logging
from pathlib import Path

# Third-party imports
from datasets import load_dataset

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_dataset(output_dir: Path, splits: list[str] | None = None) -> None:
	"""
	Download NoveltyBench dataset from HuggingFace.

	Args:
		output_dir: Directory to save the dataset
		splits: List of splits to download (default: ["curated", "wildchat"])

	Raises:
		RuntimeError: If download fails
	"""
	if splits is None:
		splits = ["curated", "wildchat"]

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	logger.info("=" * 60)
	logger.info("NoveltyBench Dataset Download")
	logger.info("=" * 60)
	logger.info("Dataset: yimingzhang/novelty-bench")
	logger.info(f"Splits: {', '.join(splits)}")
	logger.info(f"Output directory: {output_dir}")
	logger.info("=" * 60)

	logger.info("\nDownloading dataset from HuggingFace...")

	# Download each split separately
	for split in splits:
		logger.info(f"Downloading split: {split}...")
		dataset_split = load_dataset("yimingzhang/novelty-bench", split=split)

		split_dir = output_dir / split
		split_dir.mkdir(parents=True, exist_ok=True)

		logger.info(f"Saving {split} split to {split_dir}...")
		dataset_split.save_to_disk(str(split_dir))

	logger.info("\n" + "=" * 60)
	logger.info("✓ Download complete!")
	logger.info("=" * 60)
	logger.info(f"Dataset saved to: {output_dir}")
	logger.info(f"Available splits: {', '.join(splits)}")
	logger.info("\nTo use this dataset for a baseline experiment, run:")
	logger.info("python -m experiments.noveltybench.run_baseline_noveltybench_experiment \\")
	logger.info("--model /path/to/model \\")
	logger.info(f"--dataset-path {output_dir} \\")
	logger.info("--scorer-type local")
	logger.info("=" * 60)


def main():
	"""Main entry point."""
	parser = argparse.ArgumentParser(
		description="Download NoveltyBench dataset for offline use",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Download to default location
  python scripts/download_dataset.py

  # Download to custom location
  python scripts/download_dataset.py \\
      --output-dir /projects/BSTEWART/dataset_storage/noveltybench

  # Download specific splits only
  python scripts/download_dataset.py \\
      --output-dir /projects/BSTEWART/dataset_storage/noveltybench \\
      --splits curated
		""",
	)

	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("/projects/BSTEWART/dataset_storage/noveltybench"),
		help="Directory to save the dataset (default: /projects/BSTEWART/dataset_storage/noveltybench)",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		choices=["curated", "wildchat"],
		default=None,
		help="Splits to download (default: all splits)",
	)

	args = parser.parse_args()

	try:
		download_dataset(args.output_dir, args.splits)
	except Exception as e:
		logger.error(f"Download failed: {e}")
		exit(1)


if __name__ == "__main__":
	main()
