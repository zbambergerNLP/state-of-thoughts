"""
Download Hugging Face models directly to a specified directory.

Usage:
    python scripts/download_models.py --model_directory /path/to/model_storage
    python scripts/download_models.py --model_directory /path/to/model_storage --model Qwen/Qwen3-30B-A3B-Instruct-2507
"""

import argparse
import logging
import os

from huggingface_hub import snapshot_download
from tqdm import tqdm

import constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default models used by the argument generation experiment
DEFAULT_GENERATIVE_MODEL = f"Qwen/{constants.OpenSourceModel.QWEN_3_30B_A3B_INSTRUCT_2507}"
DEFAULT_RERANKER_MODEL = f"Qwen/{constants.OpenSourceModel.QWEN_3_RERANKER_8B}"


def download_model(
    model_name: str,
    save_directory: str,
    show_progress: bool = True,
) -> None:
    """
    Download a Huggingface model to a local directory without loading it.

    Args:
        model_name: Name of the model on Huggingface (e.g., 'Qwen/Qwen3-30B-A3B-Instruct-2507')
        save_directory: Local directory path to save the model
        show_progress: Whether to display progress bar during download
    """
    os.makedirs(save_directory, exist_ok=True)

    logger.info("Downloading %s to %s...", model_name, save_directory)

    snapshot_download(
        repo_id=model_name,
        local_dir=save_directory,
        local_dir_use_symlinks=False,
        revision="main",
        resume_download=True,
        max_workers=12,
        tqdm_class=tqdm if show_progress else None,
    )

    # Give read permissions to the model directory
    os.chmod(save_directory, mode=0o555)  # noqa: S103

    logger.info("✓ Successfully downloaded %s to %s", model_name, save_directory)
    logger.info("\tModel files are available at: %s", os.path.abspath(save_directory))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models to a local directory.",
    )
    parser.add_argument(
        "--model_directory",
        type=str,
        required=True,
        help="Base directory where models will be saved. Each model is saved in a subdirectory named after the model (e.g., model_directory/Qwen3-30B-A3B-Instruct-2507).",
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        default=None,
        metavar="REPO_ID",
        help="Hugging Face model repo ID (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507). Can be repeated for multiple models. If omitted, downloads the default generative and reranker models.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable the download progress bar.",
    )
    args = parser.parse_args()

    if args.model:
        models = args.model
    else:
        models = [DEFAULT_GENERATIVE_MODEL, DEFAULT_RERANKER_MODEL]
        logger.info("No models specified; downloading default models for argument generation.")

    for model_name in models:
        # Use the last part of the repo ID as the subdirectory name (e.g., Qwen3-30B-A3B-Instruct-2507)
        subdir_name = model_name.split("/")[-1]
        save_directory = os.path.join(args.model_directory, subdir_name)
        download_model(
            model_name=model_name,
            save_directory=save_directory,
            show_progress=not args.no_progress,
        )


if __name__ == "__main__":
    main()
