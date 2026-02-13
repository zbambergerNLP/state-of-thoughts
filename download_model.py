"""
Download Hugging Face models directly to a specified directory.

Usage:
    python download_model.py
"""

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

def download_model(model_name, save_directory, show_progress=True):
    """
    Download a Huggingface model to a local directory without loading it.

    Args:
        model_name (str): Name of the model on Huggingface (e.g., 'Qwen/Qwen3-30B-A3B-Instruct-2507')
        save_directory (str): Local directory path to save the model
        show_progress (bool): Whether to display progress bar during download
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    logger.info(f"Downloading {model_name} to {save_directory}...")

    # Use snapshot_download to download all model files without loading the model
    snapshot_download(
        repo_id=model_name,
        local_dir=save_directory,
        local_dir_use_symlinks=False,  # Actually download files instead of using symlinks
        revision="main",  # Use the main branch
        resume_download=True,  # Resume interrupted downloads
        max_workers=12,  # Number of workers to use for downloading
        tqdm_class=tqdm if show_progress else None  # Show progress bar if requested
    )

    # Give read-only access to the directory (read + execute, no write)
    os.chmod(save_directory, mode=0o555)  # noqa: S103

    logger.info(f"✓ Successfully downloaded {model_name} to {save_directory}")
    logger.info(f"\tModel files are available at: {os.path.abspath(save_directory)}")

# Example usage
if __name__ == "__main__":
    # NOTE: Change the line below to the model you want to download.
    # Don't forget to add `.value`, since models are stored as enums in constants.py.
    model_name = constants.OpenSourceModel.MINISTRAL_3_14B_REASONING_2512.value

    # TODO[P3]: Add in the readme a section about adding support for new models.
    # and provide guidance on how to download and use the models below and new models.
    if model_name.startswith("Qwen3-"):
        model_name = "Qwen/" + model_name
    elif model_name.startswith("Ministral-"):
        model_name = "mistralai/" + model_name
    elif model_name.startswith("NVIDIA-Nemotron-"):
        model_name = "nvidia/" + model_name

    # IMPORTANT: Update this path to a directory where you have write access!
    # Run these commands on the server to find available storage:
    #   ls -la /scratch/gpfs/
    #   groups  # to see what groups you're in

    # Option 1: Use a group directory (if you're in BSTEWART group)
    # Extract model name from repo_id for directory naming
    model_dir_name = model_name.split("/")[-1]
    save_dir = f"/projects/BSTEWART/model_storage/{model_dir_name}"

    # Option 2: Or use your home directory (WARNING: may have small quota)
    # username = os.environ.get("USER", "zb8227")
    # save_dir = f"/home/{username}/model_storage/Qwen3-30B-A3B"

    logger.info(f"Attempting to download to: {save_dir}")
    logger.info("If this fails with PermissionError, update the save_dir path in the script")

    download_model(model_name, save_dir)
