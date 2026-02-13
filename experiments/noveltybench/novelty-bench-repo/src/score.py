import argparse
import asyncio
import bisect
import functools
import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

CONCURRENT_REQUESTS = 1
EMBEDDING_MODEL_TYPE = Literal["local", "openai"]

reward_thresholds = [
    -7.71875,
    -6.28125,
    -6.0,
    -5.71875,
    -5.5,
    -5.0,
    -4.375,
    -3.4375,
    -2.046875,
]


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for scoring script."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        root_logger.setLevel(numeric_level)
    logger.setLevel(numeric_level)


def transform_raw_reward(reward: float) -> int:
    # score of 1 to 10
    return bisect.bisect_left(reward_thresholds, reward) + 1


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@functools.cache
def rm_and_tokenizer(
    model_path: str | None = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: str = "eager",
):
    """
    Load and cache the local reward model and tokenizer.

    Args:
        model_path: Optional path to local model directory. If None, downloads from HuggingFace.

    Returns:
        tuple: (model, tokenizer) for Skywork-Reward-Gemma-2-27B-v0.2
    """
    # Load model and tokenizer
    if model_path is None:
        model_path = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"

    logger.info(f"Loading reward model from {model_path}")
    dtype = DTYPE_MAP.get(torch_dtype, torch.bfloat16)
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        num_labels=1,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, fix_mistral_regex=True, use_fast=False
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return rm, tokenizer


@functools.cache
def get_openai_client():
    """
    Get and cache the OpenAI async client.

    Returns:
        AsyncOpenAI: OpenAI client configured with API key from environment
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set to use OpenAI embeddings"
        )
    # Strip whitespace/newlines from API key (common issue from copy-paste)
    api_key = api_key.strip()
    return AsyncOpenAI(api_key=api_key)


class Rating(BaseModel):
    rating: int


async def get_openai_embeddings(
    texts: list[str], model: str = "text-embedding-3-large"
) -> list[list[float]]:
    """
    Get embeddings from OpenAI API.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name

    Returns:
        List of embedding vectors

    Raises:
        Exception: If API call fails due to network issues, rate limits, or invalid responses
    """
    client = get_openai_client()
    try:
        response = await client.embeddings.create(input=texts, model=model)
        return [data.embedding for data in response.data]
    except Exception as e:
        raise RuntimeError(
            f"Failed to get embeddings from OpenAI API: {e}\n"
            f"Model: {model}, Number of texts: {len(texts)}"
        ) from e


def compute_embedding_similarity_score(
    prompt_embedding: list[float],
    generation_embedding: list[float],
    min_score: int = 1,
    max_score: int = 10,
) -> int:
    """
    Compute a score (1-10) based on cosine similarity between prompt and generation embeddings.

    Higher similarity indicates the generation is more relevant/aligned with the prompt.

    Args:
        prompt_embedding: Embedding vector for the prompt
        generation_embedding: Embedding vector for the generation
        min_score: Minimum score (default: 1)
        max_score: Maximum score (default: 10)

    Returns:
        Integer score between min_score and max_score
    """
    # Compute cosine similarity
    similarity = cosine_similarity(
        [prompt_embedding], [generation_embedding]
    )[0][0]

    # Map similarity [-1, 1] to score range [min_score, max_score]
    # Similarity is typically in [0, 1] for text embeddings, but can be negative
    normalized = (similarity + 1) / 2  # Map [-1, 1] to [0, 1]
    score = min_score + normalized * (max_score - min_score)

    return int(round(score))


async def score_partition_openai(
    prompt: str,
    generations: list[str],
    partition: list[int],
    model: str = "text-embedding-3-large",
) -> tuple[list[int], list[int]]:
    """
    Score generations using OpenAI embeddings and cosine similarity.

    Args:
        prompt: The input prompt
        generations: List of generated responses
        partition: Partition indices for each generation
        model: OpenAI embedding model name

    Returns:
        Tuple of (generation_scores, partition_scores)
    """
    # Prepare texts for embedding
    texts = [prompt] + generations
    embeddings = await get_openai_embeddings(texts, model=model)

    prompt_embedding = embeddings[0]
    generation_embeddings = embeddings[1:]

    # Compute similarity scores
    scores = [
        compute_embedding_similarity_score(prompt_embedding, gen_emb)
        for gen_emb in generation_embeddings
    ]

    generation_scores = []
    partition_scores = []

    for s, p in zip(scores, partition, strict=False):
        if p == len(partition_scores):
            generation_scores.append(s)
            partition_scores.append(s)
        else:
            generation_scores.append(0)

    assert len(partition_scores) == (max(partition) + 1), (
        f"partition_scores: {partition_scores}, partition: {partition}"
    )
    return generation_scores, partition_scores


@torch.inference_mode()
async def score_partition_rm(
    prompt: str,
    generations: list[str],
    partition: list[int],
    model_path: str | None = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: str = "eager",
) -> tuple[list[int], list[int]]:
    """
    Asynchronously scores the partition using local reward model.

    Args:
        prompt: The input prompt
        generations: List of generated responses
        partition: Partition indices for each generation
        model_path: Optional path to local model directory
        torch_dtype: Torch dtype string (e.g., 'bfloat16')
        device_map: Device map passed to Transformers (e.g., 'auto')
        attn_implementation: Attention backend ('eager' or 'sdpa')

    Returns:
        Tuple of (generation_scores, partition_scores)
    """
    rm, tokenizer = rm_and_tokenizer(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    convs = [
        [
            {"content": prompt, "role": "user"},
            {"content": generation, "role": "assistant"},
        ]
        for generation in generations
    ]
    batch = tokenizer.apply_chat_template(
        convs,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(rm.device)
    # Get the reward scores
    with torch.no_grad():
        raw_rewards = rm(**batch).logits[:, 0].tolist()

    scores = [transform_raw_reward(r) for r in raw_rewards]

    generation_scores = []
    partition_scores = []

    for s, p in zip(scores, partition, strict=False):
        if p == len(partition_scores):
            generation_scores.append(s)
            partition_scores.append(s)
        else:
            generation_scores.append(0)

    assert len(partition_scores) == (max(partition) + 1), (
        f"partition_scores: {partition_scores}, partition: {partition}"
    )
    return generation_scores, partition_scores


async def process_instances(
    instances,
    output_file,
    patience,
    scorer_type: str = "local",
    embedding_model: str = "text-embedding-3-large",
    scorer_model_path: str | None = None,
    scorer_torch_dtype: str = "bfloat16",
    scorer_device_map: str = "auto",
    scorer_attn_implementation: str = "eager",
):
    """
    Processes all instances concurrently and writes results to a file.

    Args:
        instances: Dataset instances to score
        output_file: Path to output file
        patience: Discount factor for computing cumulative utility
        scorer_type: Type of scorer to use ('local' or 'openai')
        embedding_model: OpenAI embedding model name (only used when scorer_type='openai')
        scorer_model_path: Local path to scorer model (only used when scorer_type='local')
    """
    # Check if file exists and has matching keys
    if os.path.exists(output_file):
        try:
            existing = json.loads(Path(output_file).read_text())
            if not set(instances["id"]) - {item["id"] for item in existing}:
                logger.info("All prompts already scored; skipping scoring stage")
                return
        except Exception:
            pass

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    results = [None] * len(instances)

    async def process_single_instance(idx: int, instance, pbar):
        async with semaphore:
            if scorer_type == "openai":
                generation_scores, partition_scores = await score_partition_openai(
                    instance["prompt"],
                    instance["generations"],
                    instance["partition"],
                    model=embedding_model,
                )
            else:  # scorer_type == "local"
                generation_scores, partition_scores = await score_partition_rm(
                    instance["prompt"],
                    instance["generations"],
                    instance["partition"],
                    model_path=scorer_model_path,
                    torch_dtype=scorer_torch_dtype,
                    device_map=scorer_device_map,
                    attn_implementation=scorer_attn_implementation,
                )
            utility = np.average(
                generation_scores,
                weights=patience ** np.arange(len(instance["generations"])),
            )
            pbar.update(1)
            return idx, {
                **instance,
                "generation_scores": generation_scores,
                "partition_scores": partition_scores,
                "utility": utility,
            }

    with tqdm(total=len(instances), desc="Scoring generations", unit="prompt") as pbar:
        tasks = [asyncio.create_task(process_single_instance(i, instance, pbar)) for i, instance in enumerate(instances)]
        for task in asyncio.as_completed(tasks):
            idx, result = await task
            results[idx] = result

    Path(output_file).write_text(json.dumps(results, indent=2))


async def main():
    parser = argparse.ArgumentParser(
        description="Score NoveltyBench generations using reward model or embeddings"
    )
    parser.add_argument(
        "--eval-dirs", 
        nargs="+",
        help="Directories containing evaluation files", 
        required=True
    )
    parser.add_argument(
        "--patience",
        help="Discount factor for computing cumulative utility.",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--scorer-type",
        help="Type of scorer to use: 'local' (Skywork-Reward-Gemma) or 'openai' (embeddings)",
        choices=["local", "openai"],
        default="local",
    )
    parser.add_argument(
        "--embedding-model",
        help="OpenAI embedding model to use (only for scorer-type=openai)",
        default="text-embedding-3-large",
    )
    parser.add_argument(
        "--scorer-model-path",
        help="Local path to scorer model directory (only for scorer-type=local)",
        default=None,
    )
    parser.add_argument(
        "--scorer-torch-dtype",
        choices=list(DTYPE_MAP.keys()),
        default="bfloat16",
        help="Torch dtype to use when loading the local scorer (default: bfloat16)",
    )
    parser.add_argument(
        "--scorer-device-map",
        default="auto",
        help="Device map passed to AutoModelForSequenceClassification (default: auto)",
    )
    parser.add_argument(
        "--scorer-attn-implementation",
        choices=["eager", "sdpa"],
        default="eager",
        help="Attention implementation for the local scorer (default: eager)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("NOVELTY_BENCH_LOG_LEVEL", "INFO"),
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for scoring run",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    # Initialize models once
    if args.scorer_type == "openai":
        logger.info(f"Using OpenAI embeddings with model {args.embedding_model}")
        # Validate API key is available
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set to use OpenAI embeddings.\n"
                "Please set it with: export OPENAI_API_KEY='your-api-key'"
            )
        # Strip whitespace/newlines and update environment variable
        api_key = api_key.strip()
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        model_info = args.scorer_model_path or "Skywork-Reward-Gemma-2-27B-v0.2 (from HuggingFace)"
        logger.debug(f"Using local reward model: {model_info}")
        
    # Expand directories recursively
    expanded_eval_dirs = []
    for eval_dir in args.eval_dirs:
        path = Path(eval_dir)
        if not path.exists():
            logger.warning(f"Directory not found: {path}")
            continue
            
        # Check if direct hit
        if (path / "partitions.json").exists():
            expanded_eval_dirs.append(path)
            continue
            
        # Recursive search (partitions.json)
        found = sorted(list(path.rglob("partitions.json")))
        if found:
             logger.info(f"Found {len(found)} partition files under {path}")
             expanded_eval_dirs.extend([p.parent for p in found])
        else:
             logger.warning(f"No partition files found in {path} or its subdirectories.")
             
    # Remove duplicates
    unique_dirs = []
    seen = set()
    for d in expanded_eval_dirs:
        if d not in seen:
            unique_dirs.append(d)
            seen.add(d)

    for eval_dir in unique_dirs:
        logger.debug(f"Processing directory: {eval_dir}")
        try:
            partitions_json = Path(eval_dir) / "partitions.json"

            if partitions_json.exists():
                partitions_path = partitions_json
            else:
                 logger.warning(
                    f"Missing partitions file at {partitions_json}. "
                    "Skipping this directory."
                )
                 continue

            instances = load_dataset(
                "json",
                data_files=str(partitions_path),
                split="train",
            )
            # Skip failed examples (missing/empty generations/partition or explicit failure_reason).
            original_len = len(instances)
            instances = instances.filter(
                lambda x: isinstance(x.get("generations"), list)
                and len(x.get("generations")) > 0
                and isinstance(x.get("partition"), list)
                and len(x.get("partition")) > 0
                and not x.get("failure_reason"),
            )
            if len(instances) == 0:
                logger.warning(
                    f"All examples were invalid/failed in {partitions_path}; skipping scoring."
                )
                continue
            if len(instances) != original_len:
                logger.warning(
                    f"Skipping {original_len - len(instances)}/{original_len} failed examples in {partitions_path}"
                )
            logger.debug(f"Loaded {len(instances)} partitioned prompts from {partitions_path}")

            os.makedirs(eval_dir, exist_ok=True)
            output_file = os.path.join(eval_dir, "scores.json")
            
            logger.debug(f"Processing {len(instances)} prompts for {eval_dir}")
            await process_instances(
                instances,
                output_file,
                args.patience,
                args.scorer_type,
                args.embedding_model,
                args.scorer_model_path,
                args.scorer_torch_dtype,
                args.scorer_device_map,
                args.scorer_attn_implementation,
            )
            logger.info(f"Scoring complete for {eval_dir}; results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to score directory {eval_dir}: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
