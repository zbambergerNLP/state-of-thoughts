import argparse
import asyncio
import functools
import json
import logging
import os
from pathlib import Path

import sacrebleu
import torch
from datasets import load_dataset
from evaluate import load
from pydantic import BaseModel
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import oai_client

logger = logging.getLogger(__name__)

CONCURRENT_REQUESTS = 1

client = oai_client()

rouge_scorer = rouge_scorer.RougeScorer(["rouge1"])
# BERTScore will be loaded with custom cache_dir if provided
bertscorer = None


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for partition script."""
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


def get_bertscorer(cache_dir=None):
    """
    Get or load BERTScore with optional cache directory.

    Args:
        cache_dir: Optional directory to cache BERTScore models

    Returns:
        BERTScore module
    """
    global bertscorer
    if bertscorer is None:
        if cache_dir:
            # Set environment variable for transformers cache
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            logger.info(f"Loading BERTScore model cache from {cache_dir}")
        else:
            logger.info("Loading BERTScore model with default cache directory")
        bertscorer = load("bertscore")
    return bertscorer


@functools.cache
def load_deberta_tokenizer_and_model(model_path=None):
    """
    Load DeBERTa model and tokenizer for classification.

    Args:
        model_path: Optional local path to model directory. If None, downloads from HuggingFace.

    Returns:
        tuple: (tokenizer, model)
    """
    logger.info("Loading DeBERTa model for classification (first run may take a few minutes)")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # If model_path provided, load from local directory
    if model_path:
        deberta_path = os.path.join(model_path, "deberta-v3-large-generation-similarity")
        tokenizer_path = os.path.join(model_path, "deberta-v3-large")

        logger.info(f"Loading DeBERTa weights from local path: {deberta_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, fix_mistral_regex=True, use_fast=False
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(deberta_path).to(DEVICE)
    else:
        logger.info("Downloading DeBERTa weights from HuggingFace")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/deberta-v3-large", fix_mistral_regex=True, use_fast=False,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        model = AutoModelForSequenceClassification.from_pretrained(
            "yimingzhang/deberta-v3-large-generation-similarity"
        ).to(DEVICE)

    model.eval()
    return tokenizer, model


async def bleu(prompt: str, s1: str, s2: str):
    return (
        sacrebleu.corpus_bleu([s1], [[s2]]).score
        + sacrebleu.corpus_bleu([s2], [[s1]]).score
    ) / 200


async def rouge1(prompt: str, s1: str, s2: str):
    rouge_eval = rouge_scorer.score(s1, s2)
    return rouge_eval["rouge1"].fmeasure


async def bertscore(prompt: str, s1: str, s2: str, cache_dir=None):
    scorer = get_bertscorer(cache_dir)
    return scorer.compute(
        predictions=[s1],
        references=[s2],
        model_type="microsoft/deberta-large",
    )["f1"][0]


@torch.inference_mode()
async def classifier_score(prompt: str, s1: str, s2: str, model_path=None):
    tokenizer, model = load_deberta_tokenizer_and_model(model_path)
    input_ids = [tokenizer.cls_token_id]
    for s in [s1, s2]:
        input_ids.extend(
            tokenizer.encode(
                s,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            )
        )
        input_ids.append(tokenizer.sep_token_id)
        prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
    token_type_ids = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    iids = torch.tensor(input_ids, device=DEVICE, dtype=torch.int64)
    tids = torch.tensor(token_type_ids, device=DEVICE, dtype=torch.int64)

    outputs = model(input_ids=iids.unsqueeze(0), token_type_ids=tids.unsqueeze(0))
    score = outputs["logits"].softmax(-1)[0, 1]
    return score.cpu().item()


async def equivalence_check_gpt4(prompt: str, response_0: str, response_1: str) -> bool:
    class Equivalence(BaseModel):
        equivalent: bool

    """Asynchronously checks equivalence between two responses."""
    messages = [
        {
            "role": "system",
            "content": "For a given prompt, determine whether the two responses are semantically equivalent.",
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "Prompt: " + prompt,
                    "Response A: " + response_0,
                    "Response B: " + response_1,
                ],
            ),
        },
    ]

    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            max_tokens=10,
            temperature=0,
            response_format=Equivalence,
        )
        return response.choices[0].message.parsed.equivalent
    except Exception as e:
        logger.warning(f"Error in GPT-4 equivalence check: {e}")
        return False


async def equivalence_check_unigram(
    prompt: str, response_0: str, response_1: str
) -> bool:
    return await rouge1(prompt, response_0, response_1) > 0.458


async def equivalence_check_bertscore(
    prompt: str,
    response_0: str,
    response_1: str,
    cache_dir=None,
) -> bool:
    scores = await bertscore(prompt, response_0, response_1, cache_dir)
    return scores["f1"][0] > 0.719


def maybe_test_equality(response_0: str, response_1: str) -> bool | None:
    unigram_0 = response_0.strip().lower().split()
    unigram_1 = response_1.strip().lower().split()
    max_len = max(len(unigram_0), len(unigram_1))
    if max_len <= 5:
        common_unigrams = set(unigram_0) & set(unigram_1)
        return len(common_unigrams) * 2 >= max_len

    return None


async def equivalence_check_classifier(
    prompt: str,
    response_0: str,
    response_1: str,
    model_path=None,
) -> bool:
    equality = maybe_test_equality(response_0, response_1)
    if equality is not None:
        return equality
    score = await classifier_score(prompt, response_0, response_1, model_path)
    return score > 0.102


async def partition_responses(
    prompt: str,
    responses: list[str],
    equivalence_alg,
) -> list[int]:
    """Partitions responses into equivalence classes."""
    equivalence_classes = []
    partition = [-1] * len(responses)

    for i in range(len(responses)):
        if partition[i] >= 0:
            continue

        current_class = [responses[i]]
        partition[i] = len(equivalence_classes)

        for j in range(i + 1, len(responses)):
            if partition[j] == -1 and await equivalence_alg(
                prompt,
                current_class[0],
                responses[j],
            ):
                current_class.append(responses[j])
                partition[j] = len(equivalence_classes)

        equivalence_classes.append(current_class)

    assert all(p >= 0 for p in partition)
    return partition


EQUIVALENCE_ALGS = {
    "gpt4": equivalence_check_gpt4,
    "unigram": equivalence_check_unigram,
    "bertscore": equivalence_check_bertscore,
    "classifier": equivalence_check_classifier,
}


async def process_instances(instances, output_file, equivalence_alg):
    """Processes all instances concurrently and writes results to a JSON file."""
    # Check if file exists and has matching keys
    if os.path.exists(output_file):
        try:
            existing = json.loads(Path(output_file).read_text())
            if not set(instances["id"]) - {item["id"] for item in existing}:
                logger.info("All prompts already partitioned; skipping partition stage")
                return
        except Exception:
            # Handle empty or invalid files
            pass

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    results = [None] * len(instances)

    async def process_single_instance(idx: int, instance):
        async with semaphore:
            partition = await partition_responses(
                instance["prompt"],
                instance["generations"],
                equivalence_alg,
            )
            return idx, {**instance, "partition": partition, "distinct": max(partition)}

    tasks = [asyncio.create_task(process_single_instance(i, instance)) for i, instance in enumerate(instances)]

    for task in tqdm(asyncio.as_completed(tasks), total=len(instances)):
        idx, result = await task
        results[idx] = result

    Path(output_file).write_text(json.dumps(results, indent=2))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg",
        default="classifier",
        help="Equivalence testing method",
        choices=EQUIVALENCE_ALGS,
    )
    parser.add_argument(
        "--eval-dirs", 
        nargs="+",
        help="Directories to save evaluation results", 
        required=True
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to load partition models from (BERTScore, DeBERTa)",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("NOVELTY_BENCH_LOG_LEVEL", "INFO"),
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for partition run",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    equivalence_alg = EQUIVALENCE_ALGS[args.alg]

    if args.cache_dir:
        if args.alg == "classifier":
            # Pre-load DeBERTa model with the specified path
            load_deberta_tokenizer_and_model(args.cache_dir)
            # Bind the model_path to the classifier function
            equivalence_alg = functools.partial(equivalence_check_classifier, model_path=args.cache_dir)
            logger.info(f"Using partition models from {args.cache_dir}")
        elif args.alg == "bertscore":
            get_bertscorer(args.cache_dir)
            equivalence_alg = functools.partial(equivalence_check_bertscore, cache_dir=args.cache_dir)
            logger.info(f"Using BERTScore models from {args.cache_dir}")

    # Expand directories recursively
    expanded_eval_dirs = []
    for eval_dir in args.eval_dirs:
        path = Path(eval_dir)
        if not path.exists():
            logger.warning(f"Directory not found: {path}")
            continue
            
        # Check if direct hit
        if (path / "generations.json").exists():
            expanded_eval_dirs.append(path)
            continue
            
        # Recursive search
        found = sorted(list(path.rglob("generations.json")))
        if found:
            logger.info(f"Found {len(found)} generation files under {path}")
            expanded_eval_dirs.extend([p.parent for p in found])
        else:
             logger.warning(f"No generation files found in {path} or its subdirectories.")

    if not expanded_eval_dirs and args.eval_dirs:
        raise ValueError("No valid directories found to process.")

    # Remove duplicates while preserving order
    unique_dirs = []
    seen = set()
    for d in expanded_eval_dirs:
        if d not in seen:
            unique_dirs.append(d)
            seen.add(d)
            
    for eval_dir in unique_dirs:
        logger.info(f"Processing directory: {eval_dir}")
        generations_json = Path(eval_dir) / "generations.json"

        if generations_json.exists():
            generations_path = generations_json
        else:
            raise ValueError(f"Missing generations file at {generations_json}.")

        logger.info(f"Loading generations from {generations_path}")
        instances = load_dataset(
            "json",
            data_files=str(generations_path),
            split="train",
        )
        # Skip failed examples (empty generations or explicit failure_reason).
        # This allows partitioning/scoring to proceed on the rest of the dataset.
        original_len = len(instances)
        instances = instances.filter(
            lambda x: isinstance(x.get("generations"), list)
            and len(x.get("generations")) > 0
            and not x.get("failure_reason"),
        )
        if len(instances) == 0:
            logger.warning(
                f"All examples were invalid/failed in {generations_path}; skipping partition."
            )
            continue
        if len(instances) != original_len:
            logger.warning(
                f"Skipping {original_len - len(instances)}/{original_len} failed examples in {generations_path}"
            )
        logger.info(
            "Loaded %d prompts with %d generations each",
            len(instances),
            len(instances[0]["generations"]) if len(instances) else 0,
        )

        # Process instances and save results
        output_file = os.path.join(eval_dir, "partitions.json")
        logger.info(f"Starting partition processing with {args.alg} algorithm")
        logger.info(f"Processing {len(instances)} prompts")
        await process_instances(instances, output_file, equivalence_alg)
        logger.info(f"Partition complete; results saved to {output_file}")
        

if __name__ == "__main__":
    asyncio.run(main())