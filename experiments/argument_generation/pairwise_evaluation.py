"""
Pairwise argument comparison with Bradley-Terry scoring.

This script evaluates arguments pairwise using an LLM judge and computes
Bradley-Terry scores to rank arguments by persuasiveness. Supports
OpenAI, Google (Gemini), and Anthropic as judge providers.


Usage:
    # Using synthesis_type convenience flag (recommended for argument generation experiment)
    python experiments/argument_generation/pairwise_evaluation.py \
        --synthesis_type strict --num_comparisons 10000

    # Using a different judge provider
    python experiments/argument_generation/pairwise_evaluation.py \
        --synthesis_type strict --provider google --model gemini-3.1-flash-lite-preview

	for st in strict faithful restructured; do
		python experiments/argument_generation/pairwise_evaluation.py \
			--synthesis_type "$st" \
			--topic standardized_testing \
			--provider google \
			--model gemini-3.1-flash-lite-preview \
			--max_concurrent 10 \
			--num_comparisons 50000
	done

	for st in strict faithful restructured; do
		python experiments/argument_generation/pairwise_evaluation.py \
			--synthesis_type "$st" \
			--topic meat_tax \
			--provider anthropic \
			--model claude-haiku-4-5-20251001 \
			--max_concurrent 5 \
			--num_comparisons 50000
	done


Resume after crash:
    # Just re-run the same command - it will skip already-evaluated pairs
    python experiments/argument_generation/pairwise_evaluation.py \
        --synthesis_type strict --num_comparisons 10000

Calculate scores from existing comparisons only:
    python experiments/argument_generation/pairwise_evaluation.py \
        --synthesis_type strict --calculate_bt_only

Flags:
    --synthesis_type {strict,faithful,restructured}
        Synthesis type for argument generation experiment. If provided, auto-sets
        input/output paths to argument_data directory structure:
          input:  argument_data/{topic}/synthesis_{type}/{topic}_synthesis_{type}_all_results.csv
          output: argument_data/synthesis_{type}/pairwise_comparisons.jsonl
        Overrides --input_csv and --output_jsonl if specified.

    --input_csv PATH
        Path to CSV file containing arguments to evaluate.
        Required columns: topic, stance, final_argument.
        Optional columns: persona_name (included in output if present).
        Default: experiments/argument_generation/argument_data/{topic}/synthesis_strict/{topic}_synthesis_strict_all_results.csv

    --output_jsonl PATH
        Path to JSONL file for storing pairwise comparison results.
        Each line contains: arg_a_id, arg_b_id, swapped, raw_response, winner_id.
        Default: experiments/argument_generation/argument_data/synthesis_strict/pairwise_comparisons.jsonl

    --num_comparisons N
        Number of pairwise comparisons to perform.
        With 5000 arguments, 10000 comparisons gives ~2 comparisons per argument.
        Default: 10000

    --provider {openai,google,anthropic}
        LLM provider to use as the judge.
        Default: openai

    --model MODEL
        Model to use as the judge.
        Default: gpt-5-mini-2025-08-07

    --max_concurrent N
        Maximum number of concurrent API requests.
        Default: 50

    --seed N
        Random seed for reproducible pair generation.
        Default: 42

    --calculate_bt_only
        Skip running comparisons; only compute Bradley-Terry scores from
        existing JSONL file. Useful for re-analyzing results.

Environment:
    Set the appropriate API key for your provider:
    - OpenAI: OPENAI_API_KEY
    - Google: GOOGLE_API_KEY
    - Anthropic: ANTHROPIC_API_KEY

Output:
    - JSONL file with pairwise comparison results (append-mode, resumable)
    - CSV file (<output_jsonl_stem>_bt_scores.csv) with all arguments and bt_score column
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if TYPE_CHECKING:
	from experiments.argument_generation.llm_judge import AsyncLLMJudge

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """Which argument is more persuasive? You MUST choose one. Reply with ONLY "A" or "B".

Argument A:
{arg_a}

Argument B:
{arg_b}"""


def load_arguments(csv_path: Path) -> pd.DataFrame:
	"""Load arguments from CSV, preserving all columns."""
	df = pd.read_csv(csv_path)
	df["arg_id"] = df.index
	return df


def load_completed_pairs(jsonl_path: Path) -> set[tuple[int, int]]:
	"""Load already-evaluated pairs from JSONL."""
	completed = set()
	if jsonl_path.exists():
		with open(jsonl_path) as f:
			for line in f:
				rec = json.loads(line)
				completed.add((rec["arg_a_id"], rec["arg_b_id"]))
	return completed


def generate_pairs(
	n_arguments: int, num_comparisons: int, completed: set[tuple[int, int]], seed: int
) -> list[tuple[int, int]]:
	"""Generate random pairs, excluding already-completed ones."""
	random.seed(seed)
	all_pairs = []
	attempts = 0
	max_attempts = num_comparisons * 10

	while len(all_pairs) < num_comparisons and attempts < max_attempts:
		a = random.randint(0, n_arguments - 1)
		b = random.randint(0, n_arguments - 1)
		if a == b:
			continue
		# Normalize order for deduplication
		pair = (min(a, b), max(a, b))
		if pair not in completed and pair not in set(all_pairs):
			all_pairs.append(pair)
		attempts += 1

	return all_pairs


async def evaluate_pair(
	judge: AsyncLLMJudge,
	semaphore: asyncio.Semaphore,
	arg_a_id: int,
	arg_b_id: int,
	arg_a_text: str,
	arg_b_text: str,
	swap: bool,
	max_retries: int = 3,
) -> dict | None:
	"""Evaluate a single pair with the judge model.

	Returns None if the model fails to provide a valid response after retries.
	"""

	# Optionally swap to reduce position bias
	if swap:
		prompt = JUDGE_PROMPT.format(arg_a=arg_b_text, arg_b=arg_a_text)
	else:
		prompt = JUDGE_PROMPT.format(arg_a=arg_a_text, arg_b=arg_b_text)

	raw_choice = ""
	for attempt in range(max_retries):
		async with semaphore:
			content = await judge.complete(prompt)

		raw_choice = content.strip().upper()

		# Check if we got a valid response
		if "A" in raw_choice or "B" in raw_choice:
			break

		# Debug: log the actual response
		logger.warning(
			"Attempt %d for pair (%d, %d): content='%s'",
			attempt + 1,
			arg_a_id,
			arg_b_id,
			content,
		)

		# Retry with exponential backoff
		if attempt < max_retries - 1:
			await asyncio.sleep(2**attempt)

	# Parse response
	if "B" in raw_choice and "A" not in raw_choice:
		slot_winner = "B"
	elif "A" in raw_choice:
		slot_winner = "A"
	else:
		# No valid response after retries - skip this pair
		logger.warning(
			"Skipping pair (%d, %d): no valid response after %d retries",
			arg_a_id,
			arg_b_id,
			max_retries,
		)
		return None

	# Unswap if needed
	if swap:
		winner_id = arg_b_id if slot_winner == "A" else arg_a_id
	else:
		winner_id = arg_a_id if slot_winner == "A" else arg_b_id

	return {
		"arg_a_id": arg_a_id,
		"arg_b_id": arg_b_id,
		"swapped": swap,
		"raw_response": raw_choice,
		"winner_id": winner_id,
	}


class _RateLimiter:
	"""Token-bucket rate limiter for async requests."""

	def __init__(self, max_rpm: int) -> None:
		self.enabled = max_rpm > 0
		if self.enabled:
			self.interval = 60.0 / max_rpm
			self._lock = asyncio.Lock()
			self._last = 0.0

	async def acquire(self) -> None:
		if not self.enabled:
			return
		async with self._lock:
			import time

			now = time.monotonic()
			wait = self._last + self.interval - now
			if wait > 0:
				await asyncio.sleep(wait)
			self._last = time.monotonic()


async def run_comparisons(
	df: pd.DataFrame,
	pairs: list[tuple[int, int]],
	output_path: Path,
	judge: AsyncLLMJudge,
	max_concurrent: int,
	max_rpm: int = 0,
) -> None:
	"""Run all pairwise comparisons asynchronously."""
	semaphore = asyncio.Semaphore(max_concurrent)
	rate_limiter = _RateLimiter(max_rpm)

	# Create lookup for argument text
	arg_texts = df.set_index("arg_id")["final_argument"].to_dict()

	async def process_and_save(pair: tuple[int, int]) -> None:
		await rate_limiter.acquire()
		arg_a_id, arg_b_id = pair
		swap = random.choice([True, False])
		result = await evaluate_pair(
			judge=judge,
			semaphore=semaphore,
			arg_a_id=arg_a_id,
			arg_b_id=arg_b_id,
			arg_a_text=arg_texts[arg_a_id],
			arg_b_text=arg_texts[arg_b_id],
			swap=swap,
		)
		# Skip if no valid response after retries
		if result is None:
			return
		# Append immediately to JSONL
		with open(output_path, "a") as f:
			f.write(json.dumps(result) + "\n")

	tasks = [process_and_save(pair) for pair in pairs]
	await tqdm_asyncio.gather(*tasks, desc="Evaluating pairs")


def compute_bradley_terry(
	jsonl_path: Path, regularization: float = 0.0
) -> dict[int, float]:
	"""Compute Bradley-Terry scores via sklearn LogisticRegression.

	Encodes each pairwise comparison as a row in a sparse design matrix
	with +1 at the winner column and -1 at the loser column, then fits
	logistic regression to recover log-strength parameters.

	Handles sparse arg_ids natively — no remapping required by callers.

	Args:
		jsonl_path: Path to JSONL file with pairwise comparisons.
		regularization: L2 penalty on log-ratings to prevent extreme values.
			Set to 0.0 (default) for pure MLE. Higher values shrink ratings
			toward zero (equal strength).
	"""
	# Load all comparisons
	comparisons = []
	with open(jsonl_path) as f:
		for line in f:
			comparisons.append(json.loads(line))

	logger.info("Loaded %d pairwise comparisons", len(comparisons))

	# Collect all unique arg_ids and build a dense index mapping
	all_ids: set[int] = set()
	for rec in comparisons:
		all_ids.add(rec["arg_a_id"])
		all_ids.add(rec["arg_b_id"])
	id_to_col = {arg_id: idx for idx, arg_id in enumerate(sorted(all_ids))}
	n_players = len(id_to_col)

	# Build sparse design matrix: each match contributes two rows
	# (winner=+1, loser=-1) with alternating y labels to ensure both
	# classes are present for sklearn.
	rows, cols, vals = [], [], []
	y = []
	for i, rec in enumerate(comparisons):
		a, b = rec["arg_a_id"], rec["arg_b_id"]
		winner_id = rec["winner_id"]
		loser_id = b if winner_id == a else a
		w_col, l_col = id_to_col[winner_id], id_to_col[loser_id]

		# Row with y=1: winner gets +1, loser gets -1
		row_idx = 2 * i
		rows.extend([row_idx, row_idx])
		cols.extend([w_col, l_col])
		vals.extend([1.0, -1.0])
		y.append(1)

		# Row with y=0: flip signs (loser gets +1, winner gets -1)
		row_idx = 2 * i + 1
		rows.extend([row_idx, row_idx])
		cols.extend([l_col, w_col])
		vals.extend([1.0, -1.0])
		y.append(0)

	design = csr_matrix(
		(vals, (rows, cols)), shape=(2 * len(comparisons), n_players)
	)
	y_arr = np.array(y)

	# Fit logistic regression (equivalent to Bradley-Terry MLE)
	n_matches = len(comparisons)
	reg_c = (
		1.0 / (2 * regularization * n_matches) if regularization > 0 else np.inf
	)
	model = LogisticRegression(
		fit_intercept=False, C=reg_c, solver="lbfgs", max_iter=1000
	)
	model.fit(design, y_arr)

	logger.info(
		"LogisticRegression converged in %d iterations", model.n_iter_[0]
	)

	# Convert log-ratings to strength parameters (exp and normalize to mean=1)
	log_ratings = model.coef_[0]
	strengths = np.exp(log_ratings)
	strengths = strengths / strengths.mean()

	# Map back to original arg_ids
	col_to_id = {idx: arg_id for arg_id, idx in id_to_col.items()}
	return {col_to_id[i]: float(strengths[i]) for i in range(n_players)}


def main() -> None:
	parser = argparse.ArgumentParser(description="Pairwise argument evaluation")
	parser.add_argument(
		"--synthesis_type",
		type=str,
		choices=["strict", "faithful", "restructured"],
		default=None,
		help="Synthesis type. If provided, auto-sets input/output paths to argument_data directory.",
	)
	parser.add_argument(
		"--topic",
		type=str,
		default="single_use_plastic_specific_subtopics",
		help="Topic subdirectory name (default: single_use_plastic_specific_subtopics).",
	)
	parser.add_argument(
		"--input_csv",
		type=str,
		default=None,
	)
	parser.add_argument(
		"--output_jsonl",
		type=str,
		default=None,
	)
	parser.add_argument(
		"--num_comparisons", type=int, default=10000,
		help="Total target number of comparisons (including already completed).",
	)
	parser.add_argument(
		"--provider",
		type=str,
		choices=["openai", "google", "anthropic"],
		default="openai",
		help="LLM provider for the judge (default: openai).",
	)
	parser.add_argument("--model", type=str, default="gpt-5-mini-2025-08-07")
	parser.add_argument("--max_concurrent", type=int, default=50)
	parser.add_argument(
		"--max_rpm", type=int, default=0,
		help="Max requests per minute (0 = unlimited). Set to stay under API rate limits.",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--calculate_bt_only",
		action="store_true",
		help="Skip comparisons, just compute BT scores from existing JSONL",
	)
	args = parser.parse_args()

	# Handle synthesis_type convenience flag
	if args.synthesis_type:
		base_dir = Path(
			f"experiments/argument_generation/argument_data/{args.topic}/synthesis_{args.synthesis_type}"
		)
		input_path = (
			base_dir / f"{args.topic}_synthesis_{args.synthesis_type}_all_results.csv"
		)
		output_path = base_dir / "pairwise_comparisons.jsonl"
	elif args.input_csv and args.output_jsonl:
		input_path = Path(args.input_csv)
		output_path = Path(args.output_jsonl)
	else:
		# Default paths with topic
		base_dir = Path(
			f"experiments/argument_generation/argument_data/{args.topic}/synthesis_strict"
		)
		input_path = base_dir / f"{args.topic}_synthesis_strict_all_results.csv"
		output_path = base_dir / "pairwise_comparisons.jsonl"

	# Load arguments
	logger.info("Loading arguments from %s", input_path)
	df = load_arguments(input_path)
	n_arguments = len(df)
	logger.info("Loaded %d arguments", n_arguments)

	if not args.calculate_bt_only:
		from experiments.argument_generation.llm_judge import create_judge

		judge = create_judge(args.provider, args.model)
		seed = args.seed

		# Loop until we reach the target number of successful comparisons.
		# Some evaluations may fail (judge returns invalid response), so we
		# re-check and generate replacement pairs as needed.
		while True:
			completed = load_completed_pairs(output_path)
			remaining = max(0, args.num_comparisons - len(completed))
			logger.info(
				"Completed: %d / %d (need %d more)",
				len(completed), args.num_comparisons, remaining,
			)
			if remaining == 0:
				break

			# Use a different seed when backfilling so we don't replay the
			# same RNG sequence that produced the original (now completed)
			# pairs, which would waste all attempts on collisions.
			backfill_seed = seed + len(completed)
			pairs = generate_pairs(
				n_arguments, remaining, completed, backfill_seed
			)
			logger.info("Generated %d new pairs to evaluate", len(pairs))
			if not pairs:
				logger.warning("Could not generate more unique pairs, stopping")
				break

			asyncio.run(
				run_comparisons(
					df=df,
					pairs=pairs,
					output_path=output_path,
					judge=judge,
					max_concurrent=args.max_concurrent,
					max_rpm=args.max_rpm,
				)
			)

	# Compute Bradley-Terry scores
	if output_path.exists():
		logger.info("Computing Bradley-Terry scores...")
		scores = compute_bradley_terry(output_path)

		# Merge with metadata
		df["bt_score"] = df["arg_id"].map(scores)
		results = df.sort_values("bt_score", ascending=False)

		# Save scores to CSV
		scores_path = output_path.with_suffix(".csv").with_stem(
			output_path.stem + "_bt_scores"
		)
		results.to_csv(scores_path, index=False)
		logger.info("Saved Bradley-Terry scores to %s", scores_path)

		# Log summary
		logger.info("Top 10 arguments by BT score:")
		display_cols = ["arg_id", "topic", "stance", "bt_score"]
		if "persona_name" in results.columns:
			display_cols.insert(3, "persona_name")
		logger.info("\n%s", results.head(10)[display_cols])


if __name__ == "__main__":
	main()
