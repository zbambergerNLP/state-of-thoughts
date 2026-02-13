"""
Pairwise argument comparison using GPT-5-mini with Bradley-Terry scoring.

This script evaluates arguments pairwise using an LLM judge and computes
Bradley-Terry scores to rank arguments by persuasiveness.

Usage:
    # Using synthesis_type convenience flag (recommended for explainability experiment)
    python experiments/argument_generation/pairwise_evaluation.py \
        --synthesis_type strict --num_comparisons 10000

    # Using explicit paths
    python experiments/argument_generation/pairwise_evaluation.py \
        --input_csv path/to/results.csv --output_jsonl path/to/output.jsonl

Resume after crash:
    # Just re-run the same command - it will skip already-evaluated pairs
    python experiments/argument_generation/pairwise_evaluation.py \
        --synthesis_type strict --num_comparisons 10000

Calculate scores from existing comparisons only:
    python experiments/argument_generation/pairwise_evaluation.py \
        --synthesis_type strict --calculate_bt_only

Flags:
    --synthesis_type {strict,faithful,restructured}
        Synthesis type for explainability experiment. If provided, auto-sets
        input/output paths to explainability directory structure:
          input:  explainability/synthesis_{type}/explainability_synthesis_{type}_all_results.csv
          output: explainability/synthesis_{type}/pairwise_comparisons.jsonl
        Overrides --input_csv and --output_jsonl if specified.

    --input_csv PATH
        Path to CSV file containing arguments to evaluate.
        Required columns: topic, stance, final_argument.
        Optional columns: persona_name (included in output if present).
        Default: experiments/argument_generation/explainability/synthesis_strict/explainability_synthesis_strict_all_results.csv

    --output_jsonl PATH
        Path to JSONL file for storing pairwise comparison results.
        Each line contains: arg_a_id, arg_b_id, swapped, raw_response, winner_id.
        Default: experiments/argument_generation/explainability/synthesis_strict/pairwise_comparisons.jsonl

    --num_comparisons N
        Number of pairwise comparisons to perform.
        With 5000 arguments, 10000 comparisons gives ~2 comparisons per argument.
        Default: 10000

    --model MODEL
        OpenAI model to use as the judge.
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
    OPENAI_API_KEY must be set for API access.

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
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from scipy.optimize import minimize
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
	client: AsyncOpenAI,
	semaphore: asyncio.Semaphore,
	arg_a_id: int,
	arg_b_id: int,
	arg_a_text: str,
	arg_b_text: str,
	model: str,
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
			response = await client.chat.completions.create(
				model=model,
				messages=[{"role": "user", "content": prompt}],
				max_completion_tokens=1000,
			)

		choice = response.choices[0]
		content = choice.message.content
		raw_choice = (content or "").strip().upper()

		# Check if we got a valid response
		if "A" in raw_choice or "B" in raw_choice:
			break

		# Debug: log the actual response and finish reason
		logger.warning(
			"Attempt %d for pair (%d, %d): content='%s', finish_reason='%s'",
			attempt + 1,
			arg_a_id,
			arg_b_id,
			content,
			choice.finish_reason,
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


async def run_comparisons(
	df: pd.DataFrame,
	pairs: list[tuple[int, int]],
	output_path: Path,
	model: str,
	max_concurrent: int,
) -> None:
	"""Run all pairwise comparisons asynchronously."""
	client = AsyncOpenAI()
	semaphore = asyncio.Semaphore(max_concurrent)

	# Create lookup for argument text
	arg_texts = df.set_index("arg_id")["final_argument"].to_dict()

	async def process_and_save(pair: tuple[int, int]) -> None:
		arg_a_id, arg_b_id = pair
		swap = random.choice([True, False])
		result = await evaluate_pair(
			client=client,
			semaphore=semaphore,
			arg_a_id=arg_a_id,
			arg_b_id=arg_b_id,
			arg_a_text=arg_texts[arg_a_id],
			arg_b_text=arg_texts[arg_b_id],
			model=model,
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
	jsonl_path: Path, n_arguments: int, regularization: float = 0.0
) -> dict[int, float]:
	"""Compute Bradley-Terry scores via direct MLE optimization.

	Uses L-BFGS-B to maximize the log-likelihood directly, which converges
	much faster than the iterative MM algorithm.

	Args:
		jsonl_path: Path to JSONL file with pairwise comparisons.
		n_arguments: Number of arguments.
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

	# Pre-extract winner/loser pairs for efficiency
	matches = []
	for rec in comparisons:
		a, b = rec["arg_a_id"], rec["arg_b_id"]
		winner = rec["winner_id"]
		loser = b if winner == a else a
		matches.append((winner, loser))

	def neg_log_likelihood(ratings: np.ndarray) -> float:
		"""Negative log-likelihood with L2 regularization."""
		ll = 0.0
		for winner, loser in matches:
			# P(winner > loser) = exp(r_w) / (exp(r_w) + exp(r_l))
			# log P = r_w - log(exp(r_w) + exp(r_l))
			diff = ratings[winner] - ratings[loser]
			ll += diff - np.logaddexp(0, diff)  # log(sigmoid(diff))
		# L2 regularization to prevent extreme ratings
		penalty = regularization * np.sum(ratings**2)
		return -ll + penalty

	def gradient(ratings: np.ndarray) -> np.ndarray:
		"""Gradient of negative log-likelihood with L2 regularization."""
		grad = np.zeros(n_arguments)
		for winner, loser in matches:
			diff = ratings[winner] - ratings[loser]
			prob_loser = 1.0 / (1.0 + np.exp(diff))  # P(loser wins) = sigmoid(-diff)
			# For negative log-likelihood:
			# d(-L)/d(r_winner) = -prob_loser (want to increase winner rating)
			# d(-L)/d(r_loser) = +prob_loser (want to decrease loser rating)
			grad[winner] -= prob_loser
			grad[loser] += prob_loser
		# L2 regularization gradient
		grad += 2 * regularization * ratings
		return grad

	# Initialize ratings to zero (all equal)
	init_ratings = np.zeros(n_arguments)

	# Optimize using L-BFGS-B
	result = minimize(
		neg_log_likelihood,
		init_ratings,
		method="L-BFGS-B",
		jac=gradient,
		options={"maxiter": 1000},
	)

	logger.info(
		"Optimization %s after %d iterations",
		"converged" if result.success else "stopped",
		result.nit,
	)

	# Convert log-ratings to strength parameters (exp and normalize to mean=1)
	strengths = np.exp(result.x)
	strengths = strengths / strengths.mean()

	return {i: float(strengths[i]) for i in range(n_arguments)}


def main() -> None:
	parser = argparse.ArgumentParser(description="Pairwise argument evaluation")
	parser.add_argument(
		"--synthesis_type",
		type=str,
		choices=["strict", "faithful", "restructured"],
		default=None,
		help="Synthesis type. If provided, auto-sets input/output paths to explainability directory.",
	)
	parser.add_argument(
		"--input_csv",
		type=str,
		default="experiments/argument_generation/explainability/synthesis_strict/explainability_synthesis_strict_all_results.csv",
	)
	parser.add_argument(
		"--output_jsonl",
		type=str,
		default="experiments/argument_generation/explainability/synthesis_strict/pairwise_comparisons.jsonl",
	)
	parser.add_argument("--num_comparisons", type=int, default=10000)
	parser.add_argument("--model", type=str, default="gpt-5-mini-2025-08-07")
	parser.add_argument("--max_concurrent", type=int, default=50)
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
			f"experiments/argument_generation/explainability/synthesis_{args.synthesis_type}"
		)
		input_path = (
			base_dir / f"explainability_synthesis_{args.synthesis_type}_all_results.csv"
		)
		output_path = base_dir / "pairwise_comparisons.jsonl"
	else:
		input_path = Path(args.input_csv)
		output_path = Path(args.output_jsonl)

	# Load arguments
	logger.info("Loading arguments from %s", input_path)
	df = load_arguments(input_path)
	n_arguments = len(df)
	logger.info("Loaded %d arguments", n_arguments)

	if not args.calculate_bt_only:
		# Load completed pairs
		completed = load_completed_pairs(output_path)
		logger.info("Found %d already-evaluated pairs", len(completed))

		# Generate new pairs
		pairs = generate_pairs(n_arguments, args.num_comparisons, completed, args.seed)
		logger.info("Generated %d new pairs to evaluate", len(pairs))

		if pairs:
			asyncio.run(
				run_comparisons(
					df=df,
					pairs=pairs,
					output_path=output_path,
					model=args.model,
					max_concurrent=args.max_concurrent,
				)
			)

	# Compute Bradley-Terry scores
	if output_path.exists():
		logger.info("Computing Bradley-Terry scores...")
		scores = compute_bradley_terry(output_path, n_arguments)

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
