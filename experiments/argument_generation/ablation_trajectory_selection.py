"""Unified Trajectory Selection & Argument Generation

Combines trajectory selection with argument generation in one script.
Supports three selection modes: targeted (M2), random, and m1b (top-3-topics).

Three trajectory selection modes:
1. targeted: Uses M2 model predictions to select best trajectories
2. random: Randomly sampled trajectories from unobserved space
3. m1b: Trajectories using only top 3 content topics from M1b model

All modes exclude observed trajectories (n_observed > 0). Random does NOT exclude
M2 top-50 (true random baseline).

Usage:
    # Run all modes for all synthesis types
    python experiments/argument_generation/ablation_trajectory_selection.py \
        --synthesis_type all \
        --selection_mode all \
        --top_n 50 \
        --samples_per_trajectory 5 \
        --model Qwen3-30B-A3B-Instruct-2507 \
        --reranker_model Qwen3-Reranker-8B \
        --generative_gpu_index 0 \
        --reranker_gpu_index 1

    # Run specific mode
    python experiments/argument_generation/ablation_trajectory_selection.py \
        --synthesis_type strict \
        --selection_mode random
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import dspy
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from adapter.constraints import ResponseLength
from constants import Verbosity
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.scoring_local_lm import ScoringLocalVLLM
from predict.tree_of_thoughts import TreeOfThoughts
from predict.tree_of_thoughts.tree_of_thoughts import TreeOfThoughtsOutput
from predict.tree_of_thoughts.tree_parameters import TreeOfThoughtsParameters
from signatures.example_signatures import GenerateArgumentWithReasoningAndPersona

load_dotenv()

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

SYNTHESIS_TYPES = ["strict", "faithful", "restructured"]
SELECTION_MODES = ["targeted", "random", "m1b"]

# Train/test split constants (matching m1_vs_m2_analysis.py for comparability)
RANDOM_STATE = 42
TEST_SIZE = 0.4

# Fixed topic and stance (same as persona_explainability_experiment.py)
TOPIC = "The government should enforce a total ban on single-use plastics."
STANCE = "PRO"

# Load vocabularies
SCRIPT_DIR = Path(__file__).parent.resolve()
ACTION_SPACE_DIR = SCRIPT_DIR / "action_space"

with open(ACTION_SPACE_DIR / "structures.json") as f:
	STRUCTURES = list(json.load(f)["choices"].keys())

TOPIC_SUBTOPICS_FILES = {
	"single_use_plastic_specific_subtopics": "subtopics_specific_single_use_plastic.json",
	"standardized_testing": "subtopics_specific_standardized_testing.json",
	"meat_tax": "subtopics_specific_meat_tax.json",
	"social_media_age_restriction": "subtopics_specific_social_media_age_restriction.json",
	"universal_basic_income": "subtopics_specific_universal_basic_income.json",
}
SUBTOPICS: list[str] = []


# =============================================================================
# Trajectory Selection Functions
# =============================================================================


def load_rankings(synthesis_type: str, topic: str) -> pd.DataFrame:
	"""Load M2 trajectory rankings for a synthesis type.

	Tries feather format first, falls back to CSV for backwards compatibility.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		topic: Topic subdirectory name (e.g. single_use_plastic).

	Returns:
		DataFrame with trajectory rankings.
	"""
	base_path = Path(
		f"experiments/argument_generation/argument_data/{topic}/synthesis_{synthesis_type}"
		f"/m2_trajectory_rankings_{synthesis_type}"
	)
	feather_path = base_path.with_suffix(".feather")
	csv_path = base_path.with_suffix(".csv")

	if feather_path.exists():
		logger.info("Loading rankings from %s", feather_path)
		return pd.read_feather(feather_path)
	elif csv_path.exists():
		logger.info("Loading rankings from %s", csv_path)
		return pd.read_csv(csv_path)
	else:
		raise FileNotFoundError(
			f"No rankings file found at {feather_path} or {csv_path}"
		)


def build_exclusion_set(
	rankings: pd.DataFrame,
	exclude_m2_top_n: int = 0,
) -> set[tuple[str, ...]]:
	"""Build exclusion set from observed trajectories (and optionally M2 top-N).

	Args:
		rankings: M2 trajectory rankings DataFrame.
		exclude_m2_top_n: Number of top M2 trajectories to exclude (default 0 = none).

	Returns:
		Set of trajectory tuples to exclude.
	"""
	cols = [
		"structure_1",
		"content_1",
		"structure_2",
		"content_2",
		"structure_3",
		"content_3",
	]

	combined = set()

	# Only exclude M2 top-N if explicitly requested
	if exclude_m2_top_n > 0:
		m2_top = rankings.head(exclude_m2_top_n)
		exclusion_m2 = set(m2_top[cols].itertuples(index=False, name=None))
		combined |= exclusion_m2
		logger.info(
			"Excluding %d M2 top-%d trajectories", len(exclusion_m2), exclude_m2_top_n
		)

	# Always exclude observed trajectories (n_observed > 0)
	observed = rankings[rankings["n_observed"].notna() & (rankings["n_observed"] > 0)]
	exclusion_observed = set(observed[cols].itertuples(index=False, name=None))
	combined |= exclusion_observed

	logger.info(
		"Exclusion set: %d observed, %d total",
		len(exclusion_observed),
		len(combined),
	)

	return combined


def generate_all_combos() -> pd.DataFrame:
	"""Generate all ~1M trajectory combinations.

	Returns:
		DataFrame with structure_1, content_1, ..., structure_3, content_3 columns.
	"""
	combos = list(
		itertools.product(
			STRUCTURES,
			SUBTOPICS,
			STRUCTURES,
			SUBTOPICS,
			STRUCTURES,
			SUBTOPICS,
		)
	)
	return pd.DataFrame(
		combos,
		columns=[
			"structure_1",
			"content_1",
			"structure_2",
			"content_2",
			"structure_3",
			"content_3",
		],
	)


def filter_combos(
	combo_df: pd.DataFrame,
	exclusion_set: set[tuple[str, ...]],
) -> pd.DataFrame:
	"""Remove excluded trajectories from combo DataFrame.

	Args:
		combo_df: All trajectory combinations.
		exclusion_set: Set of trajectory tuples to exclude.

	Returns:
		Filtered DataFrame with excluded trajectories removed.
	"""
	cols = [
		"structure_1",
		"content_1",
		"structure_2",
		"content_2",
		"structure_3",
		"content_3",
	]
	mask = ~combo_df[cols].apply(tuple, axis=1).isin(exclusion_set)
	filtered = combo_df[mask].reset_index(drop=True)
	logger.info(
		"Filtered combos: %d -> %d (removed %d excluded)",
		len(combo_df),
		len(filtered),
		len(combo_df) - len(filtered),
	)
	return filtered


def select_targeted(synthesis_type: str, top_n: int, topic: str) -> pd.DataFrame:
	"""Select top-N trajectories from M2 rankings.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		top_n: Number of trajectories to select.
		topic: Topic subdirectory name (e.g. single_use_plastic).

	Returns:
		DataFrame with selected trajectories and predicted_score.
	"""
	logger.info("=" * 60)
	logger.info("Targeted (M2) selection for %s", synthesis_type)
	logger.info("=" * 60)

	rankings = load_rankings(synthesis_type, topic)
	selected = rankings.head(top_n).copy()

	logger.info("Selected top %d M2-predicted trajectories", len(selected))
	return selected


def select_random(
	synthesis_type: str,
	top_n: int,
	seed: int,
	topic: str,
) -> pd.DataFrame:
	"""Select random trajectories for ablation.

	Samples from all unobserved trajectories (does NOT exclude M2 top-50).
	This provides a true random baseline from the unobserved space.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		top_n: Number of trajectories to select.
		seed: Random seed.
		topic: Topic subdirectory name (e.g. single_use_plastic).

	Returns:
		DataFrame with selected trajectories.
	"""
	logger.info("=" * 60)
	logger.info("Random selection for %s", synthesis_type)
	logger.info("=" * 60)

	rankings = load_rankings(synthesis_type, topic)
	# Pass 0 for exclude_m2_top_n to only exclude observed trajectories
	exclusion_set = build_exclusion_set(rankings, exclude_m2_top_n=0)

	combo_df = generate_all_combos()
	filtered = filter_combos(combo_df, exclusion_set)

	# Random sample
	random.seed(seed)
	indices = random.sample(range(len(filtered)), min(top_n, len(filtered)))
	selected = filtered.iloc[indices].reset_index(drop=True)

	# Assign a dummy predicted_score (not meaningful for random selection)
	selected["predicted_score"] = 0.0

	logger.info("Selected %d random trajectories", len(selected))
	return selected


def select_m1b(
	synthesis_type: str,
	top_n: int,
	seed: int,
	topic: str,
) -> pd.DataFrame:
	"""Select top-3-topics trajectories for ablation.

	Uses M1b (topic-presence) model to identify the top 3 content topics,
	then filters trajectories to only those using those topics in ALL 3 steps.
	Random samples from the filtered set.

	This tests whether knowing the top content topics (from M1b) is sufficient,
	without needing M2's sequential/structural features.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		top_n: Number of trajectories to select.
		seed: Random seed for sampling.
		topic: Topic subdirectory name (e.g. single_use_plastic).

	Returns:
		DataFrame with selected trajectories.
	"""
	logger.info("=" * 60)
	logger.info("Top-3-Topics (M1b) selection for %s", synthesis_type)
	logger.info("=" * 60)

	# Step 1: Load original argument data and fit M1b to find top topics
	bt_path = Path(
		f"experiments/argument_generation/argument_data/{topic}/synthesis_{synthesis_type}"
		f"/pairwise_comparisons_bt_scores.csv"
	)
	arg_df = pd.read_csv(bt_path)

	# Train/test split for comparability with m1_vs_m2_analysis.py
	# M1b should be fit on the same 60% train split used for M2
	train_df, _ = train_test_split(
		arg_df, test_size=TEST_SIZE, random_state=RANDOM_STATE
	)
	logger.info(
		"Using train split for M1b fitting: %d/%d samples (%.0f%%)",
		len(train_df),
		len(arg_df),
		100 * len(train_df) / len(arg_df),
	)

	# Prepare data (compute rank_score) on train split only
	n_arguments = len(train_df)
	train_df = train_df.copy()  # Avoid SettingWithCopyWarning
	train_df["rank_score"] = train_df["bt_score"].rank() / n_arguments
	response_col = "rank_score"
	train_df[response_col] = (
		train_df[response_col] - train_df[response_col].mean()
	) / train_df[response_col].std()

	# Parse structure and content for steps 1-3
	def parse_action(action_str: str) -> tuple[str, str]:
		"""Parse combined action string into (structure, subtopic) tuple."""
		if pd.isna(action_str) or not action_str:
			return "", ""
		for struct in STRUCTURES:
			if action_str == struct:
				return struct, ""
			if action_str.startswith(struct + "_"):
				return struct, action_str[len(struct) + 1 :]
		return action_str, ""

	for step in [1, 2, 3]:
		col = f"step_{step}_structure"
		if col in train_df.columns:
			parsed = train_df[col].apply(lambda x: pd.Series(parse_action(x)))
			train_df[f"structure_{step}"] = parsed[0]
			train_df[f"content_{step}"] = parsed[1]

	# Create M1b features (topic presence only)
	# Binary: did this subtopic appear anywhere in trajectory?
	# Note: Include all topics (no reference category exclusion) since each trajectory
	# has exactly 3 topic slots, so features aren't perfectly collinear with intercept.
	feature_cols = []
	for subtopic in SUBTOPICS:
		col_name = f"has_{subtopic}"
		train_df[col_name] = (
			(train_df["content_1"] == subtopic)
			| (train_df["content_2"] == subtopic)
			| (train_df["content_3"] == subtopic)
		).astype(int)
		feature_cols.append(col_name)

	features_m1b = train_df[feature_cols]
	y = train_df["rank_score"]

	# Fit on train data only (for comparability with M2 in m1_vs_m2_analysis.py)
	model = LinearRegression()
	model.fit(features_m1b, y)

	# Step 2: Identify top 3 topics by coefficient (highest positive coefficients)
	coef_df = pd.DataFrame(
		{
			"feature": features_m1b.columns,
			"coefficient": model.coef_,
		}
	).sort_values("coefficient", ascending=False)

	# Extract topic names (remove "has_" prefix)
	top_3_features = coef_df.head(3)["feature"].tolist()
	top_3_topics = [f.replace("has_", "") for f in top_3_features]

	logger.info("Top 3 topics for %s: %s", synthesis_type, top_3_topics)
	logger.info("All M1b coefficients:")
	for _, row in coef_df.iterrows():
		logger.info("  %s: %.4f", row["feature"], row["coefficient"])
	logger.info("  intercept: %.4f", model.intercept_)

	# Step 3: Generate all combos and filter to top-3-topics only
	combo_df = generate_all_combos()

	# Filter: all 3 content columns must be in top_3_topics
	mask = (
		combo_df["content_1"].isin(top_3_topics)
		& combo_df["content_2"].isin(top_3_topics)
		& combo_df["content_3"].isin(top_3_topics)
	)
	filtered_by_topics = combo_df[mask].reset_index(drop=True)
	logger.info(
		"Filtered to top-3-topics: %d -> %d trajectories",
		len(combo_df),
		len(filtered_by_topics),
	)

	# Step 4: Remove observed trajectories only (no M2 top-50 exclusion)
	rankings = load_rankings(synthesis_type, topic)
	exclusion_set = build_exclusion_set(rankings, exclude_m2_top_n=0)
	filtered = filter_combos(filtered_by_topics, exclusion_set)

	# Step 5: Random sample from filtered set
	random.seed(seed)
	if len(filtered) < top_n:
		logger.warning(
			"Only %d trajectories available after filtering (requested %d)",
			len(filtered),
			top_n,
		)
	indices = random.sample(range(len(filtered)), min(top_n, len(filtered)))
	selected = filtered.iloc[indices].reset_index(drop=True)

	# No meaningful predicted score for random selection from filtered set
	selected["predicted_score"] = 0.0

	logger.info("Selected %d top-3-topics trajectories", len(selected))
	return selected


# =============================================================================
# Argument Generation Functions (from forced_trajectory_experiment.py)
# =============================================================================


def build_forced_controller_choices(
	row: pd.Series,
	depth: int,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
	"""Build forced_controller_choices dict from a trajectory rankings row.

	Maps each reasoning layer to the structure x subtopic action from the row.
	With top_k=1 and n_samples_generation=1, there's exactly 1 node per layer
	at position 0.

	Args:
		row: A row with columns like structure_1, content_1, structure_2, etc.
		depth: Number of reasoning layers (should match column count).

	Returns:
		Dict mapping (layer, node_position) to forced action dicts.
	"""
	forced: dict[tuple[int, int], list[dict[str, Any]]] = {}

	for step in range(depth):
		structure = row[f"structure_{step + 1}"]
		content = row[f"content_{step + 1}"]

		# Tool name format: "{structure}_{subtopic}" (underscore-joined)
		action_name = f"{structure}_{content}"

		forced[(step, 0)] = [
			{
				"action": action_name,
				"action_arguments": {},
				"considerations": f"Forced from {row.get('predicted_score', 'selected')} trajectory",
			}
		]

	return forced


def initialize_models(
	args: argparse.Namespace,
) -> tuple[GenerativeLocalVLLM, ScoringLocalVLLM]:
	"""Initialize two vLLM language models on separate GPUs.

	Args:
		args: Parsed command line arguments.

	Returns:
		Tuple of (generative_lm, reranker_lm).
	"""
	generative_model_name = args.model
	generative_full_model_path = os.path.join(
		args.model_directory, generative_model_name
	)
	reranker_model_name = args.reranker_model
	reranker_full_model_path = os.path.join(args.model_directory, reranker_model_name)

	logger.info(
		f"Initializing generative vLLM model on GPU {args.generative_gpu_index} "
		f"from: {generative_full_model_path}"
	)
	os.environ["CUDA_VISIBLE_DEVICES"] = args.generative_gpu_index
	generative_lm = GenerativeLocalVLLM(
		model=generative_full_model_path,
		tensor_parallel_size=args.generator_tensor_parallel_size,
		dtype=args.generator_dtype,
		gpu_memory_utilization=args.generator_gpu_memory_utilization,
		max_model_len=args.generator_max_model_len,
		enforce_eager=args.generator_enforce_eager,
		verbosity=Verbosity.INFO,
	)
	logger.info(
		f"Generative vLLM model initialized successfully on GPU {args.generative_gpu_index}"
	)

	logger.info(
		f"Initializing reranker vLLM model on GPU {args.reranker_gpu_index} "
		f"from: {reranker_full_model_path}"
	)
	os.environ["CUDA_VISIBLE_DEVICES"] = args.reranker_gpu_index
	reranker_lm = ScoringLocalVLLM(
		model=reranker_full_model_path,
		dtype=args.reranker_dtype,
		gpu_memory_utilization=args.reranker_gpu_memory_utilization,
		max_model_len=args.reranker_max_model_len,
		enforce_eager=args.reranker_enforce_eager,
		verbosity=Verbosity.INFO,
	)
	logger.info(
		f"Reranker vLLM model initialized successfully on GPU {args.reranker_gpu_index}"
	)

	os.environ["CUDA_VISIBLE_DEVICES"] = (
		f"{args.generative_gpu_index},{args.reranker_gpu_index}"
	)
	logger.info("Both GPUs are now visible for runtime operations")

	dspy.settings.configure(lm=generative_lm)
	return generative_lm, reranker_lm


def run_generation(
	trajectories: pd.DataFrame,
	args: argparse.Namespace,
	output_path: Path,
	synthesis_type: str,
	selection_mode: str,
) -> None:
	"""Run argument generation for selected trajectories.

	Args:
		trajectories: DataFrame with trajectory columns.
		args: Parsed command line arguments.
		output_path: Path to save output CSV.
		synthesis_type: One of strict, faithful, restructured.
		selection_mode: One of targeted, random, m1b.
	"""
	experiment_mode = f"synthesis_{synthesis_type}"

	generative_lm, reranker_lm = initialize_models(args)

	try:
		# Define constraints
		thought_length = ResponseLength(granularity="sentence", bounds=(1, 3))
		response_length = ResponseLength(granularity="sentence", bounds=(5, 7))

		# Configure action space
		action_space_paths = [
			str(ACTION_SPACE_DIR / "structures.json"),
			str(ACTION_SPACE_DIR / args.subtopics_file),
		]

		max_reasoning_steps = args.depth

		# Initialize TreeOfThoughts
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoningAndPersona,
			evaluator_signature=None,
			generative_lm=generative_lm,
			reranker_lm=reranker_lm,
			controller_type="reranker",
			thought_length=thought_length,
			response_length=response_length,
			max_reasoning_steps=max_reasoning_steps,
			final_output_kind=experiment_mode,
			early_stopping_enabled=args.early_stopping_enabled,
			action_space_paths=action_space_paths,
			seed=args.seed,
			verbosity=Verbosity.INFO,
		)

		num_tools = len(tot.controller.tools)
		logger.info(f"Action space created {num_tools} tools")

		# Forced trajectory uses deterministic single-path parameters
		tot_parameters = TreeOfThoughtsParameters(
			depth=args.depth,
			n_samples_generation=1,
			top_k=1,
			n_samples_judge=1,
			judge_temperature=args.judge_temperature,
			generator_temperature=args.generator_temperature,
			controller_temperature=args.controller_temperature,
			controller_use_beam_search=False,
			generator_use_beam_search=False,
			num_final_candidates=1,
			use_self_consistency=False,
			n_final_responses_per_trajectory=1,
			node_selection_strategy="greedy",
		)

		# Ensure output directory exists
		output_path.parent.mkdir(parents=True, exist_ok=True)
		run_timestamp = time.strftime("%Y%m%d_%H%M%S")

		logger.info(
			f"\nForced Trajectory Generation Configuration:\n"
			f"Topic: {TOPIC}\n"
			f"Stance: {STANCE}\n"
			f"Selection mode: {selection_mode}\n"
			f"Top N: {args.top_n}\n"
			f"Samples per trajectory: {args.samples_per_trajectory}\n"
			f"Depth: {args.depth}\n"
			f"Experiment mode: {experiment_mode}\n"
			f"Generator temperature: {args.generator_temperature}\n"
			f"Output CSV: {output_path}\n"
		)

		# Build input data
		input_data: dict[str, str] = {
			"topic": TOPIC,
			"stance": STANCE,
			"persona": (
				args.persona
				if args.persona
				else "The average person, who is not necessarily a domain-expert in the topic."
			),
		}

		# CSV column definitions
		base_columns = [
			"trajectory_rank",
			"predicted_score",
			"actual_score_mean",
			"n_observed",
			"experiment_mode",
			"selection_mode",
			"sample_idx",
			"timestamp",
			"runtime_seconds",
			"final_argument",
		]
		step_columns = []
		for step in range(1, args.depth + 1):
			step_columns.extend(
				[
					f"structure_{step}",
					f"content_{step}",
					f"step_{step}_reasoning",
					f"step_{step}_action_actual",
					f"step_{step}_internal_reasoning",
					f"step_{step}_prefix",
				]
			)
		all_columns = base_columns + step_columns

		# Write CSV header
		file_exists = output_path.exists()
		if not file_exists:
			with open(output_path, "w", newline="", encoding="utf-8") as f:
				writer = csv.DictWriter(f, fieldnames=all_columns)
				writer.writeheader()

		# Generate for each trajectory
		total_generations = len(trajectories) * args.samples_per_trajectory
		generation_count = 0

		for traj_idx, row in trajectories.iterrows():
			forced = build_forced_controller_choices(row, args.depth)

			logger.info(
				f"Trajectory {traj_idx + 1}/{len(trajectories)} "
				f"(predicted_score={row.get('predicted_score', 0):.4f}): "
				+ " -> ".join(
					f"{row[f'structure_{s}']}x{row[f'content_{s}']}"
					for s in range(1, args.depth + 1)
				)
			)

			for sample_idx in range(args.samples_per_trajectory):
				generation_count += 1
				logger.info(
					f"  Sample {sample_idx + 1}/{args.samples_per_trajectory} "
					f"(generation {generation_count}/{total_generations})"
				)

				filename = (
					f"{selection_mode}_traj{traj_idx}_sample{sample_idx}.json"
				)
				output_subdir = output_path.parent

				tot_output: TreeOfThoughtsOutput = tot.forward(
					state=input_data,
					tot_parameters=tot_parameters,
					forced_controller_choices=forced,
					do_save_tree=args.do_save_tree,
					outputs_directory=str(output_subdir),
					outputs_filename=filename,
				)

				# Extract results from leaf nodes
				leaf_nodes = [
					node
					for node in tot_output.tree.nodes.values()
					if not node.children_ids
				]

				for leaf_node in leaf_nodes:
					final_argument = leaf_node.state.output.get("argument", "")
					if not final_argument or final_argument.strip() == "":
						logger.warning(
							f"Empty argument for trajectory {traj_idx}, "
							f"sample {sample_idx}"
						)
						continue

					trajectory = leaf_node.state.controller_output_trajectory
					existing_steps = leaf_node.state.reasoning

					csv_row: dict[str, Any] = {
						"trajectory_rank": traj_idx,
						"predicted_score": row.get("predicted_score", 0),
						"actual_score_mean": row.get("actual_score_mean", ""),
						"n_observed": row.get("n_observed", ""),
						"experiment_mode": experiment_mode,
						"selection_mode": selection_mode,
						"sample_idx": sample_idx,
						"timestamp": run_timestamp,
						"runtime_seconds": tot_output.runtime,
						"final_argument": final_argument,
					}

					# Extract per-step data
					for step in range(1, args.depth + 1):
						csv_row[f"structure_{step}"] = row[f"structure_{step}"]
						csv_row[f"content_{step}"] = row[f"content_{step}"]

						# Reasoning text from generator
						if step - 1 < len(existing_steps):
							reasoning_step = existing_steps[step - 1]
							csv_row[f"step_{step}_reasoning"] = reasoning_step.get(
								"claim", ""
							)
						else:
							csv_row[f"step_{step}_reasoning"] = ""

						# Controller output (actual action taken)
						if step - 1 < len(trajectory):
							ctrl = trajectory[step - 1]
							csv_row[f"step_{step}_action_actual"] = ctrl.action
							csv_row[f"step_{step}_internal_reasoning"] = (
								ctrl.internal_reasoning
							)
							csv_row[f"step_{step}_prefix"] = ctrl.prefix
						else:
							csv_row[f"step_{step}_action_actual"] = ""
							csv_row[f"step_{step}_internal_reasoning"] = ""
							csv_row[f"step_{step}_prefix"] = ""

					# Append to CSV
					with open(output_path, "a", newline="", encoding="utf-8") as f:
						writer = csv.DictWriter(f, fieldnames=all_columns)
						writer.writerow(csv_row)

		logger.info(
			f"Generation complete. "
			f"{generation_count} generations saved to {output_path}"
		)

	finally:
		try:
			generative_lm.kill()
			reranker_lm.kill()
		except Exception as e:
			logger.warning(f"Error during cleanup: {e}")


# =============================================================================
# CLI and Main
# =============================================================================


def parse_args() -> argparse.Namespace:
	"""Parse command line arguments.

	Returns:
		Parsed arguments namespace.
	"""
	parser = argparse.ArgumentParser(
		description="Unified Trajectory Selection & Argument Generation",
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)

	# Selection arguments
	parser.add_argument(
		"--synthesis_type",
		type=str,
		choices=["strict", "faithful", "restructured", "all"],
		default="all",
		help="Synthesis type(s) to process (default: all).",
	)
	parser.add_argument(
		"--selection_mode",
		type=str,
		choices=["targeted", "random", "m1b", "all"],
		default="all",
		help="Selection mode(s) to run (default: all).",
	)
	parser.add_argument(
		"--top_n",
		type=int,
		default=50,
		help="Number of trajectories to select (default: 50).",
	)
	parser.add_argument(
		"--samples_per_trajectory",
		type=int,
		default=5,
		help="Number of argument samples to generate per trajectory (default: 5).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed (default: 42).",
	)

	# Generation arguments
	parser.add_argument(
		"--depth",
		type=int,
		default=3,
		help="Tree depth / number of reasoning steps (default: 3).",
	)
	parser.add_argument(
		"--generator_temperature",
		type=float,
		default=0.7,
		help="Generator temperature (default: 0.7).",
	)
	parser.add_argument(
		"--controller_temperature",
		type=float,
		default=None,
		help="Controller temperature (default: same as generator).",
	)
	parser.add_argument(
		"--judge_temperature",
		type=float,
		default=0.3,
		help="Judge temperature (default: 0.3).",
	)
	parser.add_argument(
		"--subtopics_file",
		type=str,
		default="subtopics.json",
		help="Subtopics JSON file name in action_space directory.",
	)
	parser.add_argument(
		"--persona",
		type=str,
		default=None,
		help="Optional persona name.",
	)
	parser.add_argument(
		"--early_stopping_enabled",
		action="store_true",
		default=False,
		help="Enable early stopping.",
	)
	parser.add_argument(
		"--do_save_tree",
		action="store_true",
		default=False,
		help="Save tree structure to disk.",
	)

	# Model arguments
	parser.add_argument(
		"--model",
		type=str,
		default="Qwen3-30B-A3B-Instruct-2507",
		help="Generative model name.",
	)
	parser.add_argument(
		"--reranker_model",
		type=str,
		default="Qwen3-Reranker-8B",
		help="Reranker model name.",
	)
	parser.add_argument(
		"--model_directory",
		type=str,
		default=os.environ.get("MODEL_DIR"),
		help="Directory where models are stored (or set MODEL_DIR env var).",
	)
	parser.add_argument(
		"--generative_gpu_index",
		type=str,
		default="0",
		help="GPU index for generative model.",
	)
	parser.add_argument(
		"--reranker_gpu_index",
		type=str,
		default="1",
		help="GPU index for reranker model.",
	)

	# vLLM generator configuration
	parser.add_argument(
		"--generator_tensor_parallel_size",
		type=int,
		default=1,
		help="Tensor parallel size for generator.",
	)
	parser.add_argument(
		"--generator_dtype",
		type=str,
		default="auto",
		help="Data type for generator.",
	)
	parser.add_argument(
		"--generator_gpu_memory_utilization",
		type=float,
		default=0.9,
		help="GPU memory utilization for generator.",
	)
	parser.add_argument(
		"--generator_max_model_len",
		type=int,
		default=16384,
		help="Max model length for generator.",
	)
	parser.add_argument(
		"--generator_enforce_eager",
		action="store_true",
		default=False,
		help="Enforce eager mode for generator.",
	)

	# vLLM reranker configuration
	parser.add_argument(
		"--reranker_dtype",
		type=str,
		default="auto",
		help="Data type for reranker.",
	)
	parser.add_argument(
		"--reranker_gpu_memory_utilization",
		type=float,
		default=0.9,
		help="GPU memory utilization for reranker.",
	)
	parser.add_argument(
		"--reranker_max_model_len",
		type=int,
		default=16384,
		help="Max model length for reranker.",
	)
	parser.add_argument(
		"--reranker_enforce_eager",
		action="store_true",
		default=False,
		help="Enforce eager mode for reranker.",
	)

	# Selection-only mode
	parser.add_argument(
		"--selection_only",
		action="store_true",
		default=False,
		help="Only select trajectories, don't generate arguments.",
	)
	parser.add_argument(
		"--topic",
		type=str,
		default="single_use_plastic_specific_subtopics",
		help="Topic subdirectory name (default: single_use_plastic_specific_subtopics).",
	)

	return parser.parse_args()


def main() -> None:
	"""Main entry point."""
	global SUBTOPICS

	args = parse_args()

	# Load subtopics dynamically based on topic
	subtopics_file = TOPIC_SUBTOPICS_FILES.get(args.topic)
	if subtopics_file is None:
		raise ValueError(
			f"Unknown topic: {args.topic}. "
			f"Must be one of: {list(TOPIC_SUBTOPICS_FILES.keys())}"
		)
	with open(ACTION_SPACE_DIR / subtopics_file) as f:
		SUBTOPICS = list(json.load(f)["choices"].keys())
	logger.info("Loaded %d subtopics for topic '%s'", len(SUBTOPICS), args.topic)

	# Determine synthesis types to process
	if args.synthesis_type == "all":
		synthesis_types = SYNTHESIS_TYPES
	else:
		synthesis_types = [args.synthesis_type]

	# Determine selection modes to run
	if args.selection_mode == "all":
		selection_modes = SELECTION_MODES
	else:
		selection_modes = [args.selection_mode]

	for synthesis_type in synthesis_types:
		for selection_mode in selection_modes:
			logger.info("\n" + "=" * 70)
			logger.info(
				"Processing: synthesis_type=%s, selection_mode=%s",
				synthesis_type,
				selection_mode,
			)
			logger.info("=" * 70)

			# Step 1: Select trajectories
			if selection_mode == "targeted":
				trajectories = select_targeted(synthesis_type, args.top_n, args.topic)
			elif selection_mode == "random":
				trajectories = select_random(synthesis_type, args.top_n, args.seed, args.topic)
			else:  # m1b
				trajectories = select_m1b(synthesis_type, args.top_n, args.seed, args.topic)

			# Step 2: Generate arguments (unless selection_only)
			if args.selection_only:
				logger.info(
					"Selection only mode: skipping argument generation. "
					"Selected %d trajectories.",
					len(trajectories),
				)
				# Optionally save the selected trajectories
				out_dir = Path(
					f"experiments/argument_generation/argument_data/{args.topic}/synthesis_{synthesis_type}"
				)
				out_dir.mkdir(parents=True, exist_ok=True)
				out_path = out_dir / f"{selection_mode}_selected_trajectories_{synthesis_type}.csv"
				trajectories.to_csv(out_path, index=False)
				logger.info("Saved selected trajectories to %s", out_path)
			else:
				output_path = Path(
					f"experiments/argument_generation/argument_data/{args.topic}/synthesis_{synthesis_type}"
					f"/targeted_generation/{selection_mode}_forced_results_{synthesis_type}.csv"
				)
				run_generation(
					trajectories,
					args,
					output_path,
					synthesis_type,
					selection_mode,
				)

	logger.info("\nAll experiments complete!")


if __name__ == "__main__":
	main()
