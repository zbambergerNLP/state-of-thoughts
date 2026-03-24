"""
Controllability evaluation: verify that controller interventions manifest in generated text.

Uses an LLM judge (OpenAI API) to evaluate 13 boolean checks per argument,
split across two separate API calls to prevent cross-contamination:

Call 1 — Step checks (judge only sees reasoning steps, not the final argument):
- Step presence (6): Each reasoning step exhibits its prescribed structure and subtopic

Call 2 — Final checks (judge only sees the final argument, not the reasoning steps):
- Final presence (6): The final argument contains each step's structure and subtopic
- Sequence (1): The final argument preserves step ordering

Usage:
    # Evaluate a single synthesis type
    python experiments/argument_generation/controllability_evaluation.py \
        --synthesis_type strict --topic single_use_plastic

    # Evaluate all synthesis types
    python experiments/argument_generation/controllability_evaluation.py \
        --synthesis_type all --topic single_use_plastic

    # Evaluate forced trajectory data (ablation)
    python experiments/argument_generation/controllability_evaluation.py \
        --synthesis_type strict --data_source targeted --topic single_use_plastic

    # Quick test with 5 rows
    python experiments/argument_generation/controllability_evaluation.py \
        --synthesis_type strict --topic single_use_plastic --max_rows 5

    # Recompute summary from existing results
    python experiments/argument_generation/controllability_evaluation.py \
        --synthesis_type strict --topic single_use_plastic --summarize_only

Resume after crash:
    # Just re-run the same command - it will skip already-evaluated arg_ids
    python experiments/argument_generation/controllability_evaluation.py \
        --synthesis_type strict --topic single_use_plastic

Environment:
    OPENAI_API_KEY must be set for API access.

Output:
    argument_data/{topic}/synthesis_{type}/controllability/results.jsonl
    argument_data/{topic}/synthesis_{type}/controllability/summary.csv

    For forced/ablation data:
    argument_data/{topic}/synthesis_{type}/controllability/{data_source}_results.jsonl
    argument_data/{topic}/synthesis_{type}/controllability/{data_source}_summary.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pure matplotlib styling (matching analyze_m1_vs_m2.py)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["text.color"] = "#333333"

SCRIPT_DIR = Path(__file__).parent.resolve()
ACTION_SPACE_DIR = SCRIPT_DIR / "action_space"
ARGUMENT_DATA_DIR = SCRIPT_DIR / "argument_data"

# Structure and subtopic vocabularies (replicated from analyze_m1_vs_m2.py)
STRUCTURES = [
	"causal_reasoning",
	"conditional",
	"concession_and_contrast",
	"addition_and_elaboration",
	"evidence_and_authority",
	"exemplification",
	"clarification_and_specification",
	"emphasis_and_evaluation",
	"sequence_and_transition",
	"conclusion_and_summary",
]

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of argumentative text. You will be given text \
along with descriptions of prescribed properties. For each check, determine \
whether the text exhibits the described property.

Each property may include:
- A "prefix": the prescribed opening word(s) that the text was instructed to \
start with (e.g., "However", "Therefore").
- A "guidance": the internal instruction given to the model describing what \
angle or lens to reason through. The text should reflect this guidance in its \
content and framing.

Respond with a JSON object containing exactly the keys specified, each with a \
boolean value (true or false). Do not include any other text."""

STEP_CHECK_KEYS = [
	"step_1_has_structure",
	"step_1_has_subtopic",
	"step_2_has_structure",
	"step_2_has_subtopic",
	"step_3_has_structure",
	"step_3_has_subtopic",
]

FINAL_CHECK_KEYS = [
	"final_has_structure_1",
	"final_has_subtopic_1",
	"final_has_structure_2",
	"final_has_subtopic_2",
	"final_has_structure_3",
	"final_has_subtopic_3",
	"final_preserves_order",
]

CHECK_KEYS = STEP_CHECK_KEYS + FINAL_CHECK_KEYS


# ---------------------------------------------------------------------------
# Action space loading
# ---------------------------------------------------------------------------


def load_action_space() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
	"""Load structure and subtopic definitions from JSON files.

	Returns:
		Tuple of (structures_dict, subtopics_dict).
		structures_dict maps name -> {"definition": ..., "prefix": ...}.
		subtopics_dict maps name -> {"definition": ..., "internal_reasoning": ...}.
	"""
	with open(ACTION_SPACE_DIR / "structures.json") as f:
		structures = {
			k: {"definition": v["definition"], "prefix": v["prefix"]}
			for k, v in json.load(f)["choices"].items()
		}
	with open(ACTION_SPACE_DIR / "subtopics.json") as f:
		subtopics = {
			k: {
				"definition": v["definition"],
				"internal_reasoning": v["internal_reasoning"],
			}
			for k, v in json.load(f)["choices"].items()
		}
	return structures, subtopics


def parse_action(action_str: str) -> tuple[str, str]:
	"""Parse combined action string into (structure, subtopic) tuple."""
	if pd.isna(action_str) or not action_str:
		return "", ""
	for struct in STRUCTURES:
		if action_str == struct:
			return struct, ""
		if action_str.startswith(struct + "_"):
			return struct, action_str[len(struct) + 1 :]
	if action_str == "finish":
		return "finish", ""
	return action_str, ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_arguments(
	synthesis_type: str,
	data_source: str,
	topic: str,
) -> pd.DataFrame:
	"""Load arguments and normalize to common schema.

	Returns DataFrame with columns:
		arg_id, final_argument, topic (if available), stance (if available),
		structure_1, content_1, step_1_reasoning,
		structure_2, content_2, step_2_reasoning,
		structure_3, content_3, step_3_reasoning

	Args:
		synthesis_type: One of strict, faithful, restructured.
		data_source: One of original, targeted, random, m1b.
		topic: Topic subdirectory name (e.g. single_use_plastic).
	"""
	base_dir = ARGUMENT_DATA_DIR / topic / f"synthesis_{synthesis_type}"

	if data_source == "original":
		csv_path = (
			base_dir
			/ f"{topic}_synthesis_{synthesis_type}_all_results.csv"
		)
		df = pd.read_csv(csv_path)
		df["arg_id"] = df.index

		# Parse combined action strings into structure + content
		for step in [1, 2, 3]:
			col = f"step_{step}_structure"
			parsed = df[col].apply(parse_action)
			df[f"structure_{step}"] = parsed.apply(lambda x: x[0])
			df[f"content_{step}"] = parsed.apply(lambda x: x[1])
	else:
		csv_path = base_dir / f"{data_source}_forced_results_{synthesis_type}.csv"
		df = pd.read_csv(csv_path)
		df["arg_id"] = df.index

	# Validate required columns exist
	required = ["arg_id", "final_argument"]
	for step in [1, 2, 3]:
		required.extend([
			f"structure_{step}",
			f"content_{step}",
			f"step_{step}_reasoning",
		])
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns in {csv_path}: {missing}")

	logger.info("Loaded %d arguments from %s", len(df), csv_path)
	return df


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_step_prompt(
	row: pd.Series,
	structures_info: dict[str, dict[str, str]],
	subtopics_info: dict[str, dict[str, str]],
) -> str:
	"""Build prompt for step-level checks (only shows reasoning steps).

	The judge sees only the 3 reasoning steps and evaluates whether each
	step exhibits its prescribed structure and subtopic. The final argument
	is deliberately excluded to prevent spillover.

	Args:
		row: A single argument row with normalized columns.
		structures_info: Structure name -> {definition, prefix} mapping.
		subtopics_info: Subtopic name -> {definition, internal_reasoning} mapping.
	"""
	parts = []

	# Context
	if "topic" in row.index and pd.notna(row.get("topic")):
		parts.append(f"Topic: {row['topic']}")
	if "stance" in row.index and pd.notna(row.get("stance")):
		parts.append(f"Stance: {row['stance']}")
	parts.append("")

	# Reasoning steps only
	for step in [1, 2, 3]:
		text = row[f"step_{step}_reasoning"]
		parts.append(f"=== Step {step} ===")
		parts.append(str(text))
		parts.append("")

	# Checks
	parts.append("=== Evaluation Checks ===")
	parts.append(
		"For each check below, respond true if the text clearly exhibits "
		"the described property, or false otherwise."
	)
	parts.append("")

	for step in [1, 2, 3]:
		struct = row[f"structure_{step}"]
		subtopic = row[f"content_{step}"]
		struct_info = structures_info.get(struct, {"definition": struct, "prefix": ""})
		struct_def = struct_info["definition"]
		struct_prefix = struct_info["prefix"]
		subtopic_info = subtopics_info.get(
			subtopic, {"definition": subtopic, "internal_reasoning": ""}
		)
		subtopic_def = subtopic_info["definition"]
		subtopic_reasoning = subtopic_info["internal_reasoning"].strip()

		parts.append(
			f'step_{step}_has_structure: Does Step {step} exhibit the discourse '
			f'structure "{struct}" — defined as: "{struct_def}" '
			f'(prescribed prefix: "{struct_prefix}")?'
		)
		subtopic_line = (
			f'step_{step}_has_subtopic: Does Step {step} discuss the subtopic '
			f'"{subtopic}" — defined as: "{subtopic_def}"'
		)
		if subtopic_reasoning:
			subtopic_line += (
				f' (guidance: "{subtopic_reasoning}")'
			)
		subtopic_line += "?"
		parts.append(subtopic_line)

	parts.append("")
	parts.append(
		"Respond with a JSON object containing exactly these 6 keys, "
		"each with a boolean value:"
	)
	parts.append(", ".join(STEP_CHECK_KEYS))

	return "\n".join(parts)


def build_final_prompt(
	row: pd.Series,
	structures_info: dict[str, dict[str, str]],
	subtopics_info: dict[str, dict[str, str]],
) -> str:
	"""Build prompt for final-argument checks (only shows the final argument).

	The judge sees only the final argument and evaluates whether it contains
	each step's prescribed structure/subtopic, plus ordering. The reasoning
	steps are deliberately excluded to prevent spillover.

	Args:
		row: A single argument row with normalized columns.
		structures_info: Structure name -> {definition, prefix} mapping.
		subtopics_info: Subtopic name -> {definition, internal_reasoning} mapping.
	"""
	parts = []

	# Context
	if "topic" in row.index and pd.notna(row.get("topic")):
		parts.append(f"Topic: {row['topic']}")
	if "stance" in row.index and pd.notna(row.get("stance")):
		parts.append(f"Stance: {row['stance']}")
	parts.append("")

	# Final argument only
	parts.append("=== Argument ===")
	parts.append(str(row["final_argument"]))
	parts.append("")

	# Describe the 3 prescribed properties for reference
	parts.append("=== Prescribed Properties ===")
	parts.append(
		"This argument was generated using three sequential reasoning steps. "
		"Each step was prescribed a discourse structure and a subtopic. "
		"Evaluate whether the argument reflects these prescriptions."
	)
	parts.append("")

	for step in [1, 2, 3]:
		struct = row[f"structure_{step}"]
		subtopic = row[f"content_{step}"]
		struct_info = structures_info.get(struct, {"definition": struct, "prefix": ""})
		struct_def = struct_info["definition"]
		struct_prefix = struct_info["prefix"]
		subtopic_info = subtopics_info.get(
			subtopic, {"definition": subtopic, "internal_reasoning": ""}
		)
		subtopic_def = subtopic_info["definition"]
		subtopic_reasoning = subtopic_info["internal_reasoning"].strip()

		prescription = (
			f'Step {step} prescription — structure: "{struct}" '
			f'("{struct_def}", prefix: "{struct_prefix}"), '
			f'subtopic: "{subtopic}" ("{subtopic_def}"'
		)
		if subtopic_reasoning:
			prescription += f', guidance: "{subtopic_reasoning}"'
		prescription += ")"
		parts.append(prescription)
	parts.append("")

	# Checks
	parts.append("=== Evaluation Checks ===")
	parts.append(
		"For each check below, respond true if the argument clearly exhibits "
		"the described property, or false otherwise."
	)
	parts.append("")

	for step in [1, 2, 3]:
		struct = row[f"structure_{step}"]
		subtopic = row[f"content_{step}"]
		struct_info = structures_info.get(struct, {"definition": struct, "prefix": ""})
		struct_def = struct_info["definition"]
		struct_prefix = struct_info["prefix"]
		subtopic_info = subtopics_info.get(
			subtopic, {"definition": subtopic, "internal_reasoning": ""}
		)
		subtopic_def = subtopic_info["definition"]
		subtopic_reasoning = subtopic_info["internal_reasoning"].strip()

		parts.append(
			f"final_has_structure_{step}: Does the argument contain "
			f'content reflecting the structure "{struct}" — '
			f'defined as: "{struct_def}" '
			f'(prescribed prefix: "{struct_prefix}")?'
		)
		subtopic_line = (
			f"final_has_subtopic_{step}: Does the argument contain "
			f'content reflecting the subtopic "{subtopic}" — '
			f'defined as: "{subtopic_def}"'
		)
		if subtopic_reasoning:
			subtopic_line += (
				f' (guidance: "{subtopic_reasoning}")'
			)
		subtopic_line += "?"
		parts.append(subtopic_line)

	parts.append("")
	parts.append(
		"final_preserves_order: Does the argument contain all three "
		"steps' prescribed content, presented in the correct order (Step 1 material "
		"appears before Step 2 material, which appears before Step 3 material)?"
	)

	parts.append("")
	parts.append(
		"Respond with a JSON object containing exactly these 7 keys, "
		"each with a boolean value:"
	)
	parts.append(", ".join(FINAL_CHECK_KEYS))

	return "\n".join(parts)


# ---------------------------------------------------------------------------
# Async evaluation
# ---------------------------------------------------------------------------


async def judge_call(
	client: AsyncOpenAI,
	semaphore: asyncio.Semaphore,
	arg_id: int,
	prompt: str,
	expected_keys: list[str],
	model: str,
	max_retries: int = 3,
) -> dict | None:
	"""Make a single LLM judge call and validate the response.

	Args:
		expected_keys: The JSON keys the response must contain.

	Returns None if the model fails to provide valid JSON after retries.
	"""
	for attempt in range(max_retries):
		try:
			async with semaphore:
				response = await client.chat.completions.create(
					model=model,
					messages=[
						{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
						{"role": "user", "content": prompt},
					],
					response_format={"type": "json_object"},
					max_completion_tokens=4000,
				)

			content = response.choices[0].message.content or ""
			result = json.loads(content)

			# Validate expected keys present
			missing_keys = [k for k in expected_keys if k not in result]
			if missing_keys:
				logger.warning(
					"Attempt %d for arg %d: missing keys %s",
					attempt + 1,
					arg_id,
					missing_keys,
				)
				if attempt < max_retries - 1:
					await asyncio.sleep(2**attempt)
				continue

			# Coerce values to bool
			return {k: bool(result[k]) for k in expected_keys} | {
				"raw_response": content,
			}

		except (json.JSONDecodeError, KeyError) as e:
			logger.warning(
				"Attempt %d for arg %d: %s", attempt + 1, arg_id, e
			)
			if attempt < max_retries - 1:
				await asyncio.sleep(2**attempt)
		except Exception as e:
			logger.warning(
				"Attempt %d for arg %d: unexpected error: %s",
				attempt + 1,
				arg_id,
				e,
			)
			if attempt < max_retries - 1:
				await asyncio.sleep(2**attempt)

	logger.warning(
		"Skipping arg %d: no valid response after %d retries",
		arg_id,
		max_retries,
	)
	return None


async def run_evaluations(
	df: pd.DataFrame,
	structures_info: dict[str, str],
	subtopics_info: dict[str, str],
	output_path: Path,
	synthesis_type: str,
	data_source: str,
	model: str,
	max_concurrent: int,
) -> None:
	"""Run all evaluations asynchronously, appending results to JSONL."""
	# Load already-completed arg_ids for resume support
	completed_ids: set[int] = set()
	if output_path.exists():
		with open(output_path) as f:
			for line in f:
				rec = json.loads(line)
				completed_ids.add(rec["arg_id"])
	logger.info("Found %d already-evaluated arguments", len(completed_ids))

	# Filter to unevaluated rows
	pending = df[~df["arg_id"].isin(completed_ids)]
	if pending.empty:
		logger.info("All arguments already evaluated")
		return

	logger.info("Evaluating %d arguments", len(pending))

	client = AsyncOpenAI()
	semaphore = asyncio.Semaphore(max_concurrent)

	# Ensure output directory exists
	output_path.parent.mkdir(parents=True, exist_ok=True)

	async def process_and_save(row: pd.Series) -> None:
		arg_id = row["arg_id"]

		# Call 1: step-level checks (judge only sees reasoning steps)
		step_prompt = build_step_prompt(row, structures_info, subtopics_info)
		step_result = await judge_call(
			client=client,
			semaphore=semaphore,
			arg_id=arg_id,
			prompt=step_prompt,
			expected_keys=STEP_CHECK_KEYS,
			model=model,
		)
		if step_result is None:
			return

		# Call 2: final-argument checks (judge only sees final argument)
		final_prompt = build_final_prompt(row, structures_info, subtopics_info)
		final_result = await judge_call(
			client=client,
			semaphore=semaphore,
			arg_id=arg_id,
			prompt=final_prompt,
			expected_keys=FINAL_CHECK_KEYS,
			model=model,
		)
		if final_result is None:
			return

		# Build full record with metadata
		record = {
			"arg_id": int(arg_id),
			"synthesis_type": synthesis_type,
			"data_source": data_source,
		}
		for step in [1, 2, 3]:
			record[f"structure_{step}"] = row[f"structure_{step}"]
			record[f"content_{step}"] = row[f"content_{step}"]
		for key in STEP_CHECK_KEYS:
			record[key] = step_result[key]
		for key in FINAL_CHECK_KEYS:
			record[key] = final_result[key]
		record["raw_response_steps"] = step_result["raw_response"]
		record["raw_response_final"] = final_result["raw_response"]

		# Append to JSONL
		with open(output_path, "a") as f:
			f.write(json.dumps(record) + "\n")

	tasks = [process_and_save(row) for _, row in pending.iterrows()]
	await tqdm_asyncio.gather(*tasks, desc="Evaluating arguments")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_summary(jsonl_path: Path) -> pd.DataFrame:
	"""Compute pass rates from JSONL results.

	Returns DataFrame with columns: check_name, pass_rate, n_evaluated, n_passed.
	"""
	records = []
	with open(jsonl_path) as f:
		for line in f:
			records.append(json.loads(line))

	if not records:
		logger.warning("No records found in %s", jsonl_path)
		return pd.DataFrame()

	df = pd.DataFrame(records)
	n = len(df)

	rows = []

	# Individual checks
	for key in CHECK_KEYS:
		n_passed = int(df[key].sum())
		rows.append({
			"check_name": key,
			"pass_rate": round(n_passed / n, 4),
			"n_evaluated": n,
			"n_passed": n_passed,
		})

	# Grouped averages
	step_structure_keys = [f"step_{s}_has_structure" for s in [1, 2, 3]]
	step_subtopic_keys = [f"step_{s}_has_subtopic" for s in [1, 2, 3]]
	final_structure_keys = [f"final_has_structure_{s}" for s in [1, 2, 3]]
	final_subtopic_keys = [f"final_has_subtopic_{s}" for s in [1, 2, 3]]

	groups = {
		"step_structure_avg": step_structure_keys,
		"step_subtopic_avg": step_subtopic_keys,
		"final_structure_avg": final_structure_keys,
		"final_subtopic_avg": final_subtopic_keys,
		"overall_step": step_structure_keys + step_subtopic_keys,
		"overall_final": final_structure_keys + final_subtopic_keys + ["final_preserves_order"],
		"overall_all_13": CHECK_KEYS,
	}

	for group_name, keys in groups.items():
		total_checks = n * len(keys)
		total_passed = int(df[keys].sum().sum())
		rows.append({
			"check_name": group_name,
			"pass_rate": round(total_passed / total_checks, 4),
			"n_evaluated": total_checks,
			"n_passed": total_passed,
		})

	# Per-structure breakdown: for each structure, find all (arg, step) pairs
	# where that structure was prescribed, then compute the pass rate.
	all_structures = sorted(
		{
			df[f"structure_{s}"].iloc[i]
			for s in [1, 2, 3]
			for i in range(n)
			if pd.notna(df[f"structure_{s}"].iloc[i])
			and df[f"structure_{s}"].iloc[i] != ""
		}
	)

	for struct in all_structures:
		# Step-level: step_X_has_structure where structure_X == struct
		step_n_eval = 0
		step_n_pass = 0
		final_n_eval = 0
		final_n_pass = 0
		for s in [1, 2, 3]:
			mask = df[f"structure_{s}"] == struct
			count = int(mask.sum())
			step_n_eval += count
			step_n_pass += int(df.loc[mask, f"step_{s}_has_structure"].sum())
			final_n_eval += count
			final_n_pass += int(df.loc[mask, f"final_has_structure_{s}"].sum())

		if step_n_eval > 0:
			rows.append({
				"check_name": f"step_structure__{struct}",
				"pass_rate": round(step_n_pass / step_n_eval, 4),
				"n_evaluated": step_n_eval,
				"n_passed": step_n_pass,
			})
		if final_n_eval > 0:
			rows.append({
				"check_name": f"final_structure__{struct}",
				"pass_rate": round(final_n_pass / final_n_eval, 4),
				"n_evaluated": final_n_eval,
				"n_passed": final_n_pass,
			})

	# Per-subtopic breakdown: same logic using content_X columns
	all_subtopics = sorted(
		{
			df[f"content_{s}"].iloc[i]
			for s in [1, 2, 3]
			for i in range(n)
			if pd.notna(df[f"content_{s}"].iloc[i])
			and df[f"content_{s}"].iloc[i] != ""
		}
	)

	for subtopic in all_subtopics:
		step_n_eval = 0
		step_n_pass = 0
		final_n_eval = 0
		final_n_pass = 0
		for s in [1, 2, 3]:
			mask = df[f"content_{s}"] == subtopic
			count = int(mask.sum())
			step_n_eval += count
			step_n_pass += int(df.loc[mask, f"step_{s}_has_subtopic"].sum())
			final_n_eval += count
			final_n_pass += int(df.loc[mask, f"final_has_subtopic_{s}"].sum())

		if step_n_eval > 0:
			rows.append({
				"check_name": f"step_subtopic__{subtopic}",
				"pass_rate": round(step_n_pass / step_n_eval, 4),
				"n_evaluated": step_n_eval,
				"n_passed": step_n_pass,
			})
		if final_n_eval > 0:
			rows.append({
				"check_name": f"final_subtopic__{subtopic}",
				"pass_rate": round(final_n_pass / final_n_eval, 4),
				"n_evaluated": final_n_eval,
				"n_passed": final_n_pass,
			})

	return pd.DataFrame(rows)


def print_summary(summary_df: pd.DataFrame, synthesis_type: str, data_source: str) -> None:
	"""Print formatted summary table."""
	if summary_df.empty:
		return

	logger.info(
		"\n--- Controllability Summary: %s (%s) ---\n%s",
		synthesis_type,
		data_source,
		summary_df.to_string(index=False),
	)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def apply_clean_style(ax):
	"""Apply clean plotting style matching analyze_m1_vs_m2.py."""
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.grid(True, alpha=0.3, zorder=0, axis="x")


def plot_summary(
	summary_df: pd.DataFrame,
	output_path: Path,
	synthesis_type: str,
	data_source: str,
) -> None:
	"""Generate a two-panel PDF with paired horizontal bars for pass rates.

	Panel A (left): per-structure pass rates (step vs final).
	Panel B (right): per-subtopic pass rates (step vs final).

	Args:
		summary_df: Summary DataFrame from compute_summary.
		output_path: Path for the output PDF.
		synthesis_type: Synthesis type label for the title.
		data_source: Data source label for the title.
	"""
	if summary_df.empty:
		logger.warning("Empty summary — skipping figure generation")
		return

	step_color = "#3498db"
	final_color = "#e67e22"
	bar_height = 0.35

	# --- Extract per-structure data ---
	step_struct = summary_df[
		summary_df["check_name"].str.startswith("step_structure__")
	].copy()
	final_struct = summary_df[
		summary_df["check_name"].str.startswith("final_structure__")
	].copy()

	step_struct["dimension"] = step_struct["check_name"].str.replace(
		"step_structure__", "", regex=False
	)
	final_struct["dimension"] = final_struct["check_name"].str.replace(
		"final_structure__", "", regex=False
	)

	struct_df = step_struct[["dimension", "pass_rate", "n_evaluated"]].rename(
		columns={"pass_rate": "step", "n_evaluated": "n"}
	).merge(
		final_struct[["dimension", "pass_rate"]].rename(
			columns={"pass_rate": "final"}
		),
		on="dimension",
		how="outer",
	).fillna(0).sort_values("step", ascending=True)

	# --- Extract per-subtopic data ---
	step_sub = summary_df[
		summary_df["check_name"].str.startswith("step_subtopic__")
	].copy()
	final_sub = summary_df[
		summary_df["check_name"].str.startswith("final_subtopic__")
	].copy()

	step_sub["dimension"] = step_sub["check_name"].str.replace(
		"step_subtopic__", "", regex=False
	)
	final_sub["dimension"] = final_sub["check_name"].str.replace(
		"final_subtopic__", "", regex=False
	)

	sub_df = step_sub[["dimension", "pass_rate", "n_evaluated"]].rename(
		columns={"pass_rate": "step", "n_evaluated": "n"}
	).merge(
		final_sub[["dimension", "pass_rate"]].rename(
			columns={"pass_rate": "final"}
		),
		on="dimension",
		how="outer",
	).fillna(0).sort_values("step", ascending=True)

	if struct_df.empty and sub_df.empty:
		logger.warning("No per-dimension data found — skipping figure")
		return

	# --- Extract sequence preservation rate ---
	seq_row = summary_df[summary_df["check_name"] == "final_preserves_order"]
	seq_rate = float(seq_row["pass_rate"].iloc[0]) if not seq_row.empty else None
	seq_n = int(seq_row["n_evaluated"].iloc[0]) if not seq_row.empty else None

	# --- Plot ---
	fig = plt.figure(figsize=(18, 7.5))
	gs = fig.add_gridspec(
		2, 2, height_ratios=[6, 1], hspace=0.45, wspace=0.65,
	)
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1])
	ax3 = fig.add_subplot(gs[1, :])

	for ax, panel_df, title in [
		(ax1, struct_df, "Per-Structure Pass Rates"),
		(ax2, sub_df, "Per-Subtopic Pass Rates"),
	]:
		if panel_df.empty:
			ax.set_visible(False)
			continue

		labels = [
			f"{d.replace('_', ' ').title()} (n={int(n)})"
			for d, n in zip(panel_df["dimension"], panel_df["n"], strict=True)
		]
		y_pos = range(len(labels))

		ax.barh(
			[y - bar_height / 2 for y in y_pos],
			panel_df["step"].values,
			height=bar_height,
			color=step_color,
			label="Step",
			zorder=3,
		)
		ax.barh(
			[y + bar_height / 2 for y in y_pos],
			panel_df["final"].values,
			height=bar_height,
			color=final_color,
			label="Final",
			zorder=3,
		)

		# Annotate bar tips
		for i, (s_val, f_val) in enumerate(
			zip(panel_df["step"].values, panel_df["final"].values, strict=True)
		):
			ax.text(
				s_val + 0.01, i - bar_height / 2, f"{s_val:.0%}",
				va="center", fontsize=8, color="#333333",
			)
			ax.text(
				f_val + 0.01, i + bar_height / 2, f"{f_val:.0%}",
				va="center", fontsize=8, color="#333333",
			)

		ax.set_yticks(list(y_pos))
		ax.set_yticklabels(labels)
		ax.set_xlim(0, 1.15)
		ax.set_xlabel("Pass Rate")
		ax.set_title(title, fontweight="bold")
		ax.legend(loc="lower right", framealpha=0.9)
		apply_clean_style(ax)

	# --- Bottom panel: sequence preservation ---
	if seq_rate is not None:
		ax3.barh(0, seq_rate, height=bar_height, color=final_color, zorder=3)
		ax3.text(
			seq_rate + 0.01, 0, f"{seq_rate:.0%}",
			va="center", fontsize=9, color="#333333",
		)
		ax3.set_yticks([0])
		ax3.set_yticklabels([f"Full Sequence (n={seq_n})"])
		ax3.set_xlim(0, 1.15)
		ax3.set_ylim(-0.5, 0.5)
		ax3.set_xlabel("Pass Rate")
		ax3.set_title("Final Argument — Step Ordering", fontweight="bold")
		apply_clean_style(ax3)
	else:
		ax3.set_visible(False)

	source_label = f" ({data_source})" if data_source != "original" else ""
	fig.suptitle(
		f"Controllability — synthesis_{synthesis_type}{source_label}",
		fontweight="bold",
		y=1.0,
	)
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info("Saved figure to %s", output_path)


def _format_dimension_label(raw: str) -> str:
	"""Convert underscore-separated dimension name to title case with lowercase conjunctions."""
	lowercase_words = {"and", "of", "the", "in", "for", "to", "a", "an"}
	words = raw.replace("_", " ").split()
	result = []
	for i, w in enumerate(words):
		if i == 0 or w.lower() not in lowercase_words:
			result.append(w.capitalize())
		else:
			result.append(w.lower())
	return " ".join(result)


def _extract_panel_data(
	summary_df: pd.DataFrame,
	prefix: str,
) -> pd.DataFrame:
	"""Extract step/final pass rates for a given dimension prefix.

	Args:
		summary_df: Summary DataFrame from compute_summary.
		prefix: Either "structure" or "subtopic".

	Returns:
		DataFrame with columns: dimension, step, final.
	"""
	step_rows = summary_df[
		summary_df["check_name"].str.startswith(f"step_{prefix}__")
	].copy()
	final_rows = summary_df[
		summary_df["check_name"].str.startswith(f"final_{prefix}__")
	].copy()

	step_rows["dimension"] = step_rows["check_name"].str.replace(
		f"step_{prefix}__", "", regex=False
	)
	final_rows["dimension"] = final_rows["check_name"].str.replace(
		f"final_{prefix}__", "", regex=False
	)

	merged = step_rows[["dimension", "pass_rate"]].rename(
		columns={"pass_rate": "step"}
	).merge(
		final_rows[["dimension", "pass_rate"]].rename(
			columns={"pass_rate": "final"}
		),
		on="dimension",
		how="outer",
	).fillna(0)

	return merged


def plot_unified_controllability(
	summaries: dict[str, pd.DataFrame],
	output_path: Path,
) -> None:
	"""Generate a 3-row stacked figure with per-synthesis controllability breakdowns.

	Each row shows one synthesis type with two panels: per-structure pass rates
	(left) and per-subtopic pass rates (right), using paired horizontal bars
	for step vs final pass rates. Row titles use "Controllability - X Synthesis".

	Args:
		summaries: Mapping from synthesis type name to its summary DataFrame.
		output_path: Path for the output PDF.
	"""
	synthesis_order = ["strict", "faithful", "restructured"]
	synthesis_titles = {
		"strict": "Controllability - Strict Synthesis",
		"faithful": "Controllability - Faithful Synthesis",
		"restructured": "Controllability - Restructured Synthesis",
	}
	present = [s for s in synthesis_order if s in summaries]
	if not present:
		logger.warning("No summaries provided, skipping unified figure")
		return

	# Synthesis-specific colors: pale for step, saturated for final
	synthesis_colors = {
		"strict": {"step": "#80c1ff", "final": "#0059b3"},      # blue
		"faithful": {"step": "#80ff80", "final": "#1a751a"},     # green
		"restructured": {"step": "#ff8080", "final": "#990000"}, # red
	}
	bar_height = 0.35

	n_rows = len(present)
	fig, axes = plt.subplots(n_rows, 2, figsize=(18, 5 * n_rows))
	if n_rows == 1:
		axes = axes.reshape(1, -1)

	for row_idx, stype in enumerate(present):
		summary_df = summaries[stype]

		# Extract per-structure data
		struct_df = _extract_panel_data(summary_df, "structure")
		# Add sample counts from step rows
		step_struct = summary_df[
			summary_df["check_name"].str.startswith("step_structure__")
		].copy()
		step_struct["dimension"] = step_struct["check_name"].str.replace(
			"step_structure__", "", regex=False
		)
		if not struct_df.empty and "n_evaluated" in step_struct.columns:
			struct_df = struct_df.merge(
				step_struct[["dimension", "n_evaluated"]].rename(
					columns={"n_evaluated": "n"}
				),
				on="dimension",
				how="left",
			)
		struct_df = struct_df.sort_values("step", ascending=True)

		# Extract per-subtopic data
		sub_df = _extract_panel_data(summary_df, "subtopic")
		step_sub = summary_df[
			summary_df["check_name"].str.startswith("step_subtopic__")
		].copy()
		step_sub["dimension"] = step_sub["check_name"].str.replace(
			"step_subtopic__", "", regex=False
		)
		if not sub_df.empty and "n_evaluated" in step_sub.columns:
			sub_df = sub_df.merge(
				step_sub[["dimension", "n_evaluated"]].rename(
					columns={"n_evaluated": "n"}
				),
				on="dimension",
				how="left",
			)
		sub_df = sub_df.sort_values("step", ascending=True)

		for col_idx, (panel_df, panel_title) in enumerate([
			(struct_df, "Per-Structure Pass Rates"),
			(sub_df, "Per-Subtopic Pass Rates"),
		]):
			ax = axes[row_idx, col_idx]
			if panel_df.empty:
				ax.set_visible(False)
				continue

			has_n = "n" in panel_df.columns
			labels = [
				f"{_format_dimension_label(d)} (n={int(n)})" if has_n else _format_dimension_label(d)
				for d, n in zip(
					panel_df["dimension"],
					panel_df["n"] if has_n else [0] * len(panel_df),
					strict=True,
				)
			]
			y_pos = range(len(labels))

			colors = synthesis_colors[stype]
			ax.barh(
				[y - bar_height / 2 for y in y_pos],
				panel_df["step"].values,
				height=bar_height,
				color=colors["step"],
				label="Reasoning Step (Claim)",
				zorder=3,
			)
			ax.barh(
				[y + bar_height / 2 for y in y_pos],
				panel_df["final"].values,
				height=bar_height,
				color=colors["final"],
				label="Final Argument",
				zorder=3,
			)

			# Annotate bar tips
			for i, (s_val, f_val) in enumerate(
				zip(panel_df["step"].values, panel_df["final"].values, strict=True)
			):
				ax.text(
					s_val + 0.01, i - bar_height / 2, f"{s_val:.0%}",
					va="center", fontsize=8, color="#333333",
				)
				ax.text(
					f_val + 0.01, i + bar_height / 2, f"{f_val:.0%}",
					va="center", fontsize=8, color="#333333",
				)

			ax.set_yticks(list(y_pos))
			ax.set_yticklabels(labels)
			ax.set_xlim(0, 1.15)
			ax.set_xlabel("Pass Rate")
			ax.set_title(panel_title, fontweight="bold")
			apply_clean_style(ax)

	plt.tight_layout(h_pad=4.0)

	# Add row titles and shared legends after tight_layout
	from matplotlib.patches import Patch
	for row_idx, stype in enumerate(present):
		row_y = axes[row_idx, 0].get_position().y1
		fig.text(
			0.5, row_y + 0.01,
			synthesis_titles[stype],
			ha="center", fontweight="bold", fontsize=13,
			transform=fig.transFigure,
		)
		# Shared legend below this row
		colors = synthesis_colors[stype]
		row_bottom = axes[row_idx, 0].get_position().y0
		legend_handles = [
			Patch(facecolor=colors["step"], label="Reasoning Step (Claim)"),
			Patch(facecolor=colors["final"], label="Final Argument"),
		]
		fig.legend(
			handles=legend_handles,
			loc="center",
			bbox_to_anchor=(0.5, row_bottom - 0.035),
			ncol=2,
			fontsize=10,
			frameon=True,
			framealpha=0.9,
		)

	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info("Saved unified controllability figure to %s", output_path)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

# Display labels and row order for the aggregated controllability table
_CONTROLLABILITY_TABLE_ROWS = [
	("step_structure_avg", "Step structure"),
	("step_subtopic_avg", "Step subtopic"),
	("final_structure_avg", "Final structure"),
	("final_subtopic_avg", "Final subtopic"),
	("final_preserves_order", "Order preservation"),
	("overall_all_13", "Overall (all 13)"),
]


def generate_controllability_latex_table(
	summary_csv_path: Path,
	output_path: Path,
) -> str:
	"""Generate a compact LaTeX table from controllability summary CSV.

	Reads the aggregated rows from summary.csv and produces a three-column
	table (Check Category, N, Pass Rate).

	Args:
		summary_csv_path: Path to the summary.csv file.
		output_path: Path where the .tex file will be saved.

	Returns:
		The LaTeX code as a string.
	"""
	df = pd.read_csv(summary_csv_path)
	lookup = df.set_index("check_name")

	lines = []
	lines.append(r"\begin{table}[htbp]")
	lines.append(r"\centering")
	lines.append(r"\small")
	lines.append(
		r"\caption{Controllability evaluation for strict synthesis "
		r"(plastic pollution). Pass rates assessed by LLM judge across "
		r"13 boolean checks per argument.}"
	)
	lines.append(r"\label{tab:controllability_strict}")
	lines.append(r"\begin{tabular}{lrc}")
	lines.append(r"\toprule")
	lines.append(r"Check Category & N & Pass Rate (\%) \\")
	lines.append(r"\midrule")

	for check_name, display_label in _CONTROLLABILITY_TABLE_ROWS:
		row = lookup.loc[check_name]
		n = int(row["n_evaluated"])
		rate = float(row["pass_rate"]) * 100

		if check_name == "overall_all_13":
			lines.append(r"\midrule")
			lines.append(
				rf"\textbf{{{display_label}}} & \textbf{{{n:,}}} & \textbf{{{rate:.1f}}} \\"
			)
		else:
			lines.append(rf"{display_label} & {n:,} & {rate:.1f} \\")

	lines.append(r"\bottomrule")
	lines.append(r"\end{tabular}")
	lines.append(r"\end{table}")

	latex_code = "\n".join(lines)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w") as f:
		f.write(latex_code + "\n")
	logger.info("Saved controllability LaTeX table to %s", output_path)

	return latex_code


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def get_output_paths(
	synthesis_type: str, data_source: str, topic: str
) -> tuple[Path, Path]:
	"""Get JSONL and summary CSV paths for given configuration."""
	base_dir = ARGUMENT_DATA_DIR / topic / f"synthesis_{synthesis_type}" / "controllability"
	if data_source == "original":
		jsonl_path = base_dir / "results.jsonl"
		csv_path = base_dir / "summary.csv"
	else:
		jsonl_path = base_dir / f"{data_source}_results.jsonl"
		csv_path = base_dir / f"{data_source}_summary.csv"
	return jsonl_path, csv_path


def run_single(
	synthesis_type: str,
	data_source: str,
	model: str,
	max_concurrent: int,
	max_rows: int | None,
	summarize_only: bool,
	topic: str,
) -> None:
	"""Run evaluation for a single synthesis type + data source."""
	structures_info, subtopics_info = load_action_space()
	jsonl_path, csv_path = get_output_paths(synthesis_type, data_source, topic)

	if not summarize_only:
		df = load_arguments(synthesis_type, data_source, topic)
		if max_rows is not None:
			df = df.head(max_rows)
			logger.info("Limited to %d rows for testing", max_rows)

		asyncio.run(
			run_evaluations(
				df=df,
				structures_info=structures_info,
				subtopics_info=subtopics_info,
				output_path=jsonl_path,
				synthesis_type=synthesis_type,
				data_source=data_source,
				model=model,
				max_concurrent=max_concurrent,
			)
		)

	# Compute and save summary
	if jsonl_path.exists():
		summary = compute_summary(jsonl_path)
		jsonl_path.parent.mkdir(parents=True, exist_ok=True)
		summary.to_csv(csv_path, index=False)
		logger.info("Saved summary to %s", csv_path)
		print_summary(summary, synthesis_type, data_source)

		figures_dir = SCRIPT_DIR / "figures" / "controllability"
		figures_dir.mkdir(parents=True, exist_ok=True)
		figure_name = (
			f"controllability_{synthesis_type}.pdf"
			if data_source == "original"
			else f"{data_source}_controllability_{synthesis_type}.pdf"
		)
		figure_path = figures_dir / figure_name
		plot_summary(summary, figure_path, synthesis_type, data_source)
	else:
		logger.warning("No results file found at %s", jsonl_path)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Controllability evaluation for argument generation"
	)
	parser.add_argument(
		"--synthesis_type",
		type=str,
		choices=["strict", "faithful", "restructured", "all"],
		required=True,
		help="Synthesis type to evaluate, or 'all' for all three.",
	)
	parser.add_argument(
		"--data_source",
		type=str,
		choices=["original", "targeted", "random", "m1b"],
		default="original",
		help="Data source: original generation or forced trajectory variant.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="gpt-5-mini-2025-08-07",
		help="OpenAI model to use as the judge.",
	)
	parser.add_argument(
		"--max_concurrent",
		type=int,
		default=50,
		help="Maximum concurrent API requests.",
	)
	parser.add_argument(
		"--max_rows",
		type=int,
		default=None,
		help="Limit number of arguments to evaluate (for testing).",
	)
	parser.add_argument(
		"--summarize_only",
		action="store_true",
		help="Skip evaluation, only compute summary from existing JSONL.",
	)
	parser.add_argument(
		"--topic",
		type=str,
		default="single_use_plastic",
		help="Topic subdirectory name (default: single_use_plastic).",
	)
	parser.add_argument(
		"--latex_table",
		action="store_true",
		help="Generate LaTeX table from existing summary CSV and exit.",
	)
	args = parser.parse_args()

	# LaTeX table generation mode
	if args.latex_table:
		_, csv_path = get_output_paths(
			args.synthesis_type if args.synthesis_type != "all" else "strict",
			args.data_source,
			args.topic,
		)
		if not csv_path.exists():
			logger.error("Summary CSV not found: %s", csv_path)
			return
		paper_dir = SCRIPT_DIR / ".." / ".." / "paper" / "tables"
		output_path = paper_dir / "controllability_strict_summary.tex"
		generate_controllability_latex_table(csv_path, output_path)
		return

	if args.synthesis_type == "all":
		synthesis_types = ["strict", "faithful", "restructured"]
	else:
		synthesis_types = [args.synthesis_type]

	for st in synthesis_types:
		logger.info("=== Processing synthesis_type=%s, data_source=%s ===", st, args.data_source)
		run_single(
			synthesis_type=st,
			data_source=args.data_source,
			model=args.model,
			max_concurrent=args.max_concurrent,
			max_rows=args.max_rows,
			summarize_only=args.summarize_only,
			topic=args.topic,
		)

	# Generate unified figure when all synthesis types are available
	if args.synthesis_type == "all" and args.data_source == "original":
		all_summaries: dict[str, pd.DataFrame] = {}
		for st in synthesis_types:
			_, csv_path = get_output_paths(st, args.data_source, args.topic)
			if csv_path.exists():
				all_summaries[st] = pd.read_csv(csv_path)
		if len(all_summaries) == len(synthesis_types):
			unified_dir = SCRIPT_DIR / "figures" / "controllability"
			unified_dir.mkdir(parents=True, exist_ok=True)
			unified_path = unified_dir / "controllability_unified.pdf"
			plot_unified_controllability(all_summaries, unified_path)
		else:
			logger.warning(
				"Skipping unified figure: only %d/%d summaries available",
				len(all_summaries), len(synthesis_types),
			)


if __name__ == "__main__":
	main()
