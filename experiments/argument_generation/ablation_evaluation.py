"""
Ablation Evaluation: Compare targeted (M2) arguments against all baselines.

Compares targeted (M2-optimized) arguments against three baselines:
1. Original Top 5% - top arguments from the original pairwise evaluation
2. Random - arguments from randomly selected trajectories
3. M1b - arguments from topic-presence-only optimized trajectories

Uses length matching, pairwise GPT-5-mini comparisons, and Bradley-Terry scoring.

Usage:
    # Run evaluation for a single synthesis type + baseline:
    python experiments/argument_generation/ablation_evaluation.py \
        --synthesis_type strict --baseline_type random

    # Run original top 5% comparison:
    python experiments/argument_generation/ablation_evaluation.py \
        --synthesis_type strict --baseline_type original

    # Recompute BT scores from existing comparisons:
    python experiments/argument_generation/ablation_evaluation.py \
        --synthesis_type strict --baseline_type m1b --calculate_bt_only

    # Create unified summary from all results:
    python experiments/argument_generation/ablation_evaluation.py --create_summary

Environment:
    OPENAI_API_KEY must be set for API access (not needed for --create_summary
    or --calculate_bt_only).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

try:
	from dotenv import load_dotenv

	load_dotenv()
except ImportError:
	pass

# Support running from repo root and from experiments/argument_generation/
sys.path.insert(0, str(Path(__file__).parent))

from pairwise_evaluation import (  # noqa: E402
	compute_bradley_terry,
	load_completed_pairs,
	run_comparisons,
)

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Synthesis type color scheme (matching m1_vs_m2_analysis.py)
SYNTHESIS_COLORS = {
	"strict": "#3498db",  # Blue
	"faithful": "#2ecc71",  # Green
	"restructured": "#e74c3c",  # Red
}

SYNTHESIS_LABELS = {
	"strict": "Strict",
	"faithful": "Faithful",
	"restructured": "Restructured",
}

# Output directory for figures
SCRIPT_DIR = Path(__file__).parent.resolve()
FIGURES_DIR = SCRIPT_DIR / "figures" / "targeted_generation"

FORCED_COLOR = "#FF9800"  # Orange for targeted arguments (consistent across types)

BASELINE_COLORS = {
	"original": "#2196F3",  # Blue for original (uses synthesis color per type)
	"random": "#9C27B0",  # Purple for random
	"m1b": "#00BCD4",  # Teal for M1b
}

BASELINE_LABELS = {
	"original": "Original Top 5%",
	"random": "Random",
	"m1b": "M1b (Topic Presence)",
}

SYNTHESIS_TYPES = ["strict", "faithful", "restructured"]
BASELINE_TYPES = ["original", "random", "m1b"]


def apply_clean_style(ax: plt.Axes) -> None:
	"""Apply clean plotting style matching m1_vs_m2_analysis.py."""
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.grid(True, alpha=0.3, zorder=0, axis="y")


def generate_random_pairs(
	all_ids: list[int],
	completed: set[tuple[int, int]],
	num_comparisons: int = 0,
	seed: int = 42,
) -> list[tuple[int, int]]:
	"""Randomly sample pairs from all possible pairs in the pool.

	Args:
		all_ids: List of all arg_ids in the combined pool.
		completed: Set of already-completed pairs.
		num_comparisons: Max pairs to generate. 0 = all pairs.
		seed: Random seed.

	Returns:
		List of (arg_a_id, arg_b_id) pairs.
	"""
	all_possible = []
	for i, id_a in enumerate(all_ids):
		for id_b in all_ids[i + 1 :]:
			pair = (min(id_a, id_b), max(id_a, id_b))
			if pair not in completed:
				all_possible.append(pair)

	if num_comparisons > 0 and num_comparisons < len(all_possible):
		random.seed(seed)
		return random.sample(all_possible, num_comparisons)
	return all_possible


def match_pairs_by_length(
	baseline: pd.DataFrame,
	targeted: pd.DataFrame,
	length_tolerance: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Greedy matched-pair selection by argument length.

	For each targeted argument, find the closest-length unused baseline
	within ±length_tolerance chars. Each baseline is used at most once.

	Args:
		baseline: DataFrame with final_argument and source columns.
		targeted: DataFrame with final_argument and source columns.
		length_tolerance: Max allowed length difference in characters.

	Returns:
		Tuple of (matched_baseline, matched_targeted, matches_df) where
		matches_df has columns baseline_idx, targeted_idx, length_diff
		for diagnostics.
	"""
	baseline = baseline.copy()
	targeted = targeted.copy()
	baseline["arg_length"] = baseline["final_argument"].str.len()
	targeted["arg_length"] = targeted["final_argument"].str.len()

	# Sort by length for deterministic matching
	baseline_sorted = baseline.sort_values("arg_length").reset_index(drop=True)
	targeted_sorted = targeted.sort_values("arg_length").reset_index(drop=True)

	used_baseline_indices: set[int] = set()
	matches: list[dict] = []

	# Iterate over targeted, find matching baseline (preserves all targeted)
	for t_idx in range(len(targeted_sorted)):
		t_len = targeted_sorted.loc[t_idx, "arg_length"]
		best_b_idx = None
		best_diff = float("inf")

		for b_idx in range(len(baseline_sorted)):
			if b_idx in used_baseline_indices:
				continue
			b_len = baseline_sorted.loc[b_idx, "arg_length"]
			diff = abs(t_len - b_len)
			if diff <= length_tolerance and diff < best_diff:
				best_diff = diff
				best_b_idx = b_idx

		if best_b_idx is not None:
			used_baseline_indices.add(best_b_idx)
			matches.append(
				{
					"targeted_idx": t_idx,
					"baseline_idx": best_b_idx,
					"length_diff": best_diff,
				}
			)

	matches_df = pd.DataFrame(matches)
	matched_targeted_indices = [m["targeted_idx"] for m in matches]
	matched_baseline_indices = [m["baseline_idx"] for m in matches]

	matched_targeted = targeted_sorted.loc[matched_targeted_indices].reset_index(
		drop=True
	)
	matched_baseline = baseline_sorted.loc[matched_baseline_indices].reset_index(
		drop=True
	)

	logger.info(
		"Matched pairs (±%d chars): %d pairs from %d targeted and %d baseline. "
		"Mean length diff: %.1f chars",
		length_tolerance,
		len(matches),
		len(targeted),
		len(baseline),
		matches_df["length_diff"].mean() if len(matches) > 0 else 0,
	)
	return matched_baseline, matched_targeted, matches_df


def load_top_originals(synthesis_type: str, top_n: int, topic: str) -> pd.DataFrame:
	"""Load top-N original arguments by BT score.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		top_n: Number of top arguments to include.
		topic: Topic subdirectory name (e.g. single_use_plastic).

	Returns:
		DataFrame with columns including final_argument, bt_score, source.
	"""
	base_dir = Path(
		f"experiments/argument_generation/argument_data/{topic}/synthesis_{synthesis_type}"
	)
	bt_path = base_dir / "pairwise_comparisons_bt_scores.csv"
	df = pd.read_csv(bt_path)
	df = df.sort_values("bt_score", ascending=False).head(top_n).reset_index(drop=True)
	df["source"] = "original"
	df["original_bt_score"] = df["bt_score"]
	return df


def load_targeted_arguments(synthesis_type: str, topic: str) -> pd.DataFrame:
	"""Load targeted (M2-optimized) forced trajectory arguments.

	Tries multiple filename patterns for backwards compatibility.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		topic: Topic subdirectory name (e.g. single_use_plastic).

	Returns:
		DataFrame with final_argument and source columns.
	"""
	base_dir = Path(
		f"experiments/argument_generation/argument_data/{topic}/synthesis_{synthesis_type}"
	)
	gen_dir = base_dir / "targeted_generation"

	# Try different filename patterns (in order of preference)
	candidates = [
		# Current naming convention (targeted_generation subdirectory)
		gen_dir / f"targeted_forced_results_{synthesis_type}.csv",
		# Legacy naming conventions (flat directory)
		base_dir / f"targeted_forced_results_{synthesis_type}.csv",
		base_dir / f"forced_trajectory_forced_results_{synthesis_type}.csv",
		base_dir / f"None_forced_results_{synthesis_type}.csv",
	]

	for path in candidates:
		if path.exists():
			logger.info("Loading targeted arguments from %s", path)
			df = pd.read_csv(path)
			df["source"] = "targeted"
			# Drop rows with empty arguments
			df = df.dropna(subset=["final_argument"])
			df = df[df["final_argument"].str.strip() != ""].reset_index(drop=True)
			return df

	raise FileNotFoundError(
		f"No targeted results CSV found for synthesis type '{synthesis_type}' in {gen_dir}. "
		f"Run ablation_trajectory_selection.py with --selection_mode targeted"
	)


def load_baseline_arguments(
	synthesis_type: str,
	baseline_type: str,
	top_n_originals: int = 250,
	topic: str = "single_use_plastic",
) -> pd.DataFrame:
	"""Load baseline arguments for comparison.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		baseline_type: One of original, random, m1b.
		top_n_originals: Number of top originals to include (only for original).
		topic: Topic subdirectory name (e.g. single_use_plastic).

	Returns:
		DataFrame with final_argument and source="baseline" columns.

	Raises:
		FileNotFoundError: If baseline results CSV not found.
	"""
	base_dir = Path(
		f"experiments/argument_generation/argument_data/{topic}/synthesis_{synthesis_type}"
	)

	if baseline_type == "original":
		# Load top-N original arguments by BT score
		df = load_top_originals(synthesis_type, top_n_originals, topic)
		df["source"] = "baseline"
		return df

	# For random and m1b, load from forced results
	gen_dir = base_dir / "targeted_generation"
	path = gen_dir / f"{baseline_type}_forced_results_{synthesis_type}.csv"

	# Fallback to old naming conventions for backwards compatibility
	if not path.exists():
		for fallback in [
			base_dir / f"{baseline_type}_forced_results_{synthesis_type}.csv",
			base_dir / f"ablation_{baseline_type}_forced_results_{synthesis_type}.csv",
		]:
			if fallback.exists():
				path = fallback
				logger.info("Using legacy file path: %s", fallback)
				break

	if not path.exists():
		raise FileNotFoundError(
			f"No baseline results CSV found at {path}. "
			f"Run ablation_trajectory_selection.py with "
			f"--synthesis_type {synthesis_type} --selection_mode {baseline_type}"
		)

	logger.info("Loading baseline arguments from %s", path)
	df = pd.read_csv(path)
	df["source"] = "baseline"
	# Drop rows with empty arguments
	df = df.dropna(subset=["final_argument"])
	df = df[df["final_argument"].str.strip() != ""].reset_index(drop=True)
	return df


def print_summary(df: pd.DataFrame, baseline_type: str) -> None:
	"""Print summary statistics comparing baseline vs targeted arguments.

	Args:
		df: DataFrame with bt_score and source columns.
		baseline_type: One of original, random, m1b (for display).
	"""
	baseline = df[df["source"] == "baseline"]
	targeted = df[df["source"] == "targeted"]
	base_label = BASELINE_LABELS.get(baseline_type, baseline_type)

	logger.info("\n" + "=" * 70)
	logger.info("EVALUATION SUMMARY (Targeted vs %s)", base_label)
	logger.info("=" * 70)

	logger.info(
		"\nPool size: %d baseline + %d targeted = %d total",
		len(baseline),
		len(targeted),
		len(df),
	)

	logger.info("\nMean BT score — baseline:  %.4f", baseline["bt_score"].mean())
	logger.info("Mean BT score — targeted:  %.4f", targeted["bt_score"].mean())
	logger.info("Median BT score — baseline:  %.4f", baseline["bt_score"].median())
	logger.info("Median BT score — targeted:  %.4f", targeted["bt_score"].median())

	# Rank analysis
	df_ranked = df.sort_values("bt_score", ascending=False).reset_index(drop=True)
	df_ranked["rank"] = range(1, len(df_ranked) + 1)

	targeted_ranks = df_ranked[df_ranked["source"] == "targeted"]["rank"]
	baseline_ranks = df_ranked[df_ranked["source"] == "baseline"]["rank"]

	logger.info("\nMean rank — targeted:  %.1f", targeted_ranks.mean())
	logger.info("Mean rank — baseline:  %.1f", baseline_ranks.mean())

	# How many targeted in top-N
	for n in [10, 50, 100]:
		if n <= len(df_ranked):
			top_n = df_ranked.head(n)
			n_targeted = (top_n["source"] == "targeted").sum()
			logger.info(
				"Targeted in top %d: %d/%d (%.1f%%)",
				n,
				n_targeted,
				n,
				n_targeted / n * 100,
			)

	# Top 10 overall
	logger.info("\nTop 10 arguments overall:")
	for _, row in df_ranked.head(10).iterrows():
		arg_preview = row["final_argument"][:80] + "..."
		logger.info(
			"  Rank %3d | %8s | BT=%.4f | %s",
			row["rank"],
			row["source"],
			row["bt_score"],
			arg_preview,
		)


def load_results_for_type(
	synthesis_type: str,
	baseline_type: str,
	topic: str = "single_use_plastic",
) -> pd.DataFrame | None:
	"""Load baseline vs targeted BT scores CSV for a synthesis type.

	Normalizes source column to always use "targeted" and "baseline" values,
	regardless of whether loading from new or legacy file formats.

	If many BT scores are NaN, recomputes them from the corresponding JSONL file.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		baseline_type: One of original, random, m1b.
		topic: Topic subdirectory name (e.g. single_use_plastic).

	Returns:
		DataFrame with bt_score and source columns (normalized to
		"targeted"/"baseline"), or None if not found.
	"""
	base_dir = Path(
		f"experiments/argument_generation/argument_data/{topic}/synthesis_{synthesis_type}"
	)
	gen_dir = base_dir / "targeted_generation"

	if baseline_type == "original":
		path = gen_dir / "targeted_vs_original_bt_scores.csv"
		# Fallback to legacy naming
		if not path.exists():
			path = base_dir / "targeted_vs_original_bt_scores.csv"
		if not path.exists():
			path = base_dir / "forced_vs_original_bt_scores.csv"
	else:
		path = gen_dir / f"targeted_vs_{baseline_type}_bt_scores.csv"
		# Fallback to legacy naming
		if not path.exists():
			path = base_dir / f"targeted_vs_{baseline_type}_bt_scores.csv"
		if not path.exists():
			path = base_dir / f"ablation_{baseline_type}_vs_targeted_bt_scores.csv"

	if not path.exists():
		logger.warning(
			"No results found for %s/%s at %s",
			synthesis_type, baseline_type, path,
		)
		return None

	df = pd.read_csv(path)

	# Normalize source column to consistent naming
	# Legacy files use "forced"/"original" or "ablation", new files use "targeted"/"baseline"
	source_mapping = {
		"forced": "targeted",
		"original": "baseline",
		"ablation": "baseline",
	}
	df["source"] = df["source"].replace(source_mapping)

	# Check if many BT scores are NaN - if so, recompute from JSONL
	n_missing = df["bt_score"].isna().sum() if "bt_score" in df.columns else len(df)
	if n_missing > len(df) * 0.1:  # More than 10% missing
		# Find corresponding JSONL file
		if baseline_type == "original":
			jsonl_path = gen_dir / "targeted_vs_original_comparisons.jsonl"
			if not jsonl_path.exists():
				jsonl_path = base_dir / "targeted_vs_original_comparisons.jsonl"
			if not jsonl_path.exists():
				jsonl_path = base_dir / "forced_vs_original_comparisons.jsonl"
		else:
			jsonl_path = gen_dir / f"targeted_vs_{baseline_type}_comparisons.jsonl"
			if not jsonl_path.exists():
				jsonl_path = base_dir / f"targeted_vs_{baseline_type}_comparisons.jsonl"
			if not jsonl_path.exists():
				jsonl_path = base_dir / f"ablation_{baseline_type}_vs_targeted_comparisons.jsonl"

		if jsonl_path.exists():
			logger.info(
				"%s/%s: %d/%d BT scores missing, recomputing from %s",
				synthesis_type, baseline_type, n_missing, len(df), jsonl_path.name,
			)
			scores = compute_bradley_terry(jsonl_path)
			df["bt_score"] = df["arg_id"].map(scores)
			n_still_missing = df["bt_score"].isna().sum()
			if n_still_missing > 0:
				logger.warning(
					"%s/%s: After recompute, %d/%d items still have no BT score",
					synthesis_type, baseline_type, n_still_missing, len(df),
				)
		else:
			logger.warning(
				"%s/%s: %d/%d BT scores missing but no JSONL found at %s",
				synthesis_type, baseline_type, n_missing, len(df), jsonl_path,
			)

	return df


def load_pre_match_data_for_type(
	synthesis_type: str,
	baseline_type: str,
	top_n_originals: int = 250,
	topic: str = "single_use_plastic",
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
	"""Load pre-match baseline and targeted data for a synthesis type.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		baseline_type: One of original, random, m1b.
		top_n_originals: Number of top originals to use (for original baseline).
		topic: Topic subdirectory name (e.g. single_use_plastic).

	Returns:
		Tuple of (baseline, targeted) DataFrames, or None if not found.
	"""
	try:
		targeted = load_targeted_arguments(synthesis_type, topic)
		baseline = load_baseline_arguments(synthesis_type, baseline_type, top_n_originals, topic)
		return baseline, targeted
	except FileNotFoundError as e:
		logger.warning(
			"Could not load pre-match data for %s/%s: %s",
			synthesis_type,
			baseline_type,
			e,
		)
		return None


def create_unified_histogram_for_synthesis(
	synthesis_type: str,
	topic: str = "single_use_plastic",
) -> None:
	"""Create 3x2 histogram figure for one synthesis type.

	Each row shows one baseline with two panels:
	- Left: All arguments (pre-match)
	- Right: Matched pairs

	Uses synthesis-type-specific colors:
	- Strict: blue tones
	- Faithful: green tones
	- Restructured: red tones

	Args:
		synthesis_type: One of strict, faithful, restructured.
		topic: Topic subdirectory name (e.g. single_use_plastic).
	"""
	base_dir = Path(
		f"experiments/argument_generation/argument_data/{topic}/synthesis_{synthesis_type}"
	)
	synthesis_label = SYNTHESIS_LABELS[synthesis_type]

	fig, axes = plt.subplots(
		len(BASELINE_TYPES),
		2,
		figsize=(14, 12),
		sharey="row",
	)

	x_range = (350, 1300)

	# Use synthesis-specific colors for both targeted and baseline
	# Targeted: darker/more saturated, Baseline: lighter
	synthesis_color = SYNTHESIS_COLORS[synthesis_type]
	targeted_color = synthesis_color
	# Create lighter version for baseline by mixing with white
	rgb = mcolors.to_rgb(synthesis_color)
	baseline_color = tuple(c * 0.5 + 0.5 for c in rgb)  # Lighter version

	# Order: random -> m1b -> original (matches summary table)
	baseline_order = ["random", "m1b", "original"]
	for row, baseline_type in enumerate(baseline_order):
		df = load_results_for_type(synthesis_type, baseline_type, topic)
		base_label = BASELINE_LABELS[baseline_type]

		if df is None:
			for col in range(2):
				axes[row, col].text(
					0.5, 0.5,
					f"No data for {synthesis_type}/{baseline_type}",
					ha="center", va="center",
					transform=axes[row, col].transAxes,
				)
				axes[row, col].set_title(
					f"{base_label} - {'All (pre-match)' if col == 0 else 'Matched'}"
				)
			continue

		# Source columns are normalized by load_results_for_type
		baseline_matched = df[df["source"] == "baseline"]
		targeted_matched = df[df["source"] == "targeted"]

		# Try to load pre-match data for left panel
		pre_match_data = load_pre_match_data_for_type(synthesis_type, baseline_type, topic=topic)

		# Left panel: all arguments (pre-match)
		ax = axes[row, 0]
		if pre_match_data is not None:
			baseline_all, targeted_all = pre_match_data
			for df_plot, label, color, alpha in [
				(targeted_all, "Targeted", targeted_color, 0.8),
				(baseline_all, base_label, baseline_color, 0.6),
			]:
				lengths = df_plot["final_argument"].str.len()
				mean_len = lengths.mean()
				ax.hist(
					lengths,
					bins=50,
					alpha=alpha,
					label=f"{label} (n={len(df_plot)}, mean={mean_len:.0f})",
					color=color,
					edgecolor="white",
					linewidth=0.5,
					range=x_range,
				)
				ax.axvline(mean_len, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
			ax.set_title(f"{base_label} - All (pre-match)", fontweight="bold")
		else:
			# Fall back to matched data
			for df_plot, label, color, alpha in [
				(targeted_matched, "Targeted", targeted_color, 0.8),
				(baseline_matched, base_label, baseline_color, 0.6),
			]:
				lengths = df_plot["final_argument"].str.len()
				mean_len = lengths.mean()
				ax.hist(
					lengths,
					bins=50,
					alpha=alpha,
					label=f"{label} (n={len(df_plot)}, mean={mean_len:.0f})",
					color=color,
					edgecolor="white",
					linewidth=0.5,
					range=x_range,
				)
				ax.axvline(mean_len, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
			ax.set_title(
				f"{base_label} - Matched (pre-match unavailable)",
				fontweight="bold",
			)

		ax.set_xlabel("Argument Length (characters)")
		if row == 1:
			ax.set_ylabel("Count")
		ax.legend(fontsize=9)
		apply_clean_style(ax)

		# Right panel: matched pairs
		ax = axes[row, 1]
		for df_plot, label, color, alpha in [
			(targeted_matched, "Targeted (matched)", targeted_color, 0.8),
			(baseline_matched, f"{base_label} (matched)", baseline_color, 0.6),
		]:
			lengths = df_plot["final_argument"].str.len()
			mean_len = lengths.mean()
			ax.hist(
				lengths,
				bins=50,
				alpha=alpha,
				label=f"{label} (n={len(df_plot)}, mean={mean_len:.0f})",
				color=color,
				edgecolor="white",
				linewidth=0.5,
				range=x_range,
			)
			ax.axvline(mean_len, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
		ax.set_xlabel("Argument Length (characters)")
		ax.set_title(f"{base_label} - Matched Pairs", fontweight="bold")
		ax.legend(fontsize=9)
		apply_clean_style(ax)

	plt.suptitle(
		f"{synthesis_label}: Targeted vs Baselines - Argument Length Distributions",
		fontweight="bold",
		y=1.02,
	)
	plt.tight_layout()
	output_path = FIGURES_DIR / f"length_histograms_unified_{synthesis_type}.pdf"
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info("Saved unified histogram to %s", output_path)


def create_all_unified_histograms(topic: str = "single_use_plastic") -> None:
	"""Create unified histograms for each synthesis type.

	Generates one figure per synthesis type in figures/:
	- length_histograms_unified_strict.pdf
	- length_histograms_unified_faithful.pdf
	- length_histograms_unified_restructured.pdf

	Args:
		topic: Topic subdirectory name (e.g. single_use_plastic).
	"""
	for synthesis_type in SYNTHESIS_TYPES:
		create_unified_histogram_for_synthesis(synthesis_type, topic)


def generate_latex_table(
	df: pd.DataFrame,
	rows_per_section: int = 3,
	label: str = "tab:ablation_trajectory_eval",
	caption: str = "Ablation Study: Targeted vs.\\ All Baselines. $N$ is the total number of length-matched argument pairs (balanced: $N/2$ targeted, $N/2$ baseline).",
) -> str:
	"""Generate a LaTeX table from the summary DataFrame.

	Args:
		df: Summary DataFrame with ablation evaluation results.
		rows_per_section: Number of rows per section (for midrule insertion).
		label: LaTeX label for the table.
		caption: LaTeX caption for the table.

	Returns:
		LaTeX table string.
	"""
	col_mapping = {
		"Baseline": "Baseline",
		"Synthesis Type": "Type",
		"N": "$N$",
		"Pairwise Comparisons": "Comparisons",
		"Win Rate (T)": "Win (T)",
		"Targeted in Top-10": "Top-10",
		"Targeted in Top-100": "Top-100",
	}

	n_cols = len(df.columns)
	col_spec = "l l " + "r " * (n_cols - 2)
	col_spec = col_spec.strip()

	lines = [
		"\\begin{table}[htbp]",
		"\\centering",
		f"\\caption{{{caption}}}",
		f"\\label{{{label}}}",
		f"\\begin{{tabular}}{{{col_spec}}}",
		"\\toprule",
	]

	# Header row
	headers = [col_mapping.get(col, col) for col in df.columns]
	lines.append(" & ".join(headers) + " \\\\")
	lines.append("\\midrule")

	# Data rows (add midrule between synthesis type sections)
	for i, (_, row) in enumerate(df.iterrows()):
		values = [str(v) for v in row.values]
		lines.append(" & ".join(values) + " \\\\")
		# Add midrule after each section (except the last)
		if (i + 1) % rows_per_section == 0 and (i + 1) < len(df):
			lines.append("\\midrule")

	lines.extend([
		"\\bottomrule",
		"\\end{tabular}",
		"\\end{table}",
	])

	return "\n".join(lines)


def _compute_win_rates_from_jsonl(
	comparisons_path: Path,
	targeted_ids: set[int],
) -> tuple[int, int, int, int]:
	"""Compute cross-group win rates from comparisons JSONL.

	Args:
		comparisons_path: Path to the comparisons JSONL file.
		targeted_ids: Set of arg_ids belonging to the targeted group.

	Returns:
		Tuple of (n_total, n_cross_comparisons, targeted_wins, baseline_wins).
	"""
	n_total = 0
	n_cross_comparisons = 0
	targeted_wins = 0
	baseline_wins = 0

	if comparisons_path.exists():
		with open(comparisons_path) as f:
			for line in f:
				rec = json.loads(line)
				n_total += 1
				a_id = rec["arg_a_id"]
				b_id = rec["arg_b_id"]
				winner_id = rec["winner_id"]

				# Only count cross-group comparisons for win rate
				a_is_targeted = a_id in targeted_ids
				b_is_targeted = b_id in targeted_ids
				if a_is_targeted != b_is_targeted:
					n_cross_comparisons += 1
					if winner_id in targeted_ids:
						targeted_wins += 1
					else:
						baseline_wins += 1

	return n_total, n_cross_comparisons, targeted_wins, baseline_wins


def create_summary_table(output_dir: Path, topic: str = "single_use_plastic") -> None:
	"""Create CSV/LaTeX summary from all synthesis types and all baselines.

	Includes all 3 baselines:
	- Original Top 5% (from forced_vs_original_bt_scores.csv)
	- Random (from ablation_random_vs_targeted_bt_scores.csv)
	- M1b (from ablation_m1b_vs_targeted_bt_scores.csv)

	Args:
		output_dir: Directory to save outputs.
		topic: Topic subdirectory name (e.g. single_use_plastic).
	"""
	rows = []
	# All baselines in order: random -> m1b -> original
	baseline_types = ["random", "m1b", "original"]

	baseline_labels = {
		"original": "Original Top 5%",
		"random": "Random",
		"m1b": "M1b (Topic Presence)",
	}

	# Iterate by synthesis type first (outer), then baseline (inner)
	for synthesis_type in SYNTHESIS_TYPES:
		for baseline_type in baseline_types:
			# Load results - load_results_for_type normalizes source columns
			df = load_results_for_type(synthesis_type, baseline_type, topic)
			if df is None:
				continue

			# Source columns are normalized to "targeted" and "baseline"
			targeted = df[df["source"] == "targeted"]
			baseline = df[df["source"] == "baseline"]

			# Find comparisons JSONL path (try targeted_generation first, then legacy)
			base_dir = Path(
				f"experiments/argument_generation/argument_data/{topic}/synthesis_{synthesis_type}"
			)
			gen_dir = base_dir / "targeted_generation"
			if baseline_type == "original":
				comparisons_path = gen_dir / "targeted_vs_original_comparisons.jsonl"
				if not comparisons_path.exists():
					comparisons_path = base_dir / "targeted_vs_original_comparisons.jsonl"
				if not comparisons_path.exists():
					comparisons_path = base_dir / "forced_vs_original_comparisons.jsonl"
			else:
				comparisons_path = gen_dir / f"targeted_vs_{baseline_type}_comparisons.jsonl"
				if not comparisons_path.exists():
					comparisons_path = base_dir / f"targeted_vs_{baseline_type}_comparisons.jsonl"
				if not comparisons_path.exists():
					comparisons_path = base_dir / f"ablation_{baseline_type}_vs_targeted_comparisons.jsonl"

			targeted_ids = set(targeted["arg_id"])

			# Compute win rates from JSONL
			n_total, n_cross, targeted_wins, baseline_wins = _compute_win_rates_from_jsonl(
				comparisons_path, targeted_ids
			)

			targeted_win_rate = (
				targeted_wins / n_cross * 100 if n_cross > 0 else 0
			)
			baseline_win_rate = (
				baseline_wins / n_cross * 100 if n_cross > 0 else 0
			)

			# Rank analysis - only include items with actual BT scores (not NaN)
			df_with_scores = df.dropna(subset=["bt_score"])
			n_scored = len(df_with_scores)
			n_total_items = len(df)
			if n_scored < n_total_items:
				logger.warning(
					"%s/%s: Only %d/%d items have BT scores (rest are NaN)",
					synthesis_type,
					baseline_type,
					n_scored,
					n_total_items,
				)
			df_ranked = df_with_scores.sort_values("bt_score", ascending=False).reset_index(drop=True)
			df_ranked["rank"] = range(1, len(df_ranked) + 1)

			row = {
				"Baseline": baseline_labels[baseline_type],
				"Synthesis Type": SYNTHESIS_LABELS[synthesis_type],
				"N": len(targeted) + len(baseline),
				"Pairwise Comparisons": n_total,
				"Win Rate (T)": f"{targeted_win_rate:.1f}\\%",
			}

			# Targeted in top-N (only among items with BT scores)
			for n in [10, 100]:
				if n <= len(df_ranked):
					top_n = df_ranked.head(n)
					n_targeted_top = (top_n["source"] == "targeted").sum()
					row[f"Targeted in Top-{n}"] = f"{n_targeted_top}/{n}"
				else:
					# Not enough scored items
					row[f"Targeted in Top-{n}"] = f"-/{n}"

			rows.append(row)

	if not rows:
		logger.error("No results found for any synthesis type / baseline type")
		return

	summary_df = pd.DataFrame(rows)

	# Save full table (all synthesis types)
	latex_content = generate_latex_table(summary_df)
	latex_path = Path("paper/latex/tables/ablation_trajectory_summary.tex")
	latex_path.parent.mkdir(parents=True, exist_ok=True)
	with open(latex_path, "w") as f:
		f.write(latex_content)
	logger.info("Saved LaTeX table to %s", latex_path)

	# Save strict-only table (without synthesis type column)
	strict_df = summary_df[summary_df["Synthesis Type"] == "Strict"].drop(
		columns=["Synthesis Type"]
	)
	strict_latex = generate_latex_table(
		strict_df,
		rows_per_section=len(strict_df) + 1,
		label="tab:targeted-trajectory-results",
		caption=(
			r"Targeted trajectory exploration vs.\ baselines ($N{=}204$--$354$ length-matched arguments, "
			r"5{,}000 pairwise comparisons each). \textbf{Win (T)}: share of pairwise wins by targeted "
			r"arguments. \textbf{Top-10/Top-100}: targeted arguments among the top-$n$ by Bradley-Terry score."
		),
	)
	strict_path = Path("paper/tables/argument_generation_targeted_summary.tex")
	strict_path.parent.mkdir(parents=True, exist_ok=True)
	with open(strict_path, "w") as f:
		f.write(strict_latex)
	logger.info("Saved strict-only LaTeX table to %s", strict_path)

	# Save all-synthesis table to paper/tables/
	all_latex = generate_latex_table(
		summary_df,
		label="tab:targeted-trajectory-results-all",
		caption=(
			r"Targeted trajectory exploration vs.\ baselines across all synthesis modes "
			r"(plastic pollution topic). $N$ is the total number of length-matched arguments "
			r"(balanced: $N/2$ targeted, $N/2$ baseline), 5{,}000 pairwise comparisons each. "
			r"\textbf{Win (T)}: share of pairwise wins by targeted arguments. "
			r"\textbf{Top-10/Top-100}: targeted arguments among the top-$n$ by Bradley-Terry score."
		),
	)
	all_path = Path("paper/tables/argument_generation_targeted_summary_all.tex")
	with open(all_path, "w") as f:
		f.write(all_latex)
	logger.info("Saved all-synthesis LaTeX table to %s", all_path)

	# Print as formatted table
	logger.info("\n" + "=" * 80)
	logger.info("ABLATION EVALUATION SUMMARY (ALL BASELINES)")
	logger.info("=" * 80)
	logger.info("\n%s", summary_df.to_string(index=False))


def run_summary_mode(output_dir: Path, topic: str = "single_use_plastic") -> None:
	"""Run summary mode: create unified figures and summary table.

	Args:
		output_dir: Directory to save outputs.
		topic: Topic subdirectory name (e.g. single_use_plastic).
	"""
	logger.info("Running summary mode (no evaluation, reading existing results)")
	output_dir.mkdir(parents=True, exist_ok=True)

	create_all_unified_histograms(topic)
	create_summary_table(output_dir, topic)

	logger.info("Summary outputs saved to %s", output_dir)


def run_evaluation(
	synthesis_type: str,
	baseline_type: str,
	num_comparisons: int,
	model: str,
	max_concurrent: int,
	seed: int,
	length_tolerance: int,
	calculate_bt_only: bool,
	top_n_originals: int = 250,
	topic: str = "single_use_plastic",
) -> None:
	"""Run evaluation for a single synthesis type and baseline type.

	Args:
		synthesis_type: One of strict, faithful, restructured.
		baseline_type: One of original, random, m1b.
		num_comparisons: Number of pairs to evaluate (0 = all).
		model: OpenAI model for judging.
		max_concurrent: Max concurrent API calls.
		seed: Random seed.
		length_tolerance: Max length difference for matched pairs.
		calculate_bt_only: If True, skip comparisons and recompute BT.
		top_n_originals: Number of top originals to include (for original baseline).
		topic: Topic subdirectory name (e.g. single_use_plastic).
	"""
	base_dir = Path(
		f"experiments/argument_generation/argument_data/{topic}/synthesis_{synthesis_type}"
	)
	gen_dir = base_dir / "targeted_generation"
	gen_dir.mkdir(parents=True, exist_ok=True)
	output_jsonl = gen_dir / f"targeted_vs_{baseline_type}_comparisons.jsonl"
	output_csv = gen_dir / f"targeted_vs_{baseline_type}_bt_scores.csv"

	# Load data
	targeted = load_targeted_arguments(synthesis_type, topic)
	baseline = load_baseline_arguments(synthesis_type, baseline_type, top_n_originals, topic)

	base_label = BASELINE_LABELS[baseline_type]
	logger.info(
		"Loaded %d targeted and %d %s arguments for %s",
		len(targeted),
		len(baseline),
		base_label,
		synthesis_type,
	)

	# Match pairs by length
	matched_baseline, matched_targeted, matches_df = match_pairs_by_length(
		baseline, targeted, length_tolerance
	)

	# Assign arg_ids: targeted get 0..N-1, baseline get N..2N-1
	matched_targeted = matched_targeted.copy()
	matched_targeted["arg_id"] = range(len(matched_targeted))

	matched_baseline = matched_baseline.copy()
	matched_baseline["arg_id"] = range(
		len(matched_targeted),
		len(matched_targeted) + len(matched_baseline),
	)

	# Combine
	combined = pd.concat([matched_targeted, matched_baseline], ignore_index=True)

	logger.info(
		"Combined pool: %d targeted (arg_ids 0-%d) + %d baseline (arg_ids %d-%d)",
		len(matched_targeted),
		len(matched_targeted) - 1,
		len(matched_baseline),
		len(matched_targeted),
		len(matched_targeted) + len(matched_baseline) - 1,
	)

	if not calculate_bt_only:
		# Load completed pairs from output file
		completed = load_completed_pairs(output_jsonl)
		logger.info("Found %d already-evaluated pairs in output", len(completed))

		# Generate random pairs from entire pool (unified sampling)
		all_ids = combined["arg_id"].tolist()
		pairs = generate_random_pairs(all_ids, completed, num_comparisons, seed)

		# Log pair type breakdown
		targeted_id_set = set(matched_targeted["arg_id"].tolist())
		baseline_id_set = set(matched_baseline["arg_id"].tolist())
		n_tt, n_bb, n_cross = 0, 0, 0
		for a, b in pairs:
			if a in targeted_id_set and b in targeted_id_set:
				n_tt += 1
			elif a in baseline_id_set and b in baseline_id_set:
				n_bb += 1
			else:
				n_cross += 1
		logger.info(
			"Generated %d new pairs: %d targeted-targeted, %d baseline-baseline, %d cross-group",
			len(pairs),
			n_tt,
			n_bb,
			n_cross,
		)

		if pairs:
			asyncio.run(
				run_comparisons(
					df=combined,
					pairs=pairs,
					output_path=output_jsonl,
					model=model,
					max_concurrent=max_concurrent,
				)
			)

	# Compute Bradley-Terry scores
	if output_jsonl.exists():
		# Compute BT scores directly from JSONL (no filtering needed -
		# the JSONL contains arg_ids that match the current combined pool)
		logger.info("Computing Bradley-Terry scores from %s...", output_jsonl)
		scores = compute_bradley_terry(output_jsonl)
		logger.info("Computed scores for %d unique arg_ids", len(scores))

		combined["bt_score"] = combined["arg_id"].map(scores)

		# Check for unmapped scores (indicates arg_id mismatch)
		n_missing = combined["bt_score"].isna().sum()
		if n_missing > 0:
			logger.warning(
				"%d/%d items have no BT score (arg_id mismatch between CSV and JSONL)",
				n_missing,
				len(combined),
			)

		# Save
		combined.sort_values("bt_score", ascending=False).to_csv(
			output_csv, index=False
		)
		logger.info("Saved BT scores to %s", output_csv)

		# Print summary
		print_summary(combined, baseline_type)
	else:
		logger.error("No comparisons JSONL found at %s", output_jsonl)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Evaluate targeted (M2) arguments against all baselines"
	)
	parser.add_argument(
		"--synthesis_type",
		type=str,
		choices=["strict", "faithful", "restructured"],
		help="Synthesis type to evaluate (required unless --create_summary).",
	)
	parser.add_argument(
		"--baseline_type",
		type=str,
		choices=["original", "random", "m1b"],
		help="Baseline type to compare against (required unless --create_summary).",
	)
	parser.add_argument(
		"--top_n_originals",
		type=int,
		default=250,
		help="Number of top original arguments to include for 'original' baseline (default: 250).",
	)
	parser.add_argument(
		"--num_comparisons",
		type=int,
		default=0,
		help="Number of pairs to evaluate (0 = all pairs, default: 0).",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="gpt-5-mini-2025-08-07",
		help="OpenAI model for judging.",
	)
	parser.add_argument("--max_concurrent", type=int, default=50)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--length_tolerance",
		type=int,
		default=5,
		help="Max length difference for matched pairs in characters (default: 5).",
	)
	parser.add_argument(
		"--calculate_bt_only",
		action="store_true",
		help="Skip comparisons, just compute BT scores from existing JSONL.",
	)
	parser.add_argument(
		"--create_summary",
		action="store_true",
		help="Create unified figures and summary table (no evaluation, reads existing results).",
	)
	parser.add_argument(
		"--topic",
		type=str,
		default="single_use_plastic_specific_subtopics",
		help="Topic subdirectory name (default: single_use_plastic_specific_subtopics).",
	)
	args = parser.parse_args()

	# Handle --create_summary mode
	if args.create_summary:
		output_dir = Path(f"experiments/argument_generation/argument_data/{args.topic}")
		run_summary_mode(output_dir, args.topic)
		return

	# For regular evaluation, synthesis_type and baseline_type are required
	if args.synthesis_type is None:
		parser.error("--synthesis_type is required unless --create_summary is used")
	if args.baseline_type is None:
		parser.error("--baseline_type is required unless --create_summary is used")

	run_evaluation(
		synthesis_type=args.synthesis_type,
		baseline_type=args.baseline_type,
		num_comparisons=args.num_comparisons,
		model=args.model,
		max_concurrent=args.max_concurrent,
		seed=args.seed,
		length_tolerance=args.length_tolerance,
		calculate_bt_only=args.calculate_bt_only,
		top_n_originals=args.top_n_originals,
		topic=args.topic,
	)


if __name__ == "__main__":
	main()
