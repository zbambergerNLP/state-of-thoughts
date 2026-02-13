"""
NoveltyBench shared utilities used by multiple experiment runners.
"""

# Standard library imports
import argparse
import logging
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# Third-party imports
from datasets import Dataset, load_from_disk

# Local imports
from experiments.noveltybench.flags import ACTION_SPACE_DIR
from experiments.shared_utils import format_tool_descriptions_from_action_space

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOVELTYBENCH_REPO = REPO_ROOT / "experiments" / "noveltybench" / "novelty-bench-repo"

ACTION_SPACE_COMBOS: dict[str, list[Path]] = {
	"controlled": [
		ACTION_SPACE_DIR / "personalities.json",
		ACTION_SPACE_DIR / "target_audiences.json",
	],
	"uncontrolled": [],
}


def preset_display_name(name: str, *, wrap_width: int | None = 30) -> str:
	"""Convert raw preset keys into human-readable labels.

	Used by both plotting and summary tables. When writing markdown tables, prefer
	``wrap_width=None`` to avoid embedded newlines.
	"""

	def _wrap(s: str) -> str:
		return textwrap.fill(s, width=wrap_width) if wrap_width else s

	# Baselines
	if name == "baseline_InstructionFollowing":
		return _wrap("Baseline")
	if name == "baseline_InstructionFollowingCoT":
		return _wrap("Baseline CoT")
	if name == "baseline_InstructionFollowingWithTools":
		return _wrap("Baseline w/ Action Space")
	if name == "baseline_InstructionFollowingWithToolsCoT":
		return _wrap("Baseline CoT w/ Action Space")

	# ToT
	if name.startswith("tot_"):
		has_tools = "and_tools" in name or "with_tools" in name
		tool_suffix = " w/ Action Space" if has_tools else ""
		config = name.replace("tot_", "").split("instruction_following")[0].strip("_")
		# config is now just "controlled" or "uncontrolled" (or potentially legacy values).
		if config == "uncontrolled":
			return _wrap(f"Baseline ToT{tool_suffix}")
		return _wrap(f"STATe of Thoughts{tool_suffix}")

	# Fallback
	return _wrap(name.replace("_", " "))


def load_and_prepare_dataset(
	dataset_path: Path,
	split: Literal["train", "test"],
	subset: Literal["curated", "wildchat"] | None = None,
	max_examples: int | None = None,
) -> Dataset:
	"""Load and prepare the NoveltyBench dataset from local storage.

	Args:
		dataset_path: Path to the dataset directory.
		split: Split to load (train/test).
		subset: Subset to load (curated/wildchat).
		max_examples: Maximum number of examples to load.

	Returns:
		The loaded dataset.
	"""
	dataset_path = Path(dataset_path)
	if not dataset_path.exists():
		raise ValueError(f"Dataset path does not exist: {dataset_path}")

	if subset is None:
		raise ValueError("subset must be provided (curated or wildchat).")

	dataset_dir = dataset_path / subset
	if not dataset_dir.exists():
		dataset_dir = dataset_path / "noveltybench" / subset
	if not dataset_dir.exists():
		raise FileNotFoundError(f"Dataset subset '{subset}' not found in {dataset_path}")

	logger.info(f"Loading from {dataset_dir}")
	dataset = load_from_disk(str(dataset_dir))

	if subset == "curated":
		if split == "train":
			dataset = dataset.select(range(10))
			logger.info("Selected curated train subset: first 10 examples")
		else:
			total_len = len(dataset)
			dataset = dataset.select(range(10, total_len)) if total_len > 10 else dataset

	if max_examples is not None:
		dataset = dataset.select(range(min(max_examples, len(dataset))))

	return dataset


def format_tool_descriptions_for_tot(action_space_paths: list[str] | None = None) -> str:
	"""Format tool descriptions from NoveltyBench action space JSONs for ToT signatures."""
	files_to_titles = {
		"personalities.json": "PERSONALITY TRAITS",
		"target_audiences.json": "TARGET AUDIENCES",
	}

	if action_space_paths is None:
		action_space_paths = [
			str(ACTION_SPACE_DIR / "personalities.json"),
			str(ACTION_SPACE_DIR / "target_audiences.json"),
		]
	else:
		allowed_files = {"personalities.json", "target_audiences.json"}
		action_space_paths = [
			path for path in action_space_paths if Path(path).name in allowed_files
		]

	return format_tool_descriptions_from_action_space(
		ACTION_SPACE_DIR,
		files_to_titles,
		action_space_paths,
	)


def postprocess_noveltybench_runs(
	*,
	experiment_name: str,
	model_path: str,
	experiment_output_dirs: list[Path],
	partition_workers: int,
	skip_evaluation: bool,
	repo_root: Path | None = None,
	timestamp: str | None = None,
) -> None:
	"""
	Run NoveltyBench postprocessing over a set of completed run directories.

	This function:
	1. Runs evaluation (partitioning + scoring) unless skip_evaluation=True
	2. Generates plots/aggregation only if evaluation was run (skip_evaluation=False)

	Args:
		experiment_name: Experiment root name (e.g. "noveltybench_curated_test").
		model_path: Model name or path used for the run.
		experiment_output_dirs: List of run output directories created during the experiment.
		partition_workers: Number of workers for partitioning.
		skip_evaluation: If True, skip partitioning/scoring and plotting.
			If False, runs full evaluation then generates plots.
		repo_root: Repo root path. If None, uses module-level REPO_ROOT.
		timestamp: Timestamp used to name the run dirs file. If None, uses current UTC time.
	"""
	if not experiment_output_dirs:
		logger.warning("No run directories provided; skipping postprocess.")
		return

	if repo_root is None:
		repo_root = REPO_ROOT

	if timestamp is None:
		timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

	model_name = Path(model_path).name.lower().replace("-", "_")
	results_dir = repo_root / "experiments" / "results" / experiment_name / model_name
	results_dir.mkdir(parents=True, exist_ok=True)

	cmd = [
		sys.executable,
		"-m",
		"experiments.noveltybench.evaluate_noveltybench_experiment",
		"--experiment_dir",
		str(results_dir),
		"--partition_workers",
		str(partition_workers),
	]
	if skip_evaluation:
		cmd.append("--skip_evaluation")
	subprocess.run(  # noqa: S603,S607
		cmd,
		check=True,
	)
	# Plotting pass: only generate plots if evaluation was run
	if not skip_evaluation:
		subprocess.run(  # noqa: S603,S607
			[
				sys.executable,
				"-m",
				"experiments.noveltybench.postprocess_noveltybench_experiment",
				"--experiment_dir",
				str(results_dir),
			],
			check=True,
		)
