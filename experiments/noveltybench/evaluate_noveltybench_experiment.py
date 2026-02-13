"""Evaluate NoveltyBench run directories (partition + score + plot) and write summary tables.

Example usage (cluster, evaluate a whole model directory):

```bash
python -m experiments.noveltybench.evaluate_noveltybench_experiment \
	--experiment_dir experiments/results/noveltybench_curated_test/qwen3_vl_30b_a3b_instruct_fp8 \
	--partition_workers 1
```

This module owns the evaluation workflow:
- partition + score
- update each run's results.json with computed metrics
- write summary tables (CSV/Markdown)
- generate plots (merged from postprocess script)
- print an rsync snippet to download results
"""

# Standard library imports
import concurrent.futures
import getpass
import json
import logging
import os
import socket
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

# Local imports
from experiments.noveltybench.flags import noveltybench_parser
from experiments.noveltybench.noveltybench_utils import preset_display_name
from experiments.shared_utils import (
	PRESET_COLOR_MAP,
	build_preset_key_from_metadata,
	get_preset_display_order,
	normalize_preset_key,
	style_boxplot,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOVELTYBENCH_REPO = REPO_ROOT / "experiments" / "noveltybench" / "novelty-bench-repo"


@dataclass(frozen=True)
class EvalConfig:
	"""Configuration for evaluating a batch of NoveltyBench runs.

	Attributes:
		partition_algorithm: The partitioning algorithm to use.
			Must be one of: "classifier" (DeBERTa), "bertscore" (BERTScore), or "unigram" (ROUGE).
		model_directory: The directory where models are stored.
		scorer_type: The type of scorer to use (e.g., "local", "openai").
		scorer_model: The resolved path to the scorer model.
		score_patience: The fraction of top clusters retained during scoring.
		score_cuda_device: The CUDA device to use for scoring (e.g., "0", "1", "auto").
	"""

	partition_algorithm: Literal["classifier", "bertscore", "unigram"]
	model_directory: str
	scorer_type: str
	scorer_model: str
	score_patience: float
	score_cuda_device: str


@contextmanager
def score_cuda_context(device: str | None):
	"""
	Context manager for setting CUDA_VISIBLE_DEVICES for scoring.

	Args:
		device: Device to set (e.g. "0", "1", "0,1"). If None or "auto", does nothing.
	"""
	if not device or device == "auto":
		yield
		return
	old = os.environ.get("CUDA_VISIBLE_DEVICES")
	os.environ["CUDA_VISIBLE_DEVICES"] = device
	torch.cuda.empty_cache()
	try:
		yield
	finally:
		if old:
			os.environ["CUDA_VISIBLE_DEVICES"] = old
		else:
			os.environ.pop("CUDA_VISIBLE_DEVICES", None)
		torch.cuda.empty_cache()


def _failure_stats_from_results_json(data: dict[str, Any]) -> dict[str, float | int]:
	"""Compute basic failure stats from a loaded results.json dict.

	Args:
		data: The loaded results.json dictionary containing the examples and failed indices.

	Returns:
		A dictionary containing the number of examples, the number of failed examples, and the
		failure rate. Keys in the resulting dictionary are:
		- "num_examples": The number of examples.
		- "num_failed": The number of failed examples.
		- "failure_rate": The failure rate.
	"""
	examples = data.get("examples") or []
	failed_indices = data.get("failed_indices") or []
	if failed_indices:
		num_failed = len(failed_indices)
	else:
		num_failed = sum(1 for ex in examples if (ex or {}).get("failure_reason"))
	num_examples = len(examples)
	failure_rate = (num_failed / num_examples) if num_examples else 0.0
	return {"num_examples": num_examples, "num_failed": num_failed, "failure_rate": failure_rate}


def _load_results_json(path: Path) -> dict[str, Any] | None:
	"""Load results.json with basic validation.

	Args:
		path: Path to the results.json file.

	Returns:
		The loaded results.json dictionary or None if the file is not found or cannot be loaded.
	"""
	if not path.exists():
		logger.warning(f"Results.json file not found. Tried '{path}'")
		return None
	try:
		content = path.read_text().strip()
		if not content:
			return None
		return json.loads(content)
	except Exception as e:
		logger.warning(f"Failed to process {path}: {e}")
		return None


def _run_subprocess_script(args: list[str]) -> None:
	"""
	Run a script from the NoveltyBench repository.

	Args:
		args: List of arguments to pass to the script. The first argument is the script name, and
			the subsequent arguments are passed to the script. The script name must be either
			"partition" or "score".
	"""
	if not NOVELTYBENCH_REPO.exists():
		logger.error(f"NoveltyBench repo not found at {NOVELTYBENCH_REPO}")
		return

	script_name = args[0]
	assert script_name in ["partition", "score"], f"Invalid script name: {script_name}"
	cmd = [sys.executable, "-m", f"src.{script_name}"] + args[1:]
	# Log concise version: script name and number of directories if applicable
	if script_name == "score" and "--eval-dirs" in args:
		eval_dirs_idx = args.index("--eval-dirs")
		# Count directories after --eval-dirs until next flag (starts with --)
		num_dirs = 0
		for i in range(eval_dirs_idx + 1, len(args)):
			if args[i].startswith("--"):
				break
			num_dirs += 1
		logger.info(f"Running scoring for {num_dirs} directories")
	else:
		logger.debug(f"Running: {' '.join(cmd)}")
	try:
		subprocess.run(cmd, check=True, cwd=str(NOVELTYBENCH_REPO), capture_output=False, text=True)  # noqa: S603,S607
	except subprocess.CalledProcessError as e:
		logger.error(f"Command failed with exit code {e.returncode}")


def _has_nonempty_generations(eval_dir: Path) -> bool:
	"""Return True if eval_dir has a readable, non-empty generations.json."""
	gen_path = eval_dir / "generations.json"
	if not gen_path.exists():
		return False
	try:
		with open(gen_path) as f:
			data = json.load(f)
		if not (isinstance(data, list) and len(data) > 0):
			return False
		# Basic shape validation: partition expects {"prompt":..., "generations":[...]} records.
		has_any_generations = False
		for i, rec in enumerate(data):
			if not isinstance(rec, dict):
				logger.warning(f"Invalid record type in {gen_path} at index {i}: {type(rec)}")
				return False
			if "prompt" not in rec or "generations" not in rec:
				logger.warning(f"Missing keys in {gen_path} at index {i}: {sorted(rec.keys())}")
				return False
			if isinstance(rec.get("generations"), list) and len(rec.get("generations")) > 0:
				has_any_generations = True
				break
		if not has_any_generations:
			logger.warning(f"No successful generations found in {gen_path}")
			return False
		return True
	except Exception as e:
		logger.warning(f"Skipping unreadable generations.json at {gen_path}: {e}")
		return False


def _resolve_scorer_model_path(
	scorer_model_path: str,
	model_directory: str,
) -> str:
	"""Resolve and validate that the scorer model path exists.

	The scorer_model_path may be either a full path or just a model name. If it's just a name,
	it will be combined with model_directory to create the full path.

	Args:
		scorer_model_path: The path or name of the scorer model (e.g., Skywork-Reward-Gemma-2-27B-v0.2).
		model_directory: The directory where models are stored.

	Returns:
		The resolved scorer model path if it exists.

	Raises:
		ValueError: If the scorer model path does not exist (either as-is or combined with model_directory).
	"""
	scorer_path = Path(scorer_model_path)
	if scorer_path.exists():
		return scorer_model_path

	# Try combining with model_directory
	if model_directory:
		candidate = Path(model_directory) / scorer_model_path
		if candidate.exists():
			return str(candidate)

	raise ValueError(
		f"Scorer model path does not exist: {scorer_model_path}. "
		f"Tried: {scorer_model_path} and {Path(model_directory) / scorer_model_path}. "
		"Please ensure the path is correct."
	)


def _extract_run_metadata(data: dict[str, Any]) -> dict[str, Any]:
	"""Extract metadata values needed for evaluating a given run directory.

	Args:
		data: The loaded results.json dictionary containing the metadata.

	Returns:
		A dictionary containing the metadata values (namely 'partition_algorithm',
		'model_directory', 'scorer_type', 'scorer_model', 'score_patience', and
		'score_cuda_device').
	"""
	meta = data.get("metadata", {}) or {}
	return {
		"partition_algorithm": meta.get("partition_algorithm"),
		"model_directory": meta.get("model_directory"),
		"scorer_type": meta.get("scorer_type"),
		"scorer_model": meta.get("scorer_model"),
		"score_patience": meta.get("score_patience"),
		"score_cuda_device": meta.get("score_cuda_device"),
	}


def _build_eval_config(meta: dict[str, Any]) -> EvalConfig:
	"""
	Build an evaluation config from metadata.

	Args:
		meta: The metadata dictionary containing the partition algorithm, model directory,
			scorer type, scorer model path, score patience, and score CUDA device.

	Returns:
		An EvalConfig dataclass instance containing the evaluation configuration.

	Raises:
		ValueError: If the metadata is missing any of the required keys.
	"""
	required_keys = [
		"partition_algorithm",
		"model_directory",
		"scorer_type",
		"scorer_model",
		"score_patience",
		"score_cuda_device",
	]
	if not all(meta.get(k) is not None for k in required_keys):
		raise ValueError(f"Missing required metadata keys: {required_keys}. Got: {meta.keys()}")
	scorer_model = _resolve_scorer_model_path(meta["scorer_model"], meta["model_directory"])
	return EvalConfig(
		partition_algorithm=meta["partition_algorithm"],
		model_directory=meta["model_directory"],
		scorer_type=meta["scorer_type"],
		scorer_model=scorer_model,
		score_patience=meta["score_patience"],
		score_cuda_device=meta["score_cuda_device"],
	)


def _collect_eval_batches(run_dirs: list[Path]) -> dict[EvalConfig, list[Path]]:
	"""
	Collect run directories grouped by eval config.

	Args:
		run_dirs: List of run directories to collect batches for.
			We assume that each run directory contains a results.json file with metadata (namely
			'partition_algorithm', 'model_directory', 'scorer_type', 'scorer_model',
			'score_patience', and 'score_cuda_device'). If the metadata is missing any of these
			keys, the run directory is skipped.

	Returns:
		A dictionary mapping EvalConfig instances to lists of run directories.
	"""
	batches: dict[EvalConfig, list[Path]] = defaultdict(list)
	for run_dir in run_dirs:
		res_file = run_dir / "results.json"
		data = _load_results_json(res_file)
		if not data:
			logger.warning(f"Skipping {res_file} during eval check: no data")
			continue
		metrics = data.get("metrics") or {}
		if "mean_distinct" in metrics and "mean_utility" in metrics:
			logger.warning(f"Skipping {res_file} during eval check: already evaluated")
			continue
		meta = _extract_run_metadata(data)
		try:
			config = _build_eval_config(meta)
		except ValueError as e:
			logger.warning(f"Skipping {res_file} during eval check: {e}")
			continue
		batches[config].append(run_dir)
	return batches


def _run_partition_for_dir(
	run_directory: Path,
	partition_algorithm: Literal["classifier", "bertscore", "unigram"],
	partition_model_directory: str | None,
) -> None:
	"""
	Run partitioning script for a single run directory.

	Args:
		run_directory: The run directory to partition.
		partition_algorithm: The partitioning algorithm to use. Must be one of: "classifier" (DeBERTa),
			"bertscore" (BERTScore), or "unigram" (ROUGE).
		partition_model_directory: The model directory to use for caching partitions (e.g., "DeBERTa").
	"""
	if (run_directory / "partitions.json").exists():
		return
	if not _has_nonempty_generations(run_directory):
		logger.warning(f"Skipping partition for empty/missing generations: {run_directory}")
		return
	cmd = ["partition", "--alg", partition_algorithm, "--eval-dirs", str(run_directory.resolve())]
	if partition_model_directory:
		cmd += ["--cache-dir", partition_model_directory]
	_run_subprocess_script(cmd)


def _run_partition_stage(
	run_directories: list[Path],
	partition_algorithm: Literal["classifier", "bertscore", "unigram"],
	partition_model_directory: str | None,
	num_partition_workers: int,
) -> list[Path]:
	"""
	Run partitioning and return directories with partitions.

	Args:
		run_directories: List of run directories to partition.
		partition_algorithm: The partitioning algorithm to use. Must be one of: "classifier" (DeBERTa),
			"bertscore" (BERTScore), or "unigram" (ROUGE).
		partition_model_directory: The model directory to use for caching partitions (e.g., "DeBERTa").
		num_partition_workers: The number of workers to use for partitioning.
	"""
	dirs_needing_partition = [d for d in run_directories if not (d / "partitions.json").exists()]
	if dirs_needing_partition:
		logger.info(
			f"Partition stage:\t{len(dirs_needing_partition)}/{len(run_directories)} "
			"directories need partitions"
		)
	if num_partition_workers <= 1:
		for d in tqdm(dirs_needing_partition, desc="Partitioning", unit="run", leave=False):
			_run_partition_for_dir(d, partition_algorithm, partition_model_directory)
	else:
		with concurrent.futures.ThreadPoolExecutor(max_workers=num_partition_workers) as executor:
			futures = [
				executor.submit(
					_run_partition_for_dir,
					run_directory,
					partition_algorithm,
					partition_model_directory,
				)
				for run_directory in dirs_needing_partition
			]
			with tqdm(total=len(futures), desc="Partitioning", unit="run", leave=False) as pbar:
				for f in concurrent.futures.as_completed(futures):
					try:
						f.result()
					except Exception as e:  # pragma: no cover
						logger.warning(f"Partition worker crashed unexpectedly: {e}")
					pbar.update(1)
	return [
		run_directory for run_directory in run_directories
		if (run_directory / "partitions.json").exists()
	]


def _run_score_stage(
	directories_to_score: list[Path],
	scorer_type: Literal["local", "openai"],
	scorer_model: str | None,
	patience: float,
	cuda_device: str | None,
) -> None:
	"""
	Run scoring script for a batch of run directories.

	Args:
		directories_to_score: List of run directories to score.
		scorer_type: The type of scorer to use (e.g., "local", "openai").
		scorer_model: The path to the scorer model.
		patience: The patience for the scoring algorithm.
		cuda_device: The CUDA device to use for scoring (e.g., "0", "1", "auto").
	"""
	if not directories_to_score:
		logger.warning("No partition files produced; skipping scoring for this batch.")
		return
	cmd = ["score", "--patience", str(patience), "--scorer-type", scorer_type, "--eval-dirs"]
	for directory in directories_to_score:
		cmd += [str(directory.resolve())]
	if scorer_model:
		cmd += ["--scorer-model-path", scorer_model]
	with score_cuda_context(cuda_device):
		_run_subprocess_script(cmd)


def _update_metrics_for_dirs(directories_to_update: list[Path]) -> None:
	"""
	Update results.json metrics for all run directories.

	Args:
		directories_to_update: List of run directories to update metrics for.
	"""
	for run_directory in tqdm(directories_to_update, desc="Updating metrics", unit="run"):
		update_metrics_for_dir(run_directory)


def _evaluate_batches(batches: dict[EvalConfig, list[Path]], num_partition_workers: int) -> None:
	"""
	Run partition + score for each batch configuration.

	Args:
		batches: Dictionary mapping EvalConfig instances to lists of run directories.
		num_partition_workers: The number of workers to use for partitioning.
	"""
	for config, dirs in batches.items():
		logger.info(f"Batch evaluating {len(dirs)} directories with config: {config}")
		dirs_to_score = _run_partition_stage(
			run_directories=dirs,
			partition_algorithm=config.partition_algorithm,
			partition_model_directory=config.model_directory,
			num_partition_workers=num_partition_workers,
		)
		_run_score_stage(
			directories_to_score=dirs_to_score,
			scorer_type=config.scorer_type,
			scorer_model=config.scorer_model,
			patience=config.score_patience,
			cuda_device=config.score_cuda_device,
		)
		_update_metrics_for_dirs(dirs)


def update_metrics_for_dir(output_dir: Path) -> None:
	"""
	Read computed metrics files and update results.json.

	Args:
		output_dir: The directory containing the results.json file.
	"""
	metrics: dict[str, float] = {}
	for file_name, key in [("scores.json", "utility"), ("partitions.json", "partition")]:
		path = output_dir / file_name
		if path.exists():
			try:
				with open(path) as f:
					data = json.load(f)
					if data:
						if key == "partition":
							metrics["mean_distinct"] = sum(
								len(set(x.get("partition", []))) for x in data
							) / len(data)
						elif key == "utility":
							metrics["mean_utility"] = sum(
								x.get("utility", 0) for x in data
							) / len(data)
			except Exception as e:
				logger.warning(f"Failed to read/parse {path}: {e}")

	results_path = output_dir / "results.json"
	try:
		with open(results_path) as f:
			data = json.load(f)
	except Exception as e:
		logger.warning(f"Failed to read results.json for failure stats merge: {e}")
		return

	existing_metrics = data.get("metrics") or {}
	if not isinstance(existing_metrics, dict):
		existing_metrics = {}
	data["metrics"] = {**existing_metrics, **_failure_stats_from_results_json(data), **metrics}
	if data["metrics"]:
		logger.info(f"Updating metrics for {output_dir}: {metrics}")
		try:
			with open(results_path, "w") as f:
				json.dump(data, f, indent=2)
			logger.info(f"Saved updated results to {results_path}")
		except Exception as e:
			logger.error(f"Failed to update results.json: {e}")


def evaluate_directories(results_dir: Path, num_partition_workers: int) -> None:
	"""
	Identify runs needing evaluation and run batch scoring.

	Args:
		results_dir: The directory containing the results.json files.
		num_partition_workers: The number of workers to use for partitioning.
	"""
	run_dirs = sorted({p.parent for p in results_dir.rglob("results.json")})
	batches = _collect_eval_batches(run_dirs)
	_evaluate_batches(batches, num_partition_workers)


def evaluate_run_directories(run_dirs: list[Path], num_partition_workers: int) -> None:
	"""
	Evaluate a specific set of run directories (partition + score).

	Args:
		run_dirs: List of run directories to evaluate.
		num_partition_workers: The number of workers to use for partitioning.
	"""
	batches = _collect_eval_batches(run_dirs)
	_evaluate_batches(batches, num_partition_workers)


def _extract_common_run_fields(data: dict[str, Any], run_directory: Path) -> dict[str, Any]:
	"""Extract standard fields for summaries and plotting.

	Args:
		data: The loaded results.json dictionary containing the metadata and metrics.
		run_directory: The run directory to extract fields for.

	Returns:
		A dictionary containing the standard fields for summaries and plotting.
	"""
	metadata = data.get("metadata") or {}
	metrics = data.get("metrics") or {}
	temperature = metadata["generator_temperature"] or metadata["baseline_temperature"]
	return {
		"Model": metadata["model"],
		"Preset": build_preset_key_from_metadata(metadata),
		"Temperature": temperature,
		"Seed": metadata["seed"],
		"Failure Rate": metrics.get("failure_rate", 0),
		"Num Failed": metrics.get("num_failed", 0),
		"Num Examples": metrics.get("num_examples", 0),
		"Path": str(run_directory),
	}


def _collect_records(results_dir: Path) -> list[dict[str, Any]]:
	"""Collect per-run metrics records from results.json files."""
	records: list[dict[str, Any]] = []
	result_files = list(results_dir.rglob("results.json"))
	for result_file in tqdm(result_files, desc="Collecting results", unit="run"):
		data = _load_results_json(result_file)
		if not data:
			continue
		metrics = data.get("metrics") or {}
		mean_distinct = metrics.get("mean_distinct")
		if mean_distinct is None:
			continue
		try:
			record = _extract_common_run_fields(data, result_file.parent)
			record.update(
				{
					"Mean Distinct": mean_distinct,
					"Mean Utility": metrics["mean_utility"],
				}
			)
			records.append(record)
		except (KeyError, ValueError, TypeError) as e:
			logger.warning(
				f"Skipping {result_file.parent} due to metadata parsing error: {e}"
			)
			continue

	return records


def _extract_failed_indices(data: dict) -> set[int]:
	"""
	Extract failed example indices from results.json contents.

	Args:
		data: The loaded results.json dictionary containing the failed indices and examples.

	Returns:
		A set of failed example indices.
	"""
	failed_indices = data.get("failed_indices") or []
	if failed_indices:
		return {int(i) for i in failed_indices}
	examples = data.get("examples") or []
	return {idx for idx, ex in enumerate(examples) if (ex or {}).get("failure_reason")}


def _compute_working_metrics(
	run_directory: Path,
	failed_indices: set[int],
) -> dict[str, float] | None:
	"""
	Compute metrics using only examples without failures.

	Args:
		run_directory: The run directory to compute metrics for.
		failed_indices: The set of failed example indices.

	Returns:
		A dictionary containing the mean distinct and mean utility metrics.
	"""
	metrics: dict[str, float] = {}
	scores_path = run_directory / "scores.json"
	partitions_path = run_directory / "partitions.json"
	if not scores_path.exists() or not partitions_path.exists():
		return None

	try:
		scores = json.loads(scores_path.read_text())
		partitions = json.loads(partitions_path.read_text())
	except Exception as e:
		logger.warning(f"Failed to read working metrics from {run_directory}: {e}")
		return None

	if not isinstance(scores, list) or not isinstance(partitions, list):
		return None

	# Filter out failed examples. Since failed examples have no scores/partitions,
	# entries in these lists are for working examples. We check by entry id if present
	# (which should be an integer from our run scripts), otherwise by index.
	working_scores = []
	for idx, entry in enumerate(scores):
		entry_id = entry.get("id")
		if isinstance(entry_id, int) and entry_id not in failed_indices:
			working_scores.append(entry)
		elif entry_id is None and idx not in failed_indices:
			working_scores.append(entry)

	working_partitions = []
	for idx, entry in enumerate(partitions):
		entry_id = entry.get("id")
		if isinstance(entry_id, int) and entry_id not in failed_indices:
			working_partitions.append(entry)
		elif entry_id is None and idx not in failed_indices:
			working_partitions.append(entry)

	if working_scores:
		metrics["mean_utility"] = sum(
			x.get("utility", 0) for x in working_scores
		) / len(working_scores)
	if working_partitions:
		metrics["mean_distinct"] = sum(
			len(set(x.get("partition", []))) for x in working_partitions
		) / len(working_partitions)

	return metrics or None


def _collect_working_records(results_dir: Path) -> list[dict[str, Any]]:
	"""Collect per-run metrics records excluding failed examples.

	Args:
		results_dir: The directory containing the results.json files.

	Returns:
		A list of dictionaries containing the per-run metrics records excluding failed examples.
	"""
	records: list[dict[str, Any]] = []
	result_files = list(results_dir.rglob("results.json"))
	for result_file in tqdm(result_files, desc="Collecting working results", unit="run"):
		data = _load_results_json(result_file)
		if not data:
			continue
		failed_indices = _extract_failed_indices(data)
		working_metrics = _compute_working_metrics(result_file.parent, failed_indices)
		if not working_metrics:
			continue
		record = _extract_common_run_fields(data, result_file.parent)
		record.update(
			{
				"Mean Distinct": working_metrics.get("mean_distinct"),
				"Mean Utility": working_metrics.get("mean_utility"),
			}
		)
		records.append(record)

	return records




def plot_metrics(df: pd.DataFrame, output_dir: Path, filename_prefix: str = "") -> None:
	"""
	Plot metrics as distributions (box+whiskers + seed points), one figure per temperature.

	Args:
		df: DataFrame with metrics and metadata columns.
		output_dir: Directory to save generated plot PNGs.
		filename_prefix: Prefix for the output filename (e.g., "working_" to denote strictly
			working/non-failed examples for data used for plotting.)
	"""
	sns.set_theme(
		style="whitegrid",
		context="paper",
		font_scale=1.05,
		rc={"grid.alpha": 0.25, "axes.edgecolor": "#B0B0B0", "axes.linewidth": 0.8},
	)
	df = df.copy()
	if "Preset" in df.columns:
		df["Preset"] = df["Preset"].apply(lambda s: normalize_preset_key(str(s)))

	# Dedupe reruns: keep at most one record per (model, preset, temperature, seed).
	if {"Model", "Preset", "Temperature", "Seed", "Path"}.issubset(set(df.columns)):
		df = df.sort_values(["Model", "Preset", "Temperature", "Seed", "Path"]).drop_duplicates(
			subset=["Model", "Preset", "Temperature", "Seed"],
			keep="last",
		)

	df["DisplayPreset"] = df["Preset"].apply(lambda s: preset_display_name(str(s), wrap_width=30))

	# Get ordered list of display names using shared utility
	unique_display_names: list[str] = get_preset_display_order(list(df["DisplayPreset"].unique()))

	# Use shared color mapping, with fallback for any missing presets
	palette = PRESET_COLOR_MAP.copy()
	base_colors = sns.color_palette("colorblind", n_colors=10)
	for i, name in enumerate(unique_display_names):
		if name not in palette:
			palette[name] = base_colors[i % len(base_colors)]

	models = df["Model"].dropna().unique().tolist()
	if not models:
		logger.warning("No models found to plot.")
		return

	metric_specs = [
		("Mean Distinct", "Diversity (Mean Distinct)", "mean_distinct"),
		("Mean Utility", "Utility (Mean Utility)", "mean_utility"),
	]

	for model in tqdm(models, desc="Plotting", unit="model"):
		model_df = df[df["Model"] == model].copy()
		model_short = str(model).split("/")[-1]
		temps = sorted({float(t) for t in model_df["Temperature"].dropna().unique().tolist()})

		for temp in temps:
			temp_df = model_df[model_df["Temperature"] == temp].copy()
			for metric_col, title_prefix, metric_prefix in metric_specs:
				plot_df = temp_df.dropna(subset=[metric_col]).copy()
				if plot_df.empty:
					continue

				fig_w = max(8.5, 0.75 * len(unique_display_names) + 4.0)
				fig, ax = plt.subplots(figsize=(fig_w, 6.0))

				order_here = [x for x in unique_display_names if x in plot_df["DisplayPreset"].unique()]

				sns.boxplot(
					data=plot_df,
					x="DisplayPreset",
					y=metric_col,
					order=order_here,
					hue="DisplayPreset",
					palette=palette,
					dodge=False,
					legend=False,
					whis=(0, 100),
					showfliers=False,
					width=0.55,
					linewidth=1.0,
					ax=ax,
				)

				style_boxplot(ax=ax, order=order_here, palette=palette)

				sns.stripplot(
					data=plot_df,
					x="DisplayPreset",
					y=metric_col,
					order=order_here,
					hue="DisplayPreset",
					palette=palette,
					dodge=False,
					legend=False,
					size=4.0,
					alpha=0.65,
					jitter=0.16,
					edgecolor="white",
					linewidth=0.4,
					ax=ax,
					zorder=3,
				)

				ax.set_title(f"{title_prefix} - {model_short} - Temperature {temp:g}", fontsize=13)
				ax.set_xlabel("")
				ax.set_ylabel(title_prefix)
				plt.setp(ax.get_xticklabels(),rotation=25,ha="right")
				ax.grid(True, axis="y")
				ax.grid(False, axis="x")

				plt.tight_layout()
				temp_str = str(temp).replace(".", "_")
				out_path = output_dir / f"{filename_prefix}{metric_prefix}_{model_short}_temp_{temp_str}.png"
				plt.savefig(out_path, dpi=300, bbox_inches="tight")
				plt.close(fig)

	logger.info(f"Saved plots to {output_dir}")


def _write_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
	"""Write summary CSV and Markdown tables for the experiment.

	Args:
		df: DataFrame containing the metrics and metadata.
		output_dir: Directory to save the summary tables.
	"""
	df = df.copy()
	# Match plotting labels in summary tables (but avoid wrapping so Markdown tables render cleanly).
	df["DisplayPreset"] = df["Preset"].apply(
		lambda s: preset_display_name(normalize_preset_key(str(s)), wrap_width=None)
	)

	group_cols = ["Model", "DisplayPreset", "Temperature"]
	metric_cols = ["Mean Distinct", "Mean Utility"]
	if df["Seed"].nunique() > 1:
		logger.info("Aggregating over seeds...")
		agg_funcs = {col: ["mean", "std", "count"] for col in metric_cols}
		summary = df.groupby(group_cols).agg(agg_funcs)
		summary.columns = ["_".join(col).strip() for col in summary.columns.values]
		summary = summary.reset_index()
	else:
		summary = df[group_cols + metric_cols]

	# Rename to keep the output column header consistent.
	summary = summary.rename(columns={"DisplayPreset": "Preset"})

	csv_path = output_dir / "summary_metrics.csv"
	md_path = output_dir / "summary_metrics.md"
	try:
		summary.to_csv(csv_path, index=False)
		logger.info(f"Saved summary CSV to {csv_path}")
		with open(md_path, "w") as f:
			f.write("# NoveltyBench Experiment Summary\n\n")
			f.write(summary.to_markdown(index=False))
		logger.info(f"Saved summary Markdown to {md_path}")
		logger.info("\n" + summary.to_markdown(index=False))
	except Exception as e:
		logger.error(f"Failed to write summary tables: {e}")


def postprocess_noveltybench_results(results_dir: Path) -> None:
	"""
	Generate plots and summary tables for completed NoveltyBench runs.

	Args:
		results_dir: Directory containing run subdirectories with results.json files.
	"""
	records = _collect_records(results_dir)
	if not records:
		logger.warning("No results found.")
		return

	df = pd.DataFrame(records)
	working_records = _collect_working_records(results_dir)

	# Generate plots
	try:
		plot_metrics(df, results_dir)
	except Exception as e:
		logger.error(f"Failed to generate plots: {e}")
	if working_records:
		df_working = pd.DataFrame(working_records)
		try:
			plot_metrics(df_working, results_dir, filename_prefix="working_")
		except Exception as e:
			logger.error(f"Failed to generate working-only plots: {e}")

	_write_summary_tables(df, results_dir)


def main(argv: list[str] | None = None) -> None:
	"""CLI entrypoint."""
	args = noveltybench_parser.parse_args(argv)
	if not args.experiment_dir:
		raise ValueError("Missing required flag:\t--experiment_dir")
	results_dir = Path(args.experiment_dir)
	if not results_dir.exists():
		raise ValueError(f"Experiment directory not found: {results_dir}")
	logger.info(f"Scanning for runs under:\t{results_dir}")

	if not args.plot_only:
		if args.runs_to_evaluate:	# evaluate specific runs
			run_dirs = [Path(p).expanduser().resolve() for p in args.runs_to_evaluate]
			evaluate_run_directories(run_dirs, num_partition_workers=args.partition_workers)
		else:						# evaluate all runs in the experiment directory
			evaluate_directories(results_dir, num_partition_workers=args.partition_workers)

	postprocess_noveltybench_results(results_dir=results_dir)

	remote_path = results_dir.resolve()
	user = getpass.getuser()
	hostname = socket.getfqdn()
	logger.info("\n" + "=" * 80)
	logger.info("DOWNLOAD RESULTS")
	logger.info("=" * 80)
	logger.info("To download these results to your local machine, run:")
	logger.info("rsync -avz --exclude '*.pth' --exclude '*.pt' --exclude '*.bin' \\")
	logger.info(f"\t{user}@{hostname}:{remote_path}/ \\")
	logger.info(f"\t./experiments/results/{results_dir.name}/")
	logger.info("=" * 80 + "\n")


if __name__ == "__main__":
	main()
