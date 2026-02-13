"""Shared utilities for experiment configuration and baseline variants."""

# Standard library imports
import argparse
import copy
import csv
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

# Local imports
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.scoring_local_lm import ScoringLocalVLLM
from misc_utils import log_gpu_state, stringify_without_metadata
from predict.tree_of_thoughts.tree_of_thoughts import TreeOfThoughtsOutput
from predict.tree_of_thoughts.tree_parameters import TreeOfThoughtsParameters

logger = logging.getLogger(__name__)


@dataclass
class ExampleResult:
	"""
	Standardized result for a single example within a run.

	Attributes:
		inputs: Input arguments passed to the model (prompt/instructions).
		outputs: List of final response strings generated for each example.
		failure_reason: Optional failure reason if example failed.
		reasoning_generations: Optional list of list of reasoning strings.
			Outer list corresponds to outputs. Inner list contains reasoning steps.
			For simple CoT, the inner list typically has 1 element.
		reasoning_decisions: Optional list of list of internal controller decisions (ToT).
			Outer list corresponds to outputs. Inner list contains decision tuples.
			Tuple: (intervention_name, internal_reasoning, prefix).
		metrics: Dictionary of metric scores for this specific example.
		tree_output: Optional full tree output (ToT specific).
	"""

	inputs: dict[str, Any]
	outputs: list[str] | None = None
	failure_reason: str | None = None
	reasoning_generations: list[list[str]] | None = None
	reasoning_decisions: list[list[tuple[str, str, str]]] | None = None
	metrics: dict[str, float] | None = None
	tree_output: TreeOfThoughtsOutput | None = None


@dataclass
class RunResult:
	"""
	Standardized result object for a full experiment run.

	Attributes:
		metadata: Experiment metadata including args, vllm_config, timestamps, etc.
		examples: List of ExampleResult objects.
		metrics: Aggregated metrics for the entire run (e.g. mean accuracy).
		failed_indices: List of example indices that failed during processing.
	"""

	metadata: dict[str, Any]
	examples: list[ExampleResult]
	metrics: dict[str, float] | None = None
	failed_indices: list[int] = field(default_factory=list)


def suppress_vllm_logging() -> None:
	"""Suppress vLLM logging by setting vLLM-related loggers to WARNING level."""
	vllm_loggers = [
		"vllm",
		"vllm.engine",
		"vllm.worker",
		"vllm.distributed",
		"vllm.model_executor",
		"vllm.attention",
		"vllm.core",
		"vllm.utils",
	]
	for logger_name in vllm_loggers:
		logging.getLogger(logger_name).setLevel(logging.WARNING)


def _resolve_cuda_visible_device(original_visible: str | None, requested: str) -> str:
	"""Resolve a requested GPU index against SLURM-provided CUDA_VISIBLE_DEVICES.

	On many clusters, SLURM sets CUDA_VISIBLE_DEVICES to a comma-separated list of *physical*
	GPU IDs allocated to the job (e.g. "3,7"). In that case, users typically want to refer
	to these as index 0/1 within the job. This helper maps requested indices into that list.

	Args:
		original_visible: The original CUDA_VISIBLE_DEVICES string. For example: "3,7".
		requested: The requested GPU index. For example: "0".

	Returns:
		The resolved GPU index. For example: "3".
	"""
	if not original_visible:
		return requested

	visible = [d.strip() for d in original_visible.split(",") if d.strip()]
	if not visible:
		return requested

	# If user provided a physical ID that is already visible, keep it.
	if requested in visible:
		return requested

	# Otherwise, treat the request as an index into the visible list.
	try:
		idx = int(requested)
	except ValueError:
		return requested

	if 0 <= idx < len(visible):
		return visible[idx]
	return requested


def _camel_to_snake_for_run_names(name: str) -> str:
	"""Convert CamelCase identifiers to snake_case for run/preset names.

	Note: Some of our signature names contain the acronym `CoT`, which naive splitting would
	turn into `co_t`. We normalize a few known acronyms to keep run names stable.
	"""
	snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
	# Normalize common acronym splits in our signature names.
	snake = snake.replace("_co_t", "_cot")
	snake = snake.replace("co_t", "cot")
	snake = snake.replace("_to_t", "_tot")
	snake = snake.replace("to_t", "tot")
	# Deduplicate underscores
	snake = re.sub(r"_+", "_", snake)
	return snake

def normalize_preset_key(preset: str) -> str:
	"""Normalize preset keys so plotting uses stable, semantic configuration names.

	Args:
		preset: Raw preset key string.

	Returns:
		Normalized preset key with hyperparameter prefixes stripped and proper
		baseline/tot prefixes ensured.
	"""
	p = str(preset or "").strip()
	if not p:
		return p

	# Strip depth/samples/top_k prefixes (these are hyperparams, not semantic presets).
	tot_param_prefix_pattern = re.compile(r"^(?:tot|uncontrolled)_d\d+_s\d+_k\d+_")
	p = tot_param_prefix_pattern.sub("", p)

	# Ensure ToT presets start with "tot_" so preset_display_name() classifies them correctly.
	if not p.startswith(("baseline_", "tot_")):
		p = f"tot_{p}"

	return p

def build_preset_key_from_metadata(metadata: dict[str, Any]) -> str:
	"""Build a display-friendly preset key from stored run metadata.

	Args:
		metadata: Run metadata dictionary (typically from results.json).

	Returns:
		A preset key string suitable for display/plotting utilities.
	"""
	if not metadata:
		return "unknown"

	baseline_signature = metadata.get("baseline_signature")
	if baseline_signature:
		return f"baseline_{baseline_signature}"

	tot_signature = metadata.get("tot_signature")
	if tot_signature:
		action_space_name = metadata.get("action_space_name") or metadata.get("action_space")
		experiment_mode = metadata.get("experiment_mode")
		if not action_space_name:
			action_space_name = "uncontrolled"
		sig_snake = _camel_to_snake_for_run_names(str(tot_signature))
		preset_key = f"tot_{action_space_name}_{sig_snake}_{experiment_mode}"
		# Add evaluator type suffix if programmatic to match run name
		if metadata.get("evaluator_type") == "programmatic":
			preset_key = f"{preset_key}_programmatic"
		return preset_key

	return "unknown"


def format_tool_descriptions_from_action_space(
	action_space_dir: Path,
	files_to_titles: dict[str, str] | None = None,
	action_space_paths: list[str | Path] | None = None,
) -> str:
	"""
	Generate tool description string from action space definitions.

	Args:
		action_space_dir: Directory containing action space JSON files (used if paths not provided).
		files_to_titles: Optional mapping from JSON filename to display title.
			If None, uses default titles based on dimension names.
		action_space_paths: Optional list of specific JSON file paths to include.
			Overrules directory globbing if provided.

	Returns:
		Formatted multi-line string with tool descriptions.
	"""
	lines = ["Consider the following dimensions for your response:", ""]

	if files_to_titles is None:
		files_to_titles = {}

	# Determine which files to process
	if action_space_paths:
		json_files = [Path(p) for p in action_space_paths]
	else:
		json_files = sorted(action_space_dir.glob("*.json"))

	for json_file in json_files:
		with open(json_file, encoding="utf-8") as f:
			data = json.load(f)

		dimension_name = data.get("name", json_file.stem)
		title = files_to_titles.get(json_file.name, dimension_name.upper())
		choices = data.get("choices", {})

		if not choices:
			continue

		lines.append(f"{title}:")
		for i, (key, value) in enumerate(choices.items(), 1):
			definition = value.get("definition", "")
			item_str = f"{i}. {key}: {definition}"
			lines.append(item_str)
		lines.append("")

	return "\n".join(lines).strip()


# =============================================================================
# Model Initialization Helpers
# =============================================================================


def initialize_vllm_model(
	model_path: str,
	args: argparse.Namespace,
	cuda_device: str = "0",
	is_reranker: bool = False,
) -> GenerativeLocalVLLM | ScoringLocalVLLM:
	"""
	Initialize a vLLM model (generative or reranker) with standard arguments.

	Args:
		model_path: Path or name of the model to load.
		args: Namespace or object containing standard vLLM configuration args:
			  (tensor_parallel_size, dtype, gpu_memory_utilization, max_model_len, enforce_eager).
			  The args can be prefixed with 'generator_' or 'reranker_' depending on `is_reranker`.
		cuda_device: CUDA device string (e.g., "0", "1") to set CUDA_VISIBLE_DEVICES.
		is_reranker: If True, initializes ScoringLocalVLLM; otherwise GenerativeLocalVLLM.

	Returns:
		Initialized GenerativeLocalVLLM or ScoringLocalVLLM instance.
	"""
	log_prefix = "Reranker" if is_reranker else "Generator"
	tensor_parallel_size = args.reranker_tensor_parallel_size if is_reranker else args.generator_tensor_parallel_size
	dtype = args.reranker_dtype if is_reranker else args.generator_dtype
	gpu_utilization = args.reranker_gpu_memory_utilization if is_reranker else args.generator_gpu_memory_utilization
	max_model_len = args.reranker_max_model_len if is_reranker else args.generator_max_model_len
	enforce_eager = args.reranker_enforce_eager if is_reranker else args.generator_enforce_eager
	seed = args.seed
	logger.info(f"Initializing {log_prefix} vLLM model: {model_path}")

	# Set CUDA device
	original_device = os.environ.get("CUDA_VISIBLE_DEVICES")
	resolved_device = _resolve_cuda_visible_device(original_device, cuda_device)
	os.environ["CUDA_VISIBLE_DEVICES"] = resolved_device

	try:
		if is_reranker:
			model = ScoringLocalVLLM(
				model=model_path,
				tensor_parallel_size=tensor_parallel_size,
				dtype=dtype,
				gpu_memory_utilization=gpu_utilization,
				max_model_len=max_model_len,
				enforce_eager=enforce_eager,
				seed=seed,
				trust_remote_code=True,
			)
		else:
			model = GenerativeLocalVLLM(
				model=model_path,
				tensor_parallel_size=tensor_parallel_size,
				dtype=dtype,
				gpu_memory_utilization=gpu_utilization,
				max_model_len=max_model_len,
				enforce_eager=enforce_eager,
				enable_prefix_caching=True,
				seed=seed,
				trust_remote_code=True,
			)

		logger.info(f"{log_prefix} vLLM ready.")
		log_gpu_state(f"{log_prefix} init")
		return model

	finally:
		# Restore original CUDA_VISIBLE_DEVICES
		if original_device is None:
			os.environ.pop("CUDA_VISIBLE_DEVICES", None)
		else:
			os.environ["CUDA_VISIBLE_DEVICES"] = original_device


def cleanup_models(models: list[GenerativeLocalVLLM | ScoringLocalVLLM | None]) -> None:
	"""Release GPU memory used by models.

	Args:
		models: List of models to clean up.
	"""
	for model in models:
		if model is not None and hasattr(model, "kill"):
			model.kill()
	# vLLM/torch distributed can leave an initialized process group behind, which triggers NCCL warnings.
	try:
		if torch.distributed.is_available() and torch.distributed.is_initialized():
			torch.distributed.destroy_process_group()
	except Exception as e:
		logger.debug(f"Failed_to_destroy_process_group:\t{e}")
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	logger.info("Models cleaned up")


# =============================================================================
# Result Export Helpers
# =============================================================================


def export_results_csv(
	results: list[dict[str, Any]],
	output_path: Path,
	fieldnames: list[str] | None = None,
	fields_to_skip: list[str] | None = None,
) -> None:
	"""
	Export a list of result objects (dataclasses or dicts) to CSV.

	Args:
		results: List of result objects (must be dataclasses or dicts).
		output_path: Path to write the CSV file.
		fieldnames: Optional list of field names. If None, inferred from first result.
		fields_to_skip: Optional list of field names to skip. If None, no fields are skipped.
	"""
	if not results:
		logger.warning("No results to export to CSV")
		return

	# Infer fieldnames from the first item if not provided
	if fieldnames is None:
		first_item = results[0]
		if is_dataclass(first_item):
			fieldnames = list(asdict(first_item).keys())
		elif isinstance(first_item, dict):
			fieldnames = list(first_item.keys())
		else:
			raise ValueError(f"Cannot infer fieldnames from type {type(first_item)}")

		# Filter out complex objects usually not wanted in CSV summary
		if fields_to_skip is not None:
			fieldnames = [f for f in fieldnames if f not in fields_to_skip]

	with open(output_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for result in results:
			row = asdict(result) if is_dataclass(result) else result
			# Remove keys not in fieldnames to avoid errors
			filtered_row = {k: v for k, v in row.items() if k in fieldnames}
			writer.writerow(filtered_row)

	logger.info(f"Exported CSV to {output_path}")


def get_run_name(seed: int, temperature: float, config_name: str, **kwargs) -> str:
	"""
	Generate a standard run directory name.

	Format: {config_name}_seed{seed}_temp{temperature}

	Args:
		seed: Random seed.
		temperature: Generation temperature.
		config_name: Configuration name (e.g. "baseline_cot", "tot_personalities").
		**kwargs: Ignored key-value pairs (for compatibility).

	Returns:
		Formatted run name string.
	"""
	# Directory structure is {experiment}/{model}/{run_name}.
	assert config_name, "config_name must not be empty"
	temp_str = str(temperature).replace('.', '_')
	return f"{config_name}_seed_{seed}_temp_{temp_str}"


def create_tot_params_from_args(args: argparse.Namespace) -> TreeOfThoughtsParameters:
	"""
	Construct TreeOfThoughtsParameters from parsed command-line arguments.

	Args:
		args: Parsed arguments (Namespace or similar) containing standard ToT flags.

	Returns:
		Configured TreeOfThoughtsParameters object.
	"""
	return TreeOfThoughtsParameters(
		# Tree-Search Parameters
		depth=args.depth,
		n_samples_generation=args.n_samples_generation,
		top_k=args.top_k,
		num_final_candidates=args.num_final_candidates,
		n_final_responses_per_trajectory=args.n_final_responses_per_trajectory,
		do_pruning=args.do_pruning,
		use_self_consistency=args.use_self_consistency,
		# Evaluator parameters
		n_samples_judge=args.n_samples_judge,
		judge_temperature=args.judge_temperature,
		judge_top_p=args.judge_top_p,
		judge_top_k=args.judge_top_k,
		judge_min_p=args.judge_min_p,
		judge_use_beam_search=args.judge_use_beam_search or False,
		judge_max_tokens=args.judge_max_tokens,
		# Controller parameters
		controller_temperature=args.controller_temperature,
		controller_top_p=args.controller_top_p,
		controller_top_k=args.controller_top_k,
		controller_min_p=args.controller_min_p,
		controller_max_tokens=args.controller_max_tokens,
		controller_use_beam_search=args.controller_use_beam_search or False,
		# Generator parameters
		generator_temperature=args.generator_temperature,
		generator_max_tokens=args.generator_max_tokens,
		generator_top_p=args.generator_top_p,
		generator_top_k=args.generator_top_k,
		generator_min_p=args.generator_min_p,
		generator_use_beam_search=args.generator_use_beam_search or False,
	)

def log_baseline_experiment_summary(
	logger: logging.Logger,
	seeds: list[int],
	baseline_signature: str,
	baseline_temperature: float,
) -> None:
	"""
	Log specific summary for Baseline experiments.

	Args:
		logger: Logger object.
		seeds: List of seeds.
		baseline_signature: Baseline signature string.
		baseline_temperature: Baseline temperature.
	"""
	configs = []
	for seed in seeds:
		configs.append(
			{
				"name": f"Baseline: {baseline_signature} (Seed={seed}, Temp={baseline_temperature})",
				"params": {
					"type": "baseline",
					"seed": seed,
					"temperature": baseline_temperature,
					"signature": baseline_signature,
				},
			}
		)
	_log_configs(logger, configs)


def log_tot_experiment_summary(
	logger: logging.Logger,
	args: argparse.Namespace,
	seeds: list[int],
	tot_signature: str,
	action_space_name: str | None,
	experiment_mode: str | list[str],
) -> None:
	"""
	Log specific summary for ToT experiments.

	Args:
		logger: Logger object.
		args: Parsed arguments containing all ToT parameters.
		seeds: List of seeds.
		tot_signature: ToT signature string.
		action_space_name: Name of the action space (or None).
		experiment_mode: Experiment mode (string or list of strings).
	"""
	configs = []
	# Handle list or string for experiment_mode
	modes = [experiment_mode] if isinstance(experiment_mode, str) else experiment_mode

	for mode in modes:
		for seed in seeds:
			configs.append(
				{
					"name": (
						f"ToT: {action_space_name} | {tot_signature} | {mode} "
						f"(Seed={seed}, GenTemp={args.generator_temperature})"
					),
					"params": {
						"type": "tot",
						"seed": seed,
						"signature": tot_signature,
						"action_space": action_space_name if action_space_name else "Uncontrolled",
						"mode": mode,
						# Search parameters
						"depth": args.depth,
						"branching_factor": args.n_samples_generation,
						"beam_width": args.top_k,
						"num_final_candidates": args.num_final_candidates,
						"n_final_responses": args.n_final_responses_per_trajectory,
						"do_pruning": args.do_pruning,
						"use_self_consistency": args.use_self_consistency,
						# Evaluator parameters
						"evaluator_type": args.evaluator_type,
						"judge_temperature": args.judge_temperature,
						"judge_max_tokens": args.judge_max_tokens,
						"n_samples_judge": args.n_samples_judge,
						# Controller parameters
						"controller_type": args.controller_type,
						"controller_temperature": args.controller_temperature,
						"controller_max_tokens": args.controller_max_tokens,
						# Generator parameters
						"generator_temperature": args.generator_temperature,
						"generator_max_tokens": args.generator_max_tokens,
						"generator_top_p": args.generator_top_p,
					},
				}
			)
	_log_configs(logger, configs)


def _log_configs(logger: logging.Logger, configs: list[dict[str, Any]]) -> None:
	"""
	Internal helper to pretty-print configs.

	Args:
		logger: Logger object.
		configs: List of configs to print.
	"""
	if not configs:
		logger.warning("No experiments scheduled to run (check arguments).")
		return

	width = 80
	logger.info("\n" + "=" * width)
	logger.info(f" EXPERIMENT CONFIGURATION SUMMARY ({len(configs)} runs scheduled)".center(width))
	logger.info("=" * width + "\n")

	for i, config in enumerate(configs, 1):
		logger.info(f"{i}. {config['name']}")
		for key, value in config['params'].items():
			logger.info(f"\t* {key:<20} : {value}")
		logger.info("-" * 40)

	logger.info("\n" + "=" * width + "\n")


def setup_baseline_run(
	args: argparse.Namespace,
	seed: int,
	experiment_name: str,
	gen_lm: GenerativeLocalVLLM,
	repo_root: Path,
) -> tuple[argparse.Namespace, str, Path]:
	"""
	Setup a baseline run: create directory and return arguments for execution.

	Args:
		args: Argument namespace/object.
		seed: Random seed.
		experiment_name: Name of the experiment project (e.g. "noveltybench").
		gen_lm: Generative LM object.
		repo_root: Root path of the repository.

	Returns:
		A 3-tuple of (run_args, run_name, output_dir).
			- run_args: The arguments object (namespace) with run-specific arguments set.
			- run_name: The run name.
			- output_dir: The output directory path for the run.
	"""
	run_args = copy.deepcopy(args)
	run_args.seed = seed

	# Convert CamelCase signature to snake_case for cleaner paths
	# e.g. InstructionFollowingCoT -> instruction_following_cot
	run_name = get_run_name(
		seed=seed,
		temperature=args.baseline_temperature,
		config_name=f"baseline_{_camel_to_snake_for_run_names(args.baseline_signature)}"
	)

	# Path structure: {experiment}/{model}/{run_name}
	model_name = Path(args.model).name.lower().replace("-", "_")
	output_dir = repo_root / "experiments" / "results" / experiment_name / model_name / run_name
	output_dir.mkdir(parents=True, exist_ok=True)
	return run_args, run_name, output_dir


def setup_tot_run(
	args: argparse.Namespace,
	seed: int,
	experiment_name: str,
	gen_lm: GenerativeLocalVLLM,
	reranker_lm: ScoringLocalVLLM | None,
	repo_root: Path,
	evaluator_prm_signature: str | None = None,
	evaluator_orm_signature: str | None = None,
) -> tuple[argparse.Namespace, str, Path]:
	"""
	Setup a ToT run: create directory and return arguments for execution.

	Args:
		args: Argument namespace/object.
		seed: Random seed.
		experiment_name: Name of the experiment project (e.g. "noveltybench").
		gen_lm: Generative LM object.
		reranker_lm: Reranker LM object.
		repo_root: Root path of the repository.
		evaluator_prm_signature: Optional signature name used for PRM-style evaluation.
		evaluator_orm_signature: Optional signature name used for ORM-style evaluation.

	Returns:
		A 3-tuple of (run_args, run_name, output_dir).
			- run_args: The arguments object (namespace) with run-specific arguments set.
			- run_name: The run name.
			- output_dir: The output directory path for the run.
	"""
	run_args = copy.deepcopy(args)
	run_args.seed = seed

	# Set evaluator signatures if provided
	if evaluator_prm_signature is not None:
		run_args.evaluator_prm_signature = evaluator_prm_signature
	if evaluator_orm_signature is not None:
		run_args.evaluator_orm_signature = evaluator_orm_signature

	# Set action space paths
	if args.action_space_name not in {"controlled", "uncontrolled"}:
		raise ValueError(f"action_space_name must be 'controlled' or 'uncontrolled', got '{args.action_space_name}'")
	run_args.action_space_paths = (
		[str(p) for p in args.action_space_paths] if args.action_space_name == "controlled" else []
	)

	# Set preset name
	base_preset = (
		f"uncontrolled_tot_d{args.depth}_s{args.n_samples_generation}_k{args.top_k}"
		if args.action_space_name == "uncontrolled"
		else f"controlled_tot_d{args.depth}_s{args.n_samples_generation}_k{args.top_k}_{args.action_space_name}"
	)
	# Convert CamelCase signature to snake_case for cleaner paths
	# e.g. InstructionFollowingWithReasoning -> instruction_following_with_reasoning
	tot_sig_snake = _camel_to_snake_for_run_names(args.tot_signature)
	preset = f"{base_preset}_{tot_sig_snake}_{args.experiment_mode}"

	# Add evaluator type suffix to distinguish runs with different evaluators
	if args.evaluator_type == "programmatic":
		preset = f"{preset}_programmatic"

	# For postprocessing identification
	run_name = get_run_name(seed=seed, temperature=args.generator_temperature, config_name=preset)
	model_name = Path(args.model).name.lower().replace("-", "_")
	output_dir = (
		repo_root / "experiments/results" / experiment_name / model_name / run_name
	).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	return run_args, run_name, output_dir


def extract_tot_reasoning_and_decisions(
	tot_output: TreeOfThoughtsOutput,
) -> tuple[list[list[str]], list[list[dict[str, Any]]]]:
	"""
	Extract reasoning generations and controller decisions from ToT output.

	Args:
		tot_output: The TreeOfThoughtsOutput object properly typed.

	Returns:
		A tuple containing:
			- reasoning_generations: List of lists of reasoning generation strings/dicts.
			- reasoning_decisions: List of lists of decision dictionaries.
	"""
	reasoning_generations = []
	reasoning_decisions = []
	node_scores = []

	chains = tot_output.reasoning_steps or []

	# TODO[P1]: Make sure that chains which are extracted/used here have valid chains.
	#	We should not have any trajectories which are empty or have no nodes...
	for chain in chains:
		# The final node in the chain contains the accumulated state
		# Chains should always have at least one node (the target node itself)
		if not chain.nodes:
			logger.warning(
				"Encountered reasoning chain with no nodes. "
				"This may indicate a failure during tree construction."
			)
			reasoning_generations.append([])
			reasoning_decisions.append([])
			node_scores.append([])
			continue

		final_node = chain.nodes[-1]
		state = final_node.state

		# Extract reasoning steps
		# Note: reasoning contains all steps accumulated, including any error-related content
		steps = state.reasoning
		# Convert each step to a string, excluding metadata like `error`.
		reasoning_generations.append(list(map(stringify_without_metadata, steps)))

		# Extract node scores and controller decisions
		node_scores.append([node.score for node in chain.nodes])
		reasoning_decisions.append(
			[asdict(decision) for decision in state.controller_output_trajectory]
		)

	return reasoning_generations, reasoning_decisions


# =============================================================================
# Plotting Utilities
# =============================================================================

# Standard preset display names in the desired order (left to right)
PRESET_DISPLAY_ORDER = [
	"Baseline",
	"Baseline w/ Action Space",
	"Baseline CoT",
	"Baseline CoT w/ Action Space",
	"Baseline ToT",
	"Baseline ToT w/ Action Space",
	"STATe of Thoughts",
	"STATe of Thoughts w/ Action Space",
]

# Color mapping for preset display names
# Using colorblind-friendly palette (colorblind-safe colors)
PRESET_COLOR_MAP: dict[str, str] = {
	"Baseline": "#0173B2",  # Blue
	"Baseline w/ Action Space": "#DE8F05",  # Orange
	"Baseline CoT": "#029E73",  # Teal
	"Baseline CoT w/ Action Space": "#CC78BC",  # Pink
	"Baseline ToT": "#56B4E9",  # Light blue
	"Baseline ToT w/ Action Space": "#ECE133",  # Yellow
	"STATe of Thoughts": "#0072B2",  # Dark blue
	"STATe of Thoughts w/ Action Space": "#D55E00",  # Vermillion
}


def get_preset_display_order(present_presets: list[str]) -> list[str]:
	"""
	Get the ordered list of preset display names, filtering to only those present.

	Args:
		present_presets: List of preset display names that are actually present in the data.

	Returns:
		Ordered list of preset display names in the standard order, filtered to only
		include those that are present.
	"""
	return [preset for preset in PRESET_DISPLAY_ORDER if preset in present_presets]


def style_boxplot(ax: plt.Axes, order: list[str], palette: dict[str, str]) -> None:
	"""
	Style seaborn boxplots with consistent colors and formatting.

	This function makes boxplots lighter and color-consistent per category, suitable
	for metrics where "higher is better".

	Args:
		ax: Matplotlib axes object containing the boxplot.
		order: Ordered list of category names (x-axis labels).
		palette: Dictionary mapping category names to colors.
	"""
	boxes = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
	for patch, label in zip(boxes, order, strict=False):
		color = palette.get(label, "#666666")
		patch.set_facecolor(color)
		patch.set_alpha(0.22)
		patch.set_edgecolor(color)
		patch.set_linewidth(1.4)

	lines = ax.lines
	if not boxes:
		return
	chunk = max(1, len(lines) // len(boxes))
	for i, label in enumerate(order):
		color = palette.get(label, "#666666")
		start = i * chunk
		end = min(len(lines), (i + 1) * chunk)
		for j, line in enumerate(lines[start:end]):
			line.set_color(color)
			line.set_alpha(0.85)
			line.set_linewidth(2.0 if j == 4 else 1.2)
