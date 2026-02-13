"""
Entry point for running NoveltyBench baseline experiments (no Tree-of-Thought).

Example usage (assumes 2 GPUs -- one for generation, one for scoring):

```bash
python -m experiments.noveltybench.run_baseline_noveltybench_experiment \
	--model Qwen3-4B-Instruct-2507 \
	--model_directory /projects/BSTEWART/model_storage \
	--dataset_path /projects/BSTEWART/dataset_storage/noveltybench \
	--split train \
	--results_split debugging \
	--subset curated \
	--max_examples 5 \
	--baseline_signature InstructionFollowing \
	--score_cuda_device 1 \
	--verbosity info \
	--seeds 1024 1025 1026 \
	--baseline_temperature 0.7
```

NOTE: Possible baseline signatures are:
- InstructionFollowing
- InstructionFollowingCoT
- InstructionFollowingWithTools
- InstructionFollowingWithToolsCoT

NOTE: `--results_split debugging` writes outputs under
`experiments/results/noveltybench_{subset}_debugging/...` even when `--split train` is used.
"""

# Standard library imports
import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

# Third-party imports
import dspy
from datasets import Dataset
from tqdm import tqdm

# Local imports
from constants import Verbosity
from experiments.experiment_signatures import (
	InstructionFollowing,
	InstructionFollowingCoT,
	InstructionFollowingWithTools,
	InstructionFollowingWithToolsCoT,
)
from experiments.noveltybench.flags import noveltybench_parser
from experiments.noveltybench.noveltybench_utils import (
	ACTION_SPACE_COMBOS,
	REPO_ROOT,
	format_tool_descriptions_for_tot,
	load_and_prepare_dataset,
)
from experiments.shared_utils import (
	ExampleResult,
	RunResult,
	cleanup_models,
	initialize_vllm_model,
	log_baseline_experiment_summary,
	setup_baseline_run,
	suppress_vllm_logging,
)
from lm.generative_local_lm import GenerativeLocalVLLM
from misc_utils import resolve_model_path, serialize_object
from predict.local_predict import LocalPredict

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _run_baseline_inference(
	args: argparse.Namespace,
	run_output_dir: Path,
	lm: GenerativeLocalVLLM,
	dataset: Dataset,
) -> RunResult:
	"""
	Run standard NoveltyBench baseline inference.

	Args:
		args: The namespace containing experiment arguments.
		run_output_dir: The directory to save results to.
		lm: The local VLLM model to use for inference.
		dataset: The dataset to run inference on.

	Returns:
		RunResult: The results of the baseline inference run.
	"""
	if args.baseline_signature == "InstructionFollowing":
		baseline_signature, use_tools, use_cot = InstructionFollowing, False, False
	elif args.baseline_signature == "InstructionFollowingCoT":
		baseline_signature, use_tools, use_cot = InstructionFollowingCoT, False, True
	elif args.baseline_signature == "InstructionFollowingWithTools":
		baseline_signature, use_tools, use_cot = InstructionFollowingWithTools, True, False
	elif args.baseline_signature == "InstructionFollowingWithToolsCoT":
		baseline_signature, use_tools, use_cot = InstructionFollowingWithToolsCoT, True, True
	else:
		raise ValueError(f"Unknown signature: {args.baseline_signature}")

	predictor = LocalPredict(
		signature=baseline_signature,
		verbose=Verbosity.WARNING,
	)

	processed_examples: list[ExampleResult] = []
	failed_indices: list[int] = []

	config = {
		"temperature": args.baseline_temperature,
		"max_tokens": args.max_tokens,
		"n": args.num_generations,
	}

	config["top_p"] = args.generator_top_p or 1.0
	config["top_k"] = args.generator_top_k or -1.0
	config["min_p"] = args.generator_min_p or 0.0
	config["use_beam_search"] = args.generator_use_beam_search or False

	action_space = None
	if use_tools:
		combo_paths = ACTION_SPACE_COMBOS["controlled"]
		action_space_paths = [str(path) for path in combo_paths]
		action_space = format_tool_descriptions_for_tot(action_space_paths)

	original_level = logger.level
	logger.setLevel(logging.WARNING)

	for idx, example in enumerate(dataset):
		inputs_dict = {"instruction": example["prompt"]}
		if use_tools:
			inputs_dict["action_space"] = action_space
		try:
			preds: list[dspy.Prediction] = predictor(**inputs_dict, config=config, lm=lm)
			pred = preds[0]
			generations: list[str] = []
			reasoning_generations: list[list[str]] = []

			completions = pred.completions
			answer_candidates = completions.result
			reasoning_candidates = completions.rationale if use_cot else None
			for i, candidate in enumerate(answer_candidates):
				generations.append(str(candidate).strip())
				if use_cot and reasoning_candidates:
					reasoning_generations.append([str(reasoning_candidates[i]).strip()])

			processed_examples.append(
				ExampleResult(
					inputs=inputs_dict,
					outputs=generations,
					reasoning_generations=reasoning_generations or None,
				)
			)

		except Exception as e:
			logger.warning(f"Failed baseline example {idx + 1}/{len(dataset)}: {e}")
			failed_indices.append(idx)
			processed_examples.append(
				ExampleResult(
					inputs={"instruction": example["prompt"]},
					outputs=None,
					failure_reason=str(e),
				)
			)

	logger.setLevel(original_level)

	if processed_examples:
		gen_records: list[dict[str, object]] = []
		for idx, ex in enumerate(processed_examples):
			rec: dict[str, object] = {
				"id": idx,
				"prompt": ex.inputs["instruction"],
				"generations": ex.outputs or [],
			}
			if ex.failure_reason:
				rec["failure_reason"] = ex.failure_reason
			gen_records.append(rec)
		with open(run_output_dir / "generations.json", "w") as f:
			json.dump(gen_records, f, indent=2)
		logger.info(f"Saved baseline generations to {run_output_dir / 'generations.json'}")

	result = RunResult(
		metadata=vars(args).copy(),
		examples=processed_examples,
		failed_indices=failed_indices,
		metrics={
			"num_examples": len(processed_examples),
			"num_failed": len(failed_indices),
			"failure_rate": (
				(len(failed_indices) / len(processed_examples)) if processed_examples else 0.0
			),
		},
	)
	result.metadata.update({"timestamp": datetime.now(UTC).isoformat()})
	with open(run_output_dir / "results.json", "w") as f:
		json.dump(serialize_object(asdict(result)), f, indent=2)
	logger.info(f"Saved baseline results to {run_output_dir / 'results.json'}")
	return result


def run_experiment(args: argparse.Namespace) -> None:
	"""Execute a baseline sweep based on parsed args."""

	if not args.baseline_signature:
		raise ValueError("No baseline signature provided. Use --baseline_signature or run the ToT script.")
	if args.action_space_name or args.action_space_paths:
		raise ValueError("Tree-of-Thought arguments detected. Use run_tot_noveltybench_experiment.py instead.")

	args.model = resolve_model_path(args.model, args.model_directory)
	if args.scorer_model:
		args.scorer_model = resolve_model_path(args.scorer_model, args.model_directory)

	suppress_vllm_logging()
	experiment_name = f"noveltybench_{args.subset}_{args.split}"
	log_baseline_experiment_summary(
		logger=logger,
		seeds=args.seeds,
		baseline_signature=args.baseline_signature,
		baseline_temperature=args.baseline_temperature
	)
	logger.info("Initializing generator model...")
	gen_lm = initialize_vllm_model(
		model_path=args.model,
		args=args,
		cuda_device=args.generative_gpu_index,
		is_reranker=False,
	)
	logger.info(f"Loading dataset (split: {args.split})...")
	dataset = load_and_prepare_dataset(
		dataset_path=Path(args.dataset_path),
		split=args.split,
		subset=args.subset,
		max_examples=args.max_examples,
	)
	experiment_output_dirs: list[Path] = []
	try:
		num_runs = len(args.seeds)
		with tqdm(total=num_runs, desc="Baseline runs", unit="run") as pbar:
			for seed in args.seeds:
				run_args, run_name, run_output_dir = setup_baseline_run(
					args=args,
					seed=seed,
					experiment_name=experiment_name,
					gen_lm=gen_lm,
					repo_root=REPO_ROOT,
				)
				experiment_output_dirs.append(run_output_dir)
				pbar.set_description(run_name)
				_run_baseline_inference(
					args=run_args,
					run_output_dir=run_output_dir,
					lm=gen_lm,
					dataset=dataset,
				)
				pbar.update(1)

	finally:
		cleanup_models([gen_lm])

	if experiment_output_dirs:
		results_parent_dir = experiment_output_dirs[0].parent
		logger.info("\n" + "=" * 80)
		logger.info("To process these results, run the following command:")
		logger.info("python -m experiments.noveltybench.evaluate_noveltybench_experiment \\")
		logger.info(f"--experiment_dir {results_parent_dir} \\")
		logger.info("--partition_workers 1")
		logger.info("=" * 80 + "\n")

if __name__ == "__main__":
	args = noveltybench_parser.parse_args()
	run_experiment(args)
