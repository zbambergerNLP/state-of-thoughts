"""
Entry point for running NoveltyBench Tree-of-Thought (ToT) experiments.

Example usage (assumes 3 GPUs -- one for generation, one for reranking, one for scoring):

```bash
python -m experiments.noveltybench.run_tot_noveltybench_experiment \
	--model Qwen3-4B-Instruct-2507 \
	--model_directory /projects/BSTEWART/model_storage \
	--dataset_path /projects/BSTEWART/dataset_storage/noveltybench \
	--reranker_model Qwen3-Reranker-8B \
	--split train \
	--results_split debugging \
	--subset curated \
	--max_examples 5 \
	--action_space_name controlled \
	--tot_signature InstructionFollowingWithReasoning \
	--depth 1 \
	--n_samples_generation 10 \
	--top_k 10 \
	--num_final_candidates 10 \
	--n_final_responses_per_trajectory 1 \
	--score_cuda_device 2 \
	--verbosity info \
	--seeds 1024 1025 1026 \
	--generator_temperature 0.7
```

Note: `--results_split debugging` writes outputs under
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
from datasets import Dataset
from tqdm import tqdm

# Local imports
from constants import Verbosity
from experiments.experiment_signatures import (
	InstructionFollowingWithReasoning,
	InstructionFollowingWithReasoningAndTools,
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
	create_tot_params_from_args,
	extract_tot_reasoning_and_decisions,
	initialize_vllm_model,
	log_tot_experiment_summary,
	setup_tot_run,
	suppress_vllm_logging,
)
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.scoring_local_lm import ScoringLocalVLLM
from misc_utils import resolve_model_path, serialize_object
from predict.tree_of_thoughts import TreeOfThoughts, TreeOfThoughtsOutput

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_tot_inference(
	args: argparse.Namespace,
	run_output_dir: Path,
	generative_lm: GenerativeLocalVLLM,
	reranker_lm: ScoringLocalVLLM | None,
	tot_signature_name: str,
	dataset: Dataset,
	input_keys_mapping: dict[str, str] | None = None,
) -> RunResult:
	"""Run Tree-of-Thought inference for NoveltyBench."""
	if input_keys_mapping is None:
		input_keys_mapping = {"prompt": "instruction"}

	if tot_signature_name == "InstructionFollowingWithReasoning":
		tot_signature = InstructionFollowingWithReasoning
	elif tot_signature_name == "InstructionFollowingWithReasoningAndTools":
		tot_signature = InstructionFollowingWithReasoningAndTools
	else:
		raise ValueError(f"Unknown ToT signature: {tot_signature_name}")

	tot = TreeOfThoughts(
		generator_signature=tot_signature,
		evaluator_signature=None,
		generative_lm=generative_lm,
		reranker_lm=reranker_lm,
		controller_type=args.controller_type,
		evaluator_type=args.evaluator_type,
		controller_tools=None,
		action_space_paths=args.action_space_paths,
		max_reasoning_steps=args.depth,
		early_stopping_enabled=False,
		final_output_kind=args.experiment_mode,
		seed=args.seed,
		verbosity=Verbosity.WARNING,
	)

	examples_results: list[ExampleResult] = []
	failed_indices: list[int] = []

	original_level = logger.level
	logger.setLevel(logging.WARNING)

	dataset_input_keys = input_keys_mapping.keys()
	for idx, example in enumerate(dataset):
		input_state = {input_keys_mapping[key]: example[key] for key in dataset_input_keys}
		prompt = input_state.get("instruction") or str(input_state)
		if "action_space" in tot_signature.input_fields:
			input_state["action_space"] = format_tool_descriptions_for_tot(args.action_space_paths)
		tot_params = create_tot_params_from_args(args)

		try:
			output: TreeOfThoughtsOutput = tot(
				state=input_state,
				tot_parameters=tot_params,
				do_save_tree=args.do_save_tree,
				outputs_directory=str(run_output_dir),
				outputs_filename=f"tot_details_{idx}.json",
			)

			responses = output.response_strings[:10]
			missing_responses = 10 - len(responses)
			if missing_responses > 0:
				responses.extend([responses[-1]] * missing_responses)

			reasoning_generations, reasoning_decisions = extract_tot_reasoning_and_decisions(output)

			if missing_responses > 0:
				if reasoning_generations:
					reasoning_generations.extend([reasoning_generations[-1]] * missing_responses)
				if reasoning_decisions:
					reasoning_decisions.extend([reasoning_decisions[-1]] * missing_responses)

			examples_results.append(
				ExampleResult(
					inputs={"instruction": prompt},
					outputs=responses,
					reasoning_generations=reasoning_generations or None,
					reasoning_decisions=reasoning_decisions or None,
					tree_output=None,
				)
			)

		except Exception as e:
			logger.error(f"Failed prompt {idx}: {e}")
			failed_indices.append(idx)
			examples_results.append(
				ExampleResult(
					inputs={"instruction": prompt},
					outputs=None,
					failure_reason=str(e),
				)
			)

	gen_records: list[dict[str, object]] = []
	for idx, ex in enumerate(examples_results):
		rec: dict[str, object] = {
			"id": idx,
			"prompt": ex.inputs["instruction"],
			"generations": ex.outputs or [],
		}
		if ex.failure_reason:
			rec["failure_reason"] = ex.failure_reason
		gen_records.append(rec)

	logger.setLevel(original_level)

	with open(run_output_dir / "generations.json", "w") as f:
		json.dump(gen_records, f, indent=2)
	logger.info(f"Saved tot generations to {run_output_dir / 'generations.json'}")
	result = RunResult(
		metadata=vars(args).copy(),
		examples=examples_results,
		failed_indices=failed_indices,
		metrics={
			"num_examples": len(examples_results),
			"num_failed": len(failed_indices),
			"failure_rate": (
				(len(failed_indices) / len(examples_results)) if examples_results else 0.0
			),
		},
	)
	result.metadata.update({"timestamp": datetime.now(UTC).isoformat()})

	with open(run_output_dir / "results.json", "w") as f:
		json.dump(serialize_object(asdict(result)), f, indent=2)
	logger.info(f"Saved tot results to {run_output_dir / 'results.json'}")

	return result


def run_experiment(args: argparse.Namespace) -> None:
	"""Execute a ToT sweep based on parsed args."""

	if args.baseline_signature:
		raise ValueError(
			"Baseline signatures detected. Use run_baseline_noveltybench_experiment.py instead."
		)

	args.model = resolve_model_path(args.model, args.model_directory) or args.model
	if args.reranker_model:
		args.reranker_model = (
			resolve_model_path(args.reranker_model, args.model_directory) or args.reranker_model
		)
	if args.scorer_model:
		args.scorer_model = resolve_model_path(args.scorer_model, args.model_directory) or args.scorer_model

	suppress_vllm_logging()

	seeds = args.seeds or [args.seed]
	results_split = args.results_split or args.split
	experiment_name = f"noveltybench_{args.subset}_{results_split}"
	log_tot_experiment_summary(
		logger=logger,
		args=args,
		seeds=seeds,
		tot_signature=args.tot_signature,
		action_space_name=args.action_space_name,
		experiment_mode=args.experiment_mode,
	)
	logger.info("Initializing models...")
	gen_lm = initialize_vllm_model(
		model_path=args.model,
		args=args,
		cuda_device=args.generative_gpu_index,
		is_reranker=False,
	)
	reranker_lm = None
	needs_scoring_model = args.action_space_name != "uncontrolled" and (
		args.controller_type == "reranker" or args.evaluator_type == "reranker"
	)
	if needs_scoring_model:
		if not args.reranker_model:
			raise ValueError("Reranker model path required.")
		if args.reranker_gpu_index == args.generative_gpu_index:
			raise ValueError(
				"Reranker and generative models must use different GPUs. "
				"Set --generative_gpu_index and --reranker_gpu_index to different values."
			)
		reranker_lm = initialize_vllm_model(
			model_path=args.reranker_model,
			args=args,
			cuda_device=args.reranker_gpu_index,
			is_reranker=True,
		)

	logger.info(f"Loading dataset (split: {args.split})...")
	dataset = load_and_prepare_dataset(
		dataset_path=Path(args.dataset_path),
		split=args.split,
		subset=args.subset,
		max_examples=args.max_examples,
	)

	experiment_output_dirs: list[Path] = []

	# Explicitly map action space name to paths
	if args.action_space_name in ACTION_SPACE_COMBOS:
		args.action_space_paths = [str(p) for p in ACTION_SPACE_COMBOS[args.action_space_name]]
	else:
		raise ValueError(f"Unknown action space name: {args.action_space_name}")

	try:
		# Enforce do_pruning = False for NoveltyBench
		args.do_pruning = False
		num_tot_runs = len(seeds)
		with tqdm(total=num_tot_runs, desc="ToT runs", unit="run") as pbar:
			for seed in seeds:
				run_args, run_name, run_output_dir = setup_tot_run(
					args=args,
					seed=seed,
					experiment_name=experiment_name,
					gen_lm=gen_lm,
					reranker_lm=reranker_lm,
					repo_root=REPO_ROOT,
				)
				experiment_output_dirs.append(run_output_dir)
				pbar.set_description(run_name)
				run_tot_inference(
					args=run_args,
					run_output_dir=run_output_dir,
					generative_lm=gen_lm,
					reranker_lm=reranker_lm,
					tot_signature_name=args.tot_signature,
					dataset=dataset,
				)
				pbar.update(1)

	finally:
		cleanup_models([gen_lm, reranker_lm])

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
