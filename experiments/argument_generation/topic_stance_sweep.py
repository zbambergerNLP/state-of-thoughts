"""
Topic-Stance Sweep Experiment: Multi-Topic Argument Generation

This script runs TreeOfThoughts argument generation across multiple topics and stances
using SYNTHESIS_RESTRUCTURED output mode. Each job handles one topic/stance combination
without persona conditioning.

Key features:
- Takes --topic and --stance as CLI arguments
- Uses SYNTHESIS_RESTRUCTURED for coherent argument reorganization
- 100-tool action space (10 structures × 10 subtopics)
- Individual CSV per job for robustness
- Fixed seed for consistency across runs

Usage:
	python experiments/argument_generation/topic_stance_sweep.py [args...]

Example usage:

```bash
python experiments/argument_generation/topic_stance_sweep.py \
--model Qwen3-30B-A3B-Instruct-2507 \
--reranker_model Qwen3-Reranker-8B \
--generative_gpu_index 0 \
--reranker_gpu_index 1 \
--outputs_directory experiments/argument_generation/topic_stance_sweep/ \
--outputs_filename topic_stance_sweep \
--topic "The government should implement a Universal Basic Income (UBI) for all citizens." \
--stance PRO \
--seed 42 \
--depth 3 \
--n_samples_generation 25 \
--top_k 25 \
--n_samples_judge 1 \
--generator_temperature 0.7 \
--experiment_mode synthesis_restructured \
--do_save_tree
```

For batch submission, use the generated SBATCH scripts:
```bash
bash experiments/argument_generation/setup_topic_stance_sweep.sh
bash experiments/argument_generation/scripts/topic_stance_sweep/launch_topic_stance_sweep.sh
```
"""

import csv
import logging
import os
import time

import dspy
from dotenv import load_dotenv

# Local imports
from adapter.constraints import ResponseLength
from constants import Verbosity
from experiments.argument_generation.flags import parser as base_parser
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.scoring_local_lm import ScoringLocalVLLM
from predict.tree_of_thoughts import TreeOfThoughts
from predict.tree_of_thoughts.tree_of_thoughts import TreeOfThoughtsOutput
from predict.tree_of_thoughts.tree_parameters import TreeOfThoughtsParameters
from signatures.example_signatures import (
	GenerateArgumentWithReasoning,  # Note: No persona variant
)

load_dotenv()

# Set up logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add topic and stance arguments to parser
argument_generation_group = None
for action_group in base_parser._action_groups:
	if action_group.title and "argument generation" in action_group.title.lower():
		argument_generation_group = action_group
		break

if argument_generation_group is None:
	# Create the group if it doesn't exist
	argument_generation_group = base_parser.add_argument_group(
		"Argument Generation Options"
	)

argument_generation_group.add_argument(
	"--topic",
	type=str,
	required=True,
	help="Topic for argument generation (full text)",
)

argument_generation_group.add_argument(
	"--stance",
	type=str,
	required=True,
	choices=["PRO", "ANTI"],
	help="Stance on the topic (PRO or ANTI)",
)


def initialize_models(args):
	"""Initialize two vLLM language models on separate GPUs."""
	# Model configuration
	generative_model_name = args.model
	generative_full_model_path = os.path.join(
		args.model_directory, generative_model_name
	)
	reranker_model_name = args.reranker_model
	reranker_full_model_path = os.path.join(args.model_directory, reranker_model_name)

	# Initialize generative model on specified GPU
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

	# Initialize reranker model on specified GPU
	logger.info(
		f"Initializing reranker vLLM model on GPU {args.reranker_gpu_index} "
		f"from: {reranker_full_model_path}"
	)
	logger.info(
		f"Using HuggingFace model: {reranker_model_name} (will be configured for scoring)"
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

	# After both models are initialized, set CUDA_VISIBLE_DEVICES to show both GPUs
	os.environ["CUDA_VISIBLE_DEVICES"] = (
		f"{args.generative_gpu_index},{args.reranker_gpu_index}"
	)
	logger.info("Both GPUs are now visible for runtime operations")

	# Set default LM for dspy settings (use generative model)
	dspy.settings.configure(lm=generative_lm)
	return generative_lm, reranker_lm


def run_generation(args):
	"""Run argument generation for specified topic and stance."""
	generative_lm, reranker_lm = initialize_models(args)

	# Stance is already a string matching ArgumentStance Literal type
	stance_enum = args.stance

	try:
		# Get experiment mode
		final_output_kind = (
			args.experiment_mode[0]
			if isinstance(args.experiment_mode, list)
			else args.experiment_mode
		)

		# Define constraints
		thought_length = ResponseLength(
			granularity="sentence",
			bounds=(1, 3),
		)
		response_length = ResponseLength(
			granularity="sentence",
			bounds=(5, 7),
		)

		# Configure 100-tool action space: 10 structures × 10 subtopics
		action_space_dir = os.path.join(os.path.dirname(__file__), "action_space")
		action_space_paths = [
			os.path.join(action_space_dir, "causal_structures.json"),  # 10 structures
			os.path.join(action_space_dir, "causal_subtopics.json"),  # 10 subtopics
		]

		logger.info(
			f"\nTopic-Stance Sweep Experiment Configuration:\n"
			f"Topic: {args.topic}\n"
			f"Stance: {args.stance}\n"
			f"Seed: {args.seed}\n"
			f"Action space paths:\n\t{action_space_paths}\n"
			f"thought length: {thought_length}\n"
			f"response length: {response_length}\n"
			f"depth: {args.depth} (max_reasoning_steps={args.depth}, CSV columns for {args.depth + 1} controller outputs)\n"
			f"final output kind: {final_output_kind}\n"
			f"early stopping enabled: {args.early_stopping_enabled}\n"
			f"use self consistency: {args.use_self_consistency}\n"
			f"do pruning: {args.do_pruning}\n"
			f"use beam search: {args.controller_use_beam_search}\n"
			f"controller temperature: {args.controller_temperature}\n"
			f"generator temperature: {args.generator_temperature}\n"
			f"generator use beam search: {args.generator_use_beam_search}\n"
			f"num final candidates: {args.num_final_candidates}\n"
			f"n_samples_generation (branching factor): {args.n_samples_generation}\n"
			f"top_k: {args.top_k}\n"
		)

		# Calculate max_reasoning_steps from depth
		max_reasoning_steps = args.depth

		# Initialize TreeOfThoughts with basic argument signature (no persona)
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,  # No persona field
			evaluator_signature=None,
			generative_lm=generative_lm,
			reranker_lm=reranker_lm,
			controller_type="reranker",
			thought_length=thought_length,
			response_length=response_length,
			max_reasoning_steps=max_reasoning_steps,
			final_output_kind=final_output_kind,
			early_stopping_enabled=args.early_stopping_enabled,
			action_space_paths=action_space_paths,
			seed=args.seed,
			verbosity=Verbosity.INFO,
		)

		# Validate action space created 100 tools (10 structures × 10 subtopics)
		num_tools = len(tot.controller.tools)
		logger.info(f"Action space created {num_tools} tools")
		if num_tools != 100:
			logger.warning(
				f"Expected 100 tools from Cartesian product (10 structures × 10 subtopics), "
				f"but got {num_tools}. Check action space JSON files."
			)

		tot_parameters = TreeOfThoughtsParameters(
			depth=args.depth,
			n_samples_generation=args.n_samples_generation,
			top_k=args.top_k,
			top_k_first=args.top_k_first,
			n_samples_judge=args.n_samples_judge,
			judge_temperature=args.judge_temperature,
			generator_temperature=args.generator_temperature,
			controller_temperature=args.controller_temperature,
			controller_use_beam_search=args.controller_use_beam_search,
			generator_use_beam_search=args.generator_use_beam_search,
			num_final_candidates=args.num_final_candidates,
			use_self_consistency=args.use_self_consistency,
			n_final_responses_per_trajectory=1,  # Single final response per trajectory
		)

		# Ensure output directory exists
		if args.outputs_directory:
			os.makedirs(args.outputs_directory, exist_ok=True)

		# Create CSV output path and timestamp (shared CSV for all runs)
		csv_filename = f"{args.outputs_filename}_all_results.csv"
		csv_file_path = os.path.join(args.outputs_directory, csv_filename)
		run_timestamp = time.strftime("%Y%m%d_%H%M%S")
		logger.info(f"CSV results will be saved to: {csv_file_path}")

		# Single generation run
		logger.info(f"Generating for Topic: '{args.topic}', Stance: '{args.stance}'")

		# Build input data (no persona)
		input_data = {
			"topic": args.topic,
			"stance": stance_enum,
		}

		# Construct filename for JSON tree (individual file per run)
		# Create simple topic identifier from first 50 chars
		topic_slug = "".join(c if c.isalnum() else "_" for c in args.topic[:50])
		filename = (
			f"{args.outputs_filename}_{topic_slug.lower()}_{args.stance.lower()}.json"
		)

		tot_output = tot.forward(
			state=input_data,
			tot_parameters=tot_parameters,
			do_save_tree=args.do_save_tree,
			outputs_directory=args.outputs_directory,
			outputs_filename=filename,
		)

		# Export to CSV
		export_trajectory_to_csv(
			tot_output=tot_output,
			topic=args.topic,
			stance=args.stance,
			depth=args.depth,
			seed=args.seed,
			timestamp=run_timestamp,
			csv_file_path=csv_file_path,
		)

		# Count leaf nodes for logging
		num_leaf_nodes = sum(
			1 for node in tot_output.tree.nodes.values() if not node.children_ids
		)
		logger.info(
			f"Completed Topic: '{args.topic}', Stance: '{args.stance}'. "
			f"Exported {num_leaf_nodes} complete arguments (leaf nodes) to CSV."
		)

	finally:
		try:
			generative_lm.kill()
			reranker_lm.kill()
		except Exception as e:
			logger.warning(f"Error during cleanup: {e}")


def export_trajectory_to_csv(
	tot_output: TreeOfThoughtsOutput,
	topic: str,
	stance: str,
	depth: int,
	seed: int,
	timestamp: str,
	csv_file_path: str,
) -> None:
	"""Export TreeOfThoughts trajectory data to CSV in wide format.

	Creates one row per final argument with columns for metadata and
	step-by-step reasoning/controller data.

	Trajectory structure:
		- Layer 0: Root (input only)
		- Layers 1 to depth: Reasoning steps
		- Layer depth+1: Final response
		- Controller outputs: depth+1 decisions (one per layer transition)

	Args:
		tot_output: TreeOfThoughts output with responses and reasoning chains
		topic: Topic string
		stance: "PRO" or "ANTI"
		depth: Tree search depth (number of reasoning layers)
		seed: Random seed used for generation
		timestamp: Generation timestamp
		csv_file_path: Path to CSV file (creates or appends)
	"""
	# Calculate max controller outputs: depth + 1 (includes final "finish" decision)
	max_controller_outputs = depth + 1

	# Define column structure (no persona columns)
	base_columns = [
		"topic",
		"stance",
		"seed",
		"timestamp",
		"response_index",
		"num_controller_outputs",
		"runtime_seconds",
		"final_argument",
	]

	step_columns = []
	for step in range(1, max_controller_outputs + 1):
		step_columns.extend(
			[
				f"step_{step}_reasoning",
				f"step_{step}_action",
				f"step_{step}_structure",
				f"step_{step}_subtopic",
				f"step_{step}_style",
				f"step_{step}_internal_reasoning",
				f"step_{step}_prefix",
				f"step_{step}_considerations",
			]
		)

	all_columns = base_columns + step_columns

	# Extract all leaf nodes (complete arguments) from the tree
	leaf_nodes = [
		node for node in tot_output.tree.nodes.values() if not node.children_ids
	]

	# Extract data for each leaf node (each complete argument)
	for leaf_idx, leaf_node in enumerate(leaf_nodes):
		# Get final argument and trajectory
		final_argument = leaf_node.state.output.get("argument", "")

		# Skip if no final argument
		if not final_argument or final_argument.strip() == "":
			continue

		trajectory = leaf_node.state.controller_output_trajectory

		# Extract reasoning steps (state.reasoning is already list[dict[str, str]])
		existing_steps = leaf_node.state.reasoning

		# Build base row (no persona fields)
		row = {
			"topic": topic,
			"stance": stance,
			"seed": seed,
			"timestamp": timestamp,
			"response_index": leaf_idx,
			"final_argument": final_argument,
			"num_controller_outputs": len(trajectory),
			"runtime_seconds": tot_output.runtime,
		}

		# Check if trajectory exceeds expected depth + 1
		if len(trajectory) > max_controller_outputs:
			logger.warning(
				f"Trajectory has {len(trajectory)} controller outputs but expected {max_controller_outputs} (depth={depth}+1). "
				f"Only first {max_controller_outputs} steps will be exported."
			)

		# Extract step-by-step data (up to depth + 1)
		for step_idx, controller_output in enumerate(
			trajectory[:max_controller_outputs], start=1
		):
			# Reasoning text
			if step_idx - 1 < len(existing_steps):
				reasoning_step = existing_steps[step_idx - 1]
				# Extract 'claim' field (the reasoning field name from signature)
				row[f"step_{step_idx}_reasoning"] = reasoning_step.get("claim", "")
			else:
				row[f"step_{step_idx}_reasoning"] = ""

			# Controller action and arguments
			# With JSON-based action spaces, the action format is "structure:subtopic"
			# Parse the action to extract structure and subtopic
			action_str = controller_output.action
			if ":" in action_str:
				structure, subtopic = action_str.split(":", 1)
				row[f"step_{step_idx}_structure"] = structure
				row[f"step_{step_idx}_subtopic"] = subtopic
			else:
				# Fallback: action doesn't have expected format
				row[f"step_{step_idx}_structure"] = action_str
				row[f"step_{step_idx}_subtopic"] = ""

			row[f"step_{step_idx}_action"] = action_str
			row[f"step_{step_idx}_style"] = ""  # Not used in this experiment

			# Controller guidance
			row[f"step_{step_idx}_internal_reasoning"] = (
				controller_output.internal_reasoning
			)
			row[f"step_{step_idx}_prefix"] = controller_output.prefix
			row[f"step_{step_idx}_considerations"] = controller_output.considerations

		# Pad remaining steps with empty strings
		for step_idx in range(
			min(len(trajectory), max_controller_outputs) + 1, max_controller_outputs + 1
		):
			row[f"step_{step_idx}_reasoning"] = ""
			row[f"step_{step_idx}_action"] = ""
			row[f"step_{step_idx}_structure"] = ""
			row[f"step_{step_idx}_subtopic"] = ""
			row[f"step_{step_idx}_style"] = ""
			row[f"step_{step_idx}_internal_reasoning"] = ""
			row[f"step_{step_idx}_prefix"] = ""
			row[f"step_{step_idx}_considerations"] = ""

		# Write to CSV (append mode)
		try:
			file_exists = os.path.exists(csv_file_path)
			with open(csv_file_path, "a", newline="", encoding="utf-8") as f:
				writer = csv.DictWriter(f, fieldnames=all_columns)
				if not file_exists:
					writer.writeheader()
				writer.writerow(row)
		except OSError as e:
			logger.error(f"Failed to write to CSV {csv_file_path}: {e}")


def main():
	args = base_parser.parse_args()
	run_generation(args)


if __name__ == "__main__":
	main()
