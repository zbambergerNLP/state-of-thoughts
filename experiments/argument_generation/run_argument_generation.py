"""Simple test script for Tree of Thoughts argument generation.

This script initializes a language model and performs a full inference run
with tree of thoughts for argument generation.

NOTE: This script requires two separate GPUs to run by default.
    - The first GPU is used for the generative model (which serves the generator, evaluator, and
        [optionally] the generative controller modules).
    - The second GPU is used for the reranker model (which scores the [optional] controller
        actions).

Example usage:

```bash
python experiments/argument_generation/run_argument_generation.py \
--experiment_mode synthesis_faithful \
--do_pruning \
--do_save_tree \
--outputs_directory ./experiments/argument_generation/tot_outputs \
--outputs_filename argument_generation_depth_3_bf_5_top_k_2 \
--depth 3 \
--generator_temperature 0.7 \
--n_samples_generation 5 \
--top_k 3 \
--n_samples_judge 5 \
--judge_temperature 0.7 \
--action_space_paths \
  ./experiments/argument_generation/action_space/causal_subtopics.json \
  ./experiments/argument_generation/action_space/causal_styles.json \
  ./experiments/argument_generation/action_space/causal_structures.json
```

Possible modes:
- synthesis_strict: Very faithful to the reasoning steps.
- synthesis_faithful: Faithful to the ideas and reasoning steps, but allows light rephrasing.
- synthesis_restructured: Maintains the same broad ideas and reasoning, but allows rephrasing and restructuring for clarity and coherence.
- conclusion: Allows the model to produce the best possible final answer based on the reasoning. Prioritizes clarity and quality over strict faithfulness to structure or phrasing.
"""

# Standard library imports
import logging
import os
import sys
from argparse import Namespace

# Third-party imports
import dspy

# Local imports
from adapter.constraints import ResponseLength
from constants import Verbosity
from experiments.argument_generation.flags import (
	parser,  # extension of `experiments/flags.py`
)
from experiments.shared_utils import suppress_vllm_logging
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.scoring_local_lm import ScoringLocalVLLM
from predict.tree_of_thoughts import TreeOfThoughts
from predict.tree_of_thoughts.tree_parameters import TreeOfThoughtsParameters
from signatures.example_signatures import GenerateArgumentWithReasoning

# Set up logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Enable INFO logging for tree_of_thoughts to see intermediate steps
logging.getLogger("predict.tree_of_thoughts.tree_of_thoughts").setLevel(logging.INFO)

# Suppress vLLM logs (can include very verbose token / engine output).
suppress_vllm_logging()


def initialize_models(args: Namespace):
	"""Initialize two vLLM language models on separate GPUs.

	Args:
	    args (Namespace): Parsed command line arguments.

	Returns:
	    tuple: (generative_lm, reranker_lm) - Two LM instances on different GPUs.
	"""
	# Model configuration
	generative_model_name = args.model
	generative_full_model_path = os.path.join(
		args.model_directory, generative_model_name
	)
	reranker_model_name = args.reranker_model
	reranker_full_model_path = os.path.join(args.model_directory, reranker_model_name)

	# Initialize generative model on specified GPU
	# Set CUDA_VISIBLE_DEVICES to only show the generative GPU during initialization
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
	# This ensures both models can access their respective GPUs during runtime
	os.environ["CUDA_VISIBLE_DEVICES"] = (
		f"{args.generative_gpu_index},{args.reranker_gpu_index}"
	)
	logger.info("Both GPUs are now visible for runtime operations")

	# Set default LM for dspy settings (use generative model)
	dspy.settings.configure(lm=generative_lm)
	return generative_lm, reranker_lm


def run_argument_generation():
	"""Run tree of thoughts for argument generation."""
	# Parse arguments
	args = parser.parse_args()

	# Initialize models once (on separate GPUs)
	generative_lm, reranker_lm = initialize_models(args)

	# `experiments/flags.py` defines `--experiment_mode` with `nargs="+"`, so it parses as a list.
	# Argument generation runs a single mode; if multiple are provided, we take the first.
	final_output_kind = (
		args.experiment_mode[0]
		if isinstance(args.experiment_mode, list)
		else args.experiment_mode
	)

	try:
		logger.info("\n\n" + "#" * 80)
		logger.info(f"EXPERIMENT: {final_output_kind.upper()}")
		logger.info("#" * 80)
		logger.info("\n" + "=" * 80)
		logger.info(
			f"RUNNING EXPERIMENT WITH FINAL_OUTPUT_KIND: {final_output_kind.upper()}"
		)
		logger.info("=" * 80)

		# Define constraints
		thought_length = ResponseLength(
			granularity="sentence",
			bounds=(1, 3),
		)
		response_length = ResponseLength(
			granularity="sentence",
			bounds=(5, 7),
		)

		# Initialize TreeOfThoughts with specified final_output_kind and reranker controller
		logger.info(
			f"Initializing TreeOfThoughts module with final_output_kind={final_output_kind}..."
		)
		logger.info(
			"Using reranker controller with separate generative and reranker models"
		)
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,
			evaluator_signature=None,  # Use default evaluator
			generative_lm=generative_lm,
			reranker_lm=reranker_lm,
			controller_type="reranker",
			thought_length=thought_length,
			response_length=response_length,
			max_reasoning_steps=args.depth,
			final_output_kind=final_output_kind,
			early_stopping_enabled=args.early_stopping_enabled,
			action_space_paths=args.action_space_paths,
			consider_reasoning_in_final_eval=False,
			seed=args.seed,
			verbosity=Verbosity.INFO,
		)

		# Define input
		input_data = {
			"topic": "We should stop pretending NLP is a science.",
			"stance": "PRO",
		}

		# Define parameters from args
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
			do_pruning=args.do_pruning,
			use_self_consistency=args.use_self_consistency,
		)

		# Run inference
		logger.info(
			f"Running Tree of Thoughts inference with {final_output_kind}..."
		)
		logger.info(f"Input: {input_data}")
		logger.info(f"Parameters: {tot_parameters}")
		sys.stdout.flush()

		output = tot.forward(
			state=input_data,
			tot_parameters=tot_parameters,
			do_save_tree=args.do_save_tree,
			outputs_directory=args.outputs_directory,
			outputs_filename=args.outputs_filename,
		)
		sys.stdout.flush()

		logger.info(
			f"Tree of Thoughts inference completed successfully ({final_output_kind})"
		)
		logger.info(f"Output received: {type(output)}")
		logger.info(f"Output has {len(output.responses)} responses")
		logger.info(f"Output has {len(output.response_strings)} response strings")
		sys.stdout.flush()

		# Display results
		logger.info("\n" + "=" * 80)
		logger.info(f"RESULTS ({final_output_kind.upper()})")
		logger.info("=" * 80)
		logger.info(f"Runtime: {output.runtime:.2f} seconds")
		logger.info(f"Number of responses: {len(output.responses)}")
		logger.info(f"Tree depth: {len(output.tree.layers) - 1} layers")
		logger.info("\n" + "-" * 80)

		if len(output.response_strings) > 0:
			for i, (reasoning_chain, response_str) in enumerate(
				zip(output.reasoning_steps, output.response_strings, strict=True), 1
			):
				logger.info(f"\nFinal Argument {i} ({final_output_kind}):")
				logger.info("-" * 80)
				logger.info(response_str)
				logger.info("-" * 80)
				logger.info(f"\nReasoning chain ({len(reasoning_chain.nodes)} nodes):")
				logger.info(str(reasoning_chain))
		else:
			logger.warning("No final arguments found in output")
			logger.info(f"Responses: {[r.to_dict() for r in output.responses]}")
			logger.info(f"Response strings: {output.response_strings}")
			logger.info(f"Reasoning chains: {len(output.reasoning_steps)} chains")

		logger.info("\n" + "=" * 80)
		sys.stdout.flush()

		# Force flush all handlers to ensure output is visible
		for handler in logging.root.handlers:
			if hasattr(handler, "flush"):
				handler.flush()

	except Exception as e:
		logger.error(
			f"Error during Tree of Thoughts inference ({final_output_kind}): {e}",
			exc_info=True,
		)
		sys.stdout.flush()
		raise

	finally:
		# Cleanup - wrap in try/except to handle vLLM engine crashes gracefully
		try:
			logger.info("\nCleaning up...")
			sys.stdout.flush()
			logger.info("Killing generative model...")
			generative_lm.kill()
			logger.info("Killing reranker model...")
			reranker_lm.kill()
			logger.info("Cleanup completed")
			sys.stdout.flush()
		except Exception as e:
			logger.warning(
				f"Error during cleanup (may be due to vLLM engine crash): {e}"
			)
			sys.stdout.flush()


if __name__ == "__main__":
	run_argument_generation()
