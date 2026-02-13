"""Flags for the dspy_reasoning project.

These flags are meant to support all three reasoning tasks for our experiments:
1. Argument Generation
2. Instruction Following

Task-specific flags are defined in the respective task-specific flags file (and extend
this base flags file).
"""

# Standard library imports
import argparse

# Local imports
import constants

# ToT experiment mode options
EXPERIMENT_MODE_CHOICES = [
	"synthesis_faithful",
	"synthesis_strict",
	"synthesis_restructured",
	"conclusion",
]

# Create the base argument parser
parser = argparse.ArgumentParser(
	description="Run experiments with vLLM language models and DSPy on GPU clusters.",
	formatter_class=argparse.RawDescriptionHelpFormatter,
)

# ==============================================================================
# Shared / Cross-experiment flags
# ==============================================================================
evaluation_group = parser.add_argument_group(
	title="Evaluation Configuration",
	description="Flags used across experiments for evaluation/postprocessing",
)
evaluation_group.add_argument(
	"--experiment_dir",
	type=str,
	default=None,
	help="Path to an experiment results directory to operate on.",
)
evaluation_group.add_argument(
	"--runs_to_evaluate",
	type=str,
	nargs="+",
	default=None,
	help=(
		"Optional list of specific run directories to evaluate (overrides scanning an entire "
		"experiment directory). Paths may be relative or absolute."
	),
)
evaluation_group.add_argument(
	"--skip_evaluation",
	action="store_true",
	default=False,
	help="Skip evaluation/scoring and only run non-eval postprocessing steps.",
)

# ==============================================================================
# Generation Configuration
# ==============================================================================
generation_group = parser.add_argument_group(
	title="Generation Configuration",
	description="Settings for text generation",
)
generation_group.add_argument(
	"--num_generations",
	type=int,
	default=1,
	help="Number of generations per prompt.",
)
generation_group.add_argument(
	"--max_tokens",
	type=int,
	default=8_000,
	help="Maximum number of tokens to generate per prompt (must be less than `max_model_len`).",
)
generation_group.add_argument(
	"--seed",
	type=int,
	default=42,
	help="Random seed for reproducibility.",
)
generation_group.add_argument(
	"--seeds",
	type=int,
	nargs="+",
	default=None,
	help="List of random seeds for sweeping (overrides --seed).",
)

# ============================================================================
# Dataset Configuration
# ============================================================================
dataset_group = parser.add_argument_group(
	title="Dataset Configuration",
	description="Settings for dataset loading and processing",
)
dataset_group.add_argument(
	"--split",
	type=str,
	default="train",
	choices=["train", "test", "debugging"],
	help="Dataset split to evaluate: 'train', 'test', or 'debugging' (loads a small subset for debugging).",
)
dataset_group.add_argument(
	"--dataset_path",
	type=str,
	default=None,
	help="Path to dataset file or directory.",
)
dataset_group.add_argument(
	"--output_dir",
	type=str,
	default=None,
	help="Output directory for results.",
)
dataset_group.add_argument(
	"--max_examples",
	type=int,
	default=None,
	help="Maximum number of examples to process.",
)

# ============================================================================
# vLLM Generator Configuration
# ============================================================================
vllm_gen_group = parser.add_argument_group(
	title="vLLM Generator Configuration",
	description="Settings for the primary generative model (vLLM backend)",
)
vllm_gen_group.add_argument(
	"--model",
	type=str,
	choices=[model.value for model in constants.OpenSourceModel],
	default=constants.OpenSourceModel.QWEN_3_30B_A3B_INSTRUCT_2507.value,
	help="The HuggingFace model name to use for generation.",
)
vllm_gen_group.add_argument(
	"--model_directory",
	type=str,
	help="The directory where the generative model is stored.",
)
vllm_gen_group.add_argument(
	"--model_provider",
	type=str,
	choices=[provider.value for provider in constants.ModelProvider],
	default=constants.ModelProvider.QWEN.value,
	help="The model provider to use for the generative model.",
)
vllm_gen_group.add_argument(
	"--baseline_temperature",
	type=float,
	default=0.7,
	help="Sampling temperature for generation.",
)
vllm_gen_group.add_argument(
	"--generator_gpu_memory_utilization",
	type=float,
	default=0.9,
	help="GPU memory utilization for generator (0.0-1.0).",
)
vllm_gen_group.add_argument(
	"--generator_tensor_parallel_size",
	type=int,
	default=1,
	help="Number of GPUs for tensor parallelism (generator).",
)
vllm_gen_group.add_argument(
	"--generator_dtype",
	type=str,
	default="auto",
	help="Data type for generator weights ('auto', 'bfloat16', 'float16').",
)
vllm_gen_group.add_argument(
	"--generator_max_model_len",
	type=int,
	default=16_384,
	help="Maximum model length for the generator.",
)
vllm_gen_group.add_argument(
	"--generator_enforce_eager",
	action="store_true",
	default=False,
	help="Enable eager execution mode for generator (default: disabled).",
)
vllm_gen_group.add_argument(
	"--generative_gpu_index",
	type=str,
	default="0",
	help="GPU index for the generative model.",
)
vllm_gen_group.add_argument(
	"--gpu",
	type=str,
	default=constants.GPU.A100_80_GB.value,
	choices=[gpu.value for gpu in constants.GPU],
	help="The GPU type to use for the model.",
)
vllm_gen_group.add_argument(
	"--hf_token",
	type=str,
	default=None,
	required=False,
	help="The HuggingFace token to use for authentication.",
)
vllm_gen_group.add_argument(
	"--trust_remote_code",
	action="store_true",
	help="Whether to trust remote code when loading the model.",
)

# ============================================================================
# vLLM Reranker Configuration
# ============================================================================
vllm_reranker_group = parser.add_argument_group(
	title="vLLM Reranker Configuration",
	description="Settings for the reranker model used in Tree-of-Thought (vLLM backend)",
)
vllm_reranker_group.add_argument(
	"--reranker_model",
	type=str,
	default=constants.OpenSourceModel.QWEN_3_RERANKER_8B.value,
	help="The HuggingFace model name to use for reranking.",
)
vllm_reranker_group.add_argument(
	"--reranker_model_directory",
	type=str,
	help="The directory where the reranker model is stored.",
)
vllm_reranker_group.add_argument(
	"--reranker_model_provider",
	type=str,
	choices=[provider.value for provider in constants.ModelProvider],
	default=constants.ModelProvider.QWEN.value,
	help="The model provider to use for the reranker model.",
)
vllm_reranker_group.add_argument(
	"--reranker_model_path",
	type=str,
	default=None,
	help="Reranker model path (overrides --reranker_model if provided).",
)
vllm_reranker_group.add_argument(
	"--reranker_gpu_memory_utilization",
	type=float,
	default=0.9,
	help="GPU memory utilization for reranker (0.0-1.0).",
)
vllm_reranker_group.add_argument(
	"--reranker_tensor_parallel_size",
	type=int,
	default=1,
	help="Number of GPUs for tensor parallelism (reranker).",
)
vllm_reranker_group.add_argument(
	"--reranker_dtype",
	type=str,
	default="auto",
	help="Data type for reranker weights ('auto', 'bfloat16', 'float16').",
)
vllm_reranker_group.add_argument(
	"--reranker_max_model_len",
	type=int,
	default=16_384,
	help="Maximum model length for the reranker.",
)
vllm_reranker_group.add_argument(
	"--reranker_enforce_eager",
	action="store_true",
	default=False,
	help="Enable eager execution mode for reranker.",
)
vllm_reranker_group.add_argument(
	"--reranker_gpu_index",
	type=str,
	default="1",
	help="GPU index for the reranker model.",
)

# ============================================================================
# Baseline Configuration
# ============================================================================
baseline_group = parser.add_argument_group(
	title="Baseline Configuration",
	description="Settings for baseline (non-ToT) inference",
)

# ============================================================================
# Tree-of-Thought Configuration
# ============================================================================
tot_group = parser.add_argument_group(
	title="Tree-of-Thought Configuration",
	description="Settings for Tree-of-Thought inference",
)
tot_group.add_argument(
	"--action_space_paths",
	type=str,
	nargs="+",
	default=None,
	help="List of paths to action space JSON files.",
)
tot_group.add_argument(
	"--action_space_name",
	type=str,
	default=None,
	choices=["controlled", "uncontrolled"],
	help="""
Name/identifier for the action space configuration.
If uncontrolled is selected, we use a 'baseline' ToT (i.e., controller is not used at all).
If controlled is selected, we use the action space JSON files to guide generation.
""",
)
tot_group.add_argument(
	"--tot_verbosity",
	type=str,
	default=None,
	choices=["DEBUG", "INFO", "WARNING", "ERROR"],
	help="Verbosity level for Tree-of-Thought logging.",
)

tot_group.add_argument(
	"--depth",
	type=int,
	default=2,
	help="Tree depth.",
)
tot_group.add_argument(
	"--n_samples_generation",
	type=int,
	default=3,
	help="Number of samples for generation.",
)
tot_group.add_argument(
	"--top_k",
	type=int,
	default=2,
	help="Top k candidates to keep.",
)
tot_group.add_argument(
	"--top_k_first",
	type=int,
	default=None,
	help="Override top_k for the first layer only. If None, uses top_k.",
)
tot_group.add_argument(
	"--n_samples_judge",
	type=int,
	default=1,
	help="Number of samples for judge.",
)
tot_group.add_argument(
	"--controller_type",
	type=str,
	default="reranker",
	choices=["generator", "reranker"],
	help="Type of controller to use.",
)
tot_group.add_argument(
	"--evaluator_type",
	type=str,
	default="generator",
	choices=["generator", "reranker", "programmatic"],
	help=(
		"Type of evaluator to use. 'generator' uses a generative judge; 'reranker' uses a "
		"scoring/reranker model (deterministic); 'programmatic' uses Python scoring functions."
	),
)
tot_group.add_argument(
	"--num_final_candidates",
	type=int,
	default=1,
	help="Number of final candidates.",
)
tot_group.add_argument(
	"--n_final_responses_per_trajectory",
	type=int,
	default=1,
	help=(
		"Number of final responses to generate per trajectory "
		"(default: same as n_samples_generation)."
	),
)
tot_group.add_argument(
	"--do_pruning",
	action="store_true",
	default=False,
	help="Enable pruning -- selecting the top-k candidates from each layer of the tree (default: disabled)",
)
tot_group.add_argument(
	"--use_self_consistency",
	action="store_true",
	default=False,
	help="Enable self consistency (majority voting for final selection). NOTE: Works best for tasks that involve generating short/discrete responses.",
)
tot_group.add_argument(
	"--experiment_mode",
	type=str,
	choices=EXPERIMENT_MODE_CHOICES,
	default="synthesis_faithful",
	help=(
		f"The method for generating a final output given a trajectory of reasoning steps.\n"
		f"Must be one of {', '.join(EXPERIMENT_MODE_CHOICES)}."
	),
)
tot_group.add_argument(
	"--early_stopping_enabled",
	action="store_true",
	default=False,
	help="Enable early stopping in the Tree-of-Thought search. Enabling this will introduce a 'FINISH' action in the action space for the controller to use.",
)
tot_group.add_argument(
	"--forced_choices_json",
	type=str,
	default=None,
	help=(
		"Path to JSON file containing forced controller choices. "
		"Format: dict mapping string keys like \"(0, 0)\" to lists of action dicts."
	),
)

# ============================================================================
# Component Sampling Configuration (Tree-of-Thought only)
# ============================================================================
generator_sampling_group = parser.add_argument_group(
	title="Generator Sampling Configuration",
	description="Sampling parameters for the Tree-of-Thought generator component.",
)
generator_sampling_group.add_argument(
	"--generator_temperature",
	type=float,
	default=None,
	help="Temperature for the generator.",
)
generator_sampling_group.add_argument(
	"--generator_max_tokens",
	type=int,
	default=8_000,
	help=(
		"Maximum number of tokens for generator completions in Tree-of-Thought. "
		"If unset, defaults to 8_000."
	),
)
generator_sampling_group.add_argument(
	"--generator_use_beam_search",
	action="store_true",
	default=False,
	help="Enable beam search for generator.",
)
generator_sampling_group.add_argument(
	"--generator_top_p",
	type=float,
	default=1.0,
	help="vLLM nucleus sampling top_p for the generator (0, 1]",
)
generator_sampling_group.add_argument(
	"--generator_top_k",
	type=int,
	default=-1,
	help="vLLM sampling top_k for the generator (int, can be -1 for all tokens)",
)
generator_sampling_group.add_argument(
	"--generator_min_p",
	type=float,
	default=0.0,
	help="vLLM min_p for the generator (>=0)",
)

controller_sampling_group = parser.add_argument_group(
	title="Controller Sampling Configuration",
	description="Sampling parameters for the Tree-of-Thought controller component.",
)
controller_sampling_group.add_argument(
	"--controller_temperature",
	type=float,
	default=None,
	help="Temperature for the controller. If unset, defaults to `generator_temperature`.",
)
controller_sampling_group.add_argument(
	"--controller_max_tokens",
	type=int,
	default=8_000,
	help="Maximum number of tokens for controller completions. If unset, defaults to 8_000.",
)
controller_sampling_group.add_argument(
	"--controller_max_model_len",
	type=int,
	default=8_192,
	help="Maximum model length for the controller.",
)
controller_sampling_group.add_argument(
	"--controller_use_beam_search",
	action="store_true",
	default=False,
	help="Enable beam search for controller.",
)
controller_sampling_group.add_argument(
	"--controller_top_p",
	type=float,
	default=1.0,
	help="vLLM nucleus sampling top_p for controller (0, 1]",
)
controller_sampling_group.add_argument(
	"--controller_top_k",
	type=int,
	default=-1,
	help="vLLM sampling top_k for controller (int, can be -1 for all tokens)",
)
controller_sampling_group.add_argument(
	"--controller_min_p",
	type=float,
	default=0.0,
	help="vLLM min_p for controller (>=0)",
)

evaluator_sampling_group = parser.add_argument_group(
	title="Evaluator Sampling Configuration",
	description="Sampling parameters for the Tree-of-Thought evaluator (judge) component.",
)
evaluator_sampling_group.add_argument(
	"--judge_temperature",
	type=float,
	default=0.3,
	help="Temperature for judge.",
)
evaluator_sampling_group.add_argument(
	"--judge_max_tokens",
	type=int,
	default=8_000,
	help="Maximum number of tokens for evaluator/judge completions. If unset, defaults to 8_000.",
)
evaluator_sampling_group.add_argument(
	"--judge_max_model_len",
	type=int,
	default=8_192,
	help="Maximum model length for the judge/evaluator.",
)
evaluator_sampling_group.add_argument(
	"--judge_top_p",
	type=float,
	default=1.0,
	help="vLLM nucleus sampling top_p for judge (0, 1]",
)
evaluator_sampling_group.add_argument(
	"--judge_top_k",
	type=int,
	default=-1,
	help="vLLM sampling top_k for judge (int, can be -1 for all tokens)",
)
evaluator_sampling_group.add_argument(
	"--judge_min_p",
	type=float,
	default=0.0,
	help="vLLM min_p for judge (>=0)",
)
evaluator_sampling_group.add_argument(
	"--judge_use_beam_search",
	action="store_true",
	default=False,
	help="Enable beam search for the judge/evaluator.",
)

# ============================================================================
# Logging and Output
# ============================================================================
logging_group = parser.add_argument_group(
	title="Logging and Output",
	description="Settings for logging verbosity and tree output",
)
logging_group.add_argument(
	"--verbosity",
	type=str,
	choices=[
		constants.Verbosity.DEBUG,
		constants.Verbosity.INFO,
		constants.Verbosity.WARNING,
		constants.Verbosity.ERROR,
	],
	default=constants.Verbosity.INFO,
	help="Logging verbosity level.",
)
logging_group.add_argument(
	"--do_save_tree",
	action="store_true",
	default=False,
	help="Save the full tree structure to disk after the run completes.",
)
logging_group.add_argument(
	"--outputs_directory",
	type=str,
	default=None,
	help="Directory path for saving tree outputs.",
)
logging_group.add_argument(
	"--outputs_filename",
	type=str,
	default=None,
	help="Filename for saved tree outputs.",
)
logging_group.add_argument(
	"--results_split",
	type=str,
	default=None,
	help=(
		"Optional override for the results directory split label. If not provided, uses --split. "
		"Useful for small/debug runs (e.g. --split train --results_split debugging)."
	),
)
