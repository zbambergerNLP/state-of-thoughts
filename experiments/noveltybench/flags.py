"""
NoveltyBench-specific command-line flags and Tree-of-Thought preset configurations.

Extends the base flags from experiments.flags with NoveltyBench-specific arguments.
"""

# Standard library imports
import copy
from pathlib import Path

# Local imports
from experiments import flags

ACTION_SPACE_DIR = Path(__file__).parent / "action_space"

# =============================================================================
# NoveltyBench Parser Configuration
# =============================================================================

# Create NoveltyBench parser by extending the base parser
noveltybench_parser = copy.deepcopy(flags.parser)
noveltybench_parser.description = "NoveltyBench evaluation with vLLM and Tree-of-Thought support"

# Override defaults for NoveltyBench
noveltybench_parser.set_defaults(
	num_generations=10,
	generator_max_model_len=16_384,
	reranker_max_model_len=16_384,
	depth=1,
	n_samples_generation=10,
	top_k=1,
    num_final_candidates=10,
    n_final_responses_per_trajectory=1,
	early_stopping_enabled=False,
	tot_verbosity="WARNING",
	experiment_mode="synthesis_faithful",
)

# ==============================================================================
# NoveltyBench-Specific Flags
# ==============================================================================

# Evaluation-only configuration
noveltybench_parser.add_argument(
	"--eval_only",
	action="store_true",
	help=(
		"Only run evaluation/postprocessing for an existing results directory (no inference). "
		"Useful when generations.json files already exist."
	),
)
noveltybench_parser.add_argument(
	"--plot_only",
	action="store_true",
	help=(
		"Only generate plots and summary tables, skipping evaluation (partition + score). "
		"Useful when data is already partitioned and scored."
	),
)
noveltybench_parser.add_argument(
	"--eval_results_dir",
	type=str,
	default=None,
	help=(
		"Path to an existing NoveltyBench results directory to evaluate. "
		"Can be the full experiment root (e.g. experiments/results/noveltybench_curated_test) "
		"or a model subdirectory. If omitted, defaults to "
		"experiments/results/noveltybench_{subset}_{split}."
	),
)
noveltybench_parser.add_argument(
	"--partition_workers",
	type=int,
	default=1,
	help=(
		"Number of parallel workers for partitioning during evaluation-only runs. "
		"Safe values are typically 2-8 depending on CPU/GPU availability."
	),
)

# Dataset configuration
noveltybench_parser.add_argument(
	"--subset",
	type=str,
	choices=["curated", "wildchat"],
	default="curated",
	help="Dataset subset to evaluate (default: curated)",
)

# Partitioning configuration
noveltybench_parser.add_argument(
	"--partition_algorithm",
	type=str,
	default="classifier",
	choices=["classifier", "bertscore", "unigram"],
	help="""Partitioning algorithm to use for computing the diversity metric in noveltybench.
	Must be one of: 'classifier' (DeBERTa), 'bertscore' (BERTScore), or 'unigram' (ROUGE).
	Default: 'classifier'.
	""",
)

# Baseline Configuration
noveltybench_parser.add_argument(
	"--baseline_signature",
	type=str,
	default=None,
	choices=[
		"InstructionFollowing",
		"InstructionFollowingCoT",
		"InstructionFollowingWithTools",
		"InstructionFollowingWithToolsCoT",
	],
	help="Signature class to use for baseline inference (default: None).",
)

# Tree-of-Thought specific configuration
noveltybench_parser.add_argument(
	"--tot_signature",
	type=str,
	default="InstructionFollowingWithReasoning",
	choices=[
		"InstructionFollowingWithReasoning",
		"InstructionFollowingWithReasoningAndTools",
	],
	help="Signature class to use for Tree-of-Thought inference.",
)

# Scoring configuration
noveltybench_parser.add_argument(
	"--skip_score",
	action="store_true",
	help="Skip scoring stage (requires 32GB+ RAM)",
)
noveltybench_parser.add_argument(
	"--score_cuda_device",
	type=str,
	default="auto",
	help="CUDA_VISIBLE_DEVICES override for scoring (default: auto).",
)
noveltybench_parser.add_argument(
	"--scorer_type",
	type=str,
	choices=["local", "openai"],
	default="local",
	help="Scorer type: 'local' (Skywork-Reward-Gemma) or 'openai' (embeddings)",
)
noveltybench_parser.add_argument(
	"--scorer_model",
	type=str,
	default="Skywork-Reward-Gemma-2-27B-v0.2",
	help="Scorer model name or path",
)
noveltybench_parser.add_argument(
	"--scorer_torch_dtype",
	type=str,
	default="bfloat16",
	choices=["bfloat16", "float16", "float32"],
	help="Torch dtype for the local scorer (default: bfloat16)",
)
noveltybench_parser.add_argument(
	"--scorer_device_map",
	type=str,
	default="auto",
	help="Device map for the local scorer (default: auto)",
)
noveltybench_parser.add_argument(
	"--scorer_attn_implementation",
	type=str,
	default="eager",
	choices=["eager", "sdpa"],
	help="Attention implementation for the local scorer (default: eager)",
)
noveltybench_parser.add_argument(
	"--embedding_model",
	type=str,
	default="text-embedding-3-large",
	help="OpenAI embedding model (used when scorer-type=openai)",
)
noveltybench_parser.add_argument(
	"--score_patience",
	type=float,
	default=0.8,
	help="Fraction of top clusters retained during scoring (default: 0.8)",
)

# Logging configuration
noveltybench_parser.add_argument(
	"--noveltybench_log_level",
	type=str,
	default="DEBUG",
	choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
	help="Log level for NoveltyBench scripts (default: DEBUG)",
)
