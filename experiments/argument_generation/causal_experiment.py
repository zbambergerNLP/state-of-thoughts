"""
Causal Argument Generation Experiment

This script runs a two-step experiment:
1. Generation: Generates argument trees for given topics and stances using Tree of Thoughts.
2. Evaluation: Evaluates the generated arguments using an LLM-as-a-Judge and identifies causal patterns.

Usage:
    python experiments/argument_generation/causal_experiment.py --mode generation [args...]
    python experiments/argument_generation/causal_experiment.py --mode evaluation [args...]

Example usage for generation:

```bash
python experiments/argument_generation/causal_experiment.py \
--mode generation \
--model Qwen3-30B-A3B-Instruct-2507 \
--generative_gpu_index 0 \
--outputs_directory experiments/argument_generation/causal_experiment/ \
--outputs_filename causal_exp \
--depth 2 \
--n_samples_generation 3 \
--top_k 2 \
--n_samples_judge 1 \
--experiment_mode synthesis_strict \
--do_save_tree \
--seed 42 \
--early_stopping_enabled
```

"""

# Standard library imports
import json
import logging
import os
from argparse import Namespace
from collections import Counter
from typing import Any, Literal

# Third-party imports
import dspy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

# Local imports
from adapter.constraints import GranularityType, ResponseLength
from constants import Verbosity
from experiments.argument_generation.flags import get_openai_api_key
from experiments.argument_generation.flags import parser as base_parser
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.scoring_local_lm import ScoringLocalVLLM
from predict.tree_of_thoughts import TreeOfThoughts
from predict.tree_of_thoughts.tree_parameters import TreeOfThoughtsParameters
from signatures.example_signatures import GenerateArgumentWithReasoning

load_dotenv()

# Set up logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Topics
TOPICS = [
	"The government should implement a Universal Basic Income (UBI) for all citizens.",
	"Employees should have the legal right to work remotely if their job allows it.",
	"The government should enforce a total ban on single-use plastics.",
	"Standardized testing should be abolished as a primary measure of student performance.",
	"The government should invest heavily in nuclear energy as a primary power source.",
	"Access to social media should be restricted to individuals over the age of 16.",
	"A special tax should be imposed on meat products to reduce consumption and environmental impact.",
	"The government should phase out physical currency in favor of a fully digital payment system.",
	"Public funding for space exploration should be significantly increased.",
	"Voting in national elections should be mandatory for all eligible citizens.",
]


def _extract_controller_decisions_from_chain(chain: Any) -> list[dict[str, Any]]:
	"""Extract controller decisions from a serialized ToT reasoning chain.

	We follow the same source of truth as NoveltyBench: the accumulated
	`state.controller_output_trajectory` on the final node of the chain.

	Args:
		chain: A serialized chain. This may be:
			- a list of node dicts (as produced by `node.model_dump()`), or
			- a dict with a `nodes` key, or
			- an opaque value (in which case we return []).

	Returns:
		List of controller decision dictionaries.
	"""
	nodes: list[Any]
	if isinstance(chain, dict) and "nodes" in chain:
		nodes = chain.get("nodes") or []
	elif isinstance(chain, list):
		nodes = chain
	else:
		return []

	if not nodes:
		return []

	final_node = nodes[-1]
	if not isinstance(final_node, dict):
		return []

	state = final_node.get("state", {})
	if not isinstance(state, dict):
		return []

	trajectory = state.get("controller_output_trajectory") or []
	if not isinstance(trajectory, list):
		return []

	decisions: list[dict[str, Any]] = []
	for decision in trajectory:
		if isinstance(decision, dict):
			decisions.append(decision)
	return decisions


def _summarize_controller_actions(decisions: list[dict[str, Any]]) -> dict[str, int]:
	"""Create a compact action-frequency summary from controller decisions."""
	labels: list[str] = []
	for d in decisions:
		action = d.get("action")
		action_arguments = d.get("action_arguments")
		if isinstance(action, str) and isinstance(action_arguments, dict):
			# Prefer a stable label that includes the chosen value if obvious.
			if "choice" in action_arguments:
				labels.append(f"{action}:{action_arguments['choice']}")
			elif len(action_arguments) == 1:
				(_, v) = next(iter(action_arguments.items()))
				labels.append(f"{action}:{v}")
			else:
				labels.append(action)
		elif isinstance(action, str):
			labels.append(action)

	return dict(Counter(labels))


class SingleTurnScoreSignature(dspy.Signature):
	"""
	Evaluate the quality and persuasiveness of an argument that takes a specific stance towards a given topic.
	"""

	topic: str = dspy.InputField(
		desc="A claim or statement that is the subject of the argument."
	)
	stance: Literal["PRO", "ANTI"] = dspy.InputField(
		prefix="""
The stance (towards the topic) of the argument that you are judging.
If the argument is supposed to advocate for the topic, then the stance is "PRO".
Conversely, if the argument is meant to oppose the topic, then the stance is "ANTI".
If the argument is supposed to take a certain stance, but fails to do so in practice, then the argument is poor.
""".strip(),
	)
	argument: str = dspy.InputField(
		desc="The argument that you are evaluating.",
	)
	relevance_and_significance: float = dspy.OutputField(
		ge=0.0,
		le=1.0,
		desc="""
The score of the argument based on its relevance and significance.
When evaluating relevance and significance, consider the following questions:
- How significant is the central claim the argument is trying to establish?
- How relevant is the claim to the topic?
""".strip(),
	)
	logical_strength: float = dspy.OutputField(
		ge=0.0,
		le=1.0,
		desc="""
The score of the argument based on its logical strength.
When evaluating logical strength, consider the following questions:
- How strong and effective is the data used to support the central claim?
- Does the argument's reasoning effectively guide the audience from their initial stance to the conclusion?
- Do all the claims logically support the main point?
""".strip(),
	)
	clarity_and_concreteness: float = dspy.OutputField(
		ge=0.0,
		le=1.0,
		desc="""
The score of the argument based on its clarity and concreteness.
When evaluating clarity and concreteness, consider the following questions:
- How illustrative and clear is the argument? Does it avoid ambiguity and jargon?
- Are examples used effectively and easy to understand?
- How tangible are the claims? Are they relatable to the audience?
""".strip(),
	)
	emotional_appeal: float = dspy.OutputField(
		ge=0.0,
		le=1.0,
		desc="""
The score of the argument based on its emotional appeal.
When evaluating emotional appeal, consider the following questions:
- How well does the argument resonate emotionally with the audience?
- To what extent does the argument engage the audience's feelings and values?
""".strip(),
	)
	credibility_and_authority: float = dspy.OutputField(
		ge=0.0,
		le=1.0,
		desc="""
The score of the argument based on its credibility and authority.
When evaluating credibility and authority, consider the following questions:
- Do you trust the person who wrote the argument? Do you consider them credible on the topic?
- Is the person who wrote the argument being fair and honest in their attempt to persuade?
- Are the sources referenced in the argument relevant, credible, and persuasive in and of themselves?
""".strip(),
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


def run_generation(args: Namespace):
	"""Run argument generation for all topics and stances."""
	generative_lm, reranker_lm = initialize_models(args)

	try:
		# `experiments/flags.py` allows multiple modes; this experiment uses the first one.
		final_output_kind = (
			args.experiment_mode[0] if isinstance(args.experiment_mode, list) else args.experiment_mode
		)

		# Define constraints
		thought_length = ResponseLength(
			granularity=GranularityType.SENTENCE,
			bounds=(1, 3),
		)
		response_length = ResponseLength(
			granularity=GranularityType.SENTENCE,
			bounds=(5, 7),
		)

		# Define action space paths for dynamic tool creation
		if args.action_space_paths:
			action_space_paths = args.action_space_paths
		else:
			action_space_dir = os.path.join(os.path.dirname(__file__), "action_space")
			action_space_paths = [
				os.path.join(action_space_dir, "causal_subtopics.json"),
				os.path.join(action_space_dir, "causal_styles.json"),
				os.path.join(action_space_dir, "causal_structures.json"),
			]

		logger.info(
			f"\nAction space paths:\n\t{action_space_paths}\n"
			f"thought length: {thought_length}\n"
			f"response length: {response_length}\n"
			f"max reasoning steps: {args.depth}\n"
			f"final output kind: {final_output_kind}\n"
			f"early stopping enabled: {args.early_stopping_enabled}\n"
			f"use self consistency: {args.use_self_consistency}\n"
			f"do pruning: {args.do_pruning}\n"
			f"use beam search: {args.controller_use_beam_search}\n"
			f"controller temperature: {args.controller_temperature}\n"
			f"generator temperature: {args.generator_temperature}\n"
			f"generator use beam search: {args.generator_use_beam_search}\n"
			f"num final candidates: {args.num_final_candidates}\n"
		)

		# Initialize TreeOfThoughts
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,
			evaluator_signature=None,
			generative_lm=generative_lm,
			reranker_lm=reranker_lm,
			controller_type="reranker",
			thought_length=thought_length,
			response_length=response_length,
			max_reasoning_steps=args.depth,
			final_output_kind=final_output_kind,
			early_stopping_enabled=args.early_stopping_enabled,
			action_space_paths=action_space_paths,
			seed=args.seed,
			verbosity=Verbosity.INFO,
		)

		tot_parameters = TreeOfThoughtsParameters(
			depth=args.depth,
			n_samples_generation=args.n_samples_generation,
			top_k=args.top_k,
			n_samples_judge=args.n_samples_judge,
			judge_temperature=args.judge_temperature,
			generator_temperature=args.generator_temperature,
			controller_temperature=args.controller_temperature,
			controller_use_beam_search=args.controller_use_beam_search,
			generator_temperature=args.generator_temperature,
			generator_use_beam_search=args.generator_use_beam_search,
			num_final_candidates=args.num_final_candidates,
			do_pruning=False,  # In causal experiments we don't use pruning
			use_self_consistency=args.use_self_consistency,
		)

		# Iterate over topics and stances
		stances = ["PRO", "ANTI"]

		# For testing purposes, limit if requested
		topics_to_run = TOPICS[:1] if args.test_run else TOPICS
		stances_to_run = stances[:1] if args.test_run else stances

		# Ensure output directory exists
		if args.do_save_tree and args.outputs_directory:
			os.makedirs(args.outputs_directory, exist_ok=True)

		for topic in topics_to_run:
			for stance in stances_to_run:
				logger.info(
					f"Generating argument for Topic: '{topic}', Stance: '{stance}'"
				)
				input_data = {"topic": topic, "stance": stance}
				# Construct filename based on topic and stance
				# Sanitize topic for filename
				topic_slug = "".join(c if c.isalnum() else "_" for c in topic[:30])
				filename = f"{args.outputs_filename}_{topic_slug.lower()}_{stance.lower()}.json"

				tot.forward(
					state=input_data,
					tot_parameters=tot_parameters,
					do_save_tree=args.do_save_tree,
					outputs_directory=args.outputs_directory,
					outputs_filename=filename,
				)
				logger.info(
					f"Completed Topic: '{topic}', Stance: '{stance}'. Output saved to {os.path.join(args.outputs_directory, filename)}"
				)

	finally:
		try:
			generative_lm.kill()
			reranker_lm.kill()
		except Exception as e:
			logger.warning(f"Error during cleanup: {e}")


def run_evaluation(args):
	"""Run evaluation on generated arguments."""
	# Check for OpenAI API key
	api_key = get_openai_api_key(args)
	if not api_key:
		logger.error(
			"OPENAI_API_KEY not found in environment, .env file, or --openai_api_key flag."
		)
		return

	# Set the API key in environment for dspy/openai to pick up if needed,
	# though dspy might need it configured directly if not using env var.
	os.environ["OPENAI_API_KEY"] = api_key
	model_name = args.judge_model
	logger.info(f"Using {model_name} as judge.")
	lm = dspy.LM(model=model_name)
	dspy.settings.configure(lm=lm)

	judge = dspy.Predict(SingleTurnScoreSignature)

	# Load generated trees
	results_data = []

	output_dir = args.outputs_directory
	if not os.path.exists(output_dir):
		logger.error(f"Output directory {output_dir} does not exist.")
		return

	# Find all result files
	# Assuming files are saved as json/jsonl in the directory
	# The TreeOfThoughts saves with specific suffixes, we need to find them.
	# Usually it saves as {filename}.json

	# We need to iterate through what we expect to have generated
	topics_to_run = TOPICS[:1] if args.test_run else TOPICS
	stances = ["PRO", "ANTI"]
	stances_to_run = stances[:1] if args.test_run else stances

	for topic in topics_to_run:
		for stance in stances_to_run:
			topic_slug = "".join(c if c.isalnum() else "_" for c in topic[:30])
			# Match the filename format from run_generation: lowercase and .json extension
			filename = (
				f"{args.outputs_filename}_{topic_slug.lower()}_{stance.lower()}.json"
			)

			# Construct expected file path
			file_path = os.path.join(output_dir, filename)

			if not os.path.exists(file_path):
				logger.warning(f"File not found: {file_path}")
				continue

			with open(file_path) as f:
				data = json.load(f)

			# Extract arguments and reasoning traces
			# The structure of 'data' depends on how ToT saves it.
			# When `do_save_tree=True`, TreeOfThoughts persists the `log_tree_run(...)` payload which
			# contains `tree_outputs` with `response_string` and `reasoning_chain`.
			if "tree_outputs" in data:
				tree_outputs = data.get("tree_outputs") or []
				for i, out in enumerate(tree_outputs, 1):
					if not isinstance(out, dict):
						continue
					arg = out.get("response_string")
					chain = out.get("reasoning_chain")
					if not isinstance(arg, str):
						continue
					logger.info(f"Evaluating argument {i} for {topic} ({stance})")

					try:
						score_result = judge(topic=topic, stance=stance, argument=arg)

						scores = {
							"relevance": score_result.relevance_and_significance,
							"logical": score_result.logical_strength,
							"clarity": score_result.clarity_and_concreteness,
							"emotional": score_result.emotional_appeal,
							"credibility": score_result.credibility_and_authority,
							"overall": (
								score_result.relevance_and_significance
								+ score_result.logical_strength
								+ score_result.clarity_and_concreteness
								+ score_result.emotional_appeal
								+ score_result.credibility_and_authority
							)
							/ 5.0,
						}

						controller_decisions = _extract_controller_decisions_from_chain(chain)
						controller_actions_count = _summarize_controller_actions(controller_decisions)

						results_data.append(
							{
								"topic": topic,
								"stance": stance,
								"argument": arg,
								**scores,
								"controller_decisions": controller_decisions,
								"controller_actions_count": controller_actions_count,
								"chain": chain,  # Keep raw chain for detailed analysis.
							}
						)
					except Exception as e:
						logger.error(f"Error evaluating argument: {e}")

			# Backward-compatibility: serialized TreeOfThoughtsOutput format.
			elif "response_strings" in data and "reasoning_steps" in data:
				response_strings = data["response_strings"]
				reasoning_steps = data["reasoning_steps"]

				for i, (arg, chain) in enumerate(zip(response_strings, reasoning_steps, strict=True), 1):
					if not isinstance(arg, str):
						continue
					logger.info(f"Evaluating argument {i} for {topic} ({stance})")

					try:
						score_result = judge(topic=topic, stance=stance, argument=arg)

						scores = {
							"relevance": score_result.relevance_and_significance,
							"logical": score_result.logical_strength,
							"clarity": score_result.clarity_and_concreteness,
							"emotional": score_result.emotional_appeal,
							"credibility": score_result.credibility_and_authority,
							"overall": (
								score_result.relevance_and_significance
								+ score_result.logical_strength
								+ score_result.clarity_and_concreteness
								+ score_result.emotional_appeal
								+ score_result.credibility_and_authority
							)
							/ 5.0,
						}

						controller_decisions = _extract_controller_decisions_from_chain(chain)
						controller_actions_count = _summarize_controller_actions(controller_decisions)

						results_data.append(
							{
								"topic": topic,
								"stance": stance,
								"argument": arg,
								**scores,
								"controller_decisions": controller_decisions,
								"controller_actions_count": controller_actions_count,
								"chain": chain,
							}
						)
					except Exception as e:
						logger.error(f"Error evaluating argument: {e}")
			else:
				logger.warning(
					f"Unrecognized ToT output format in file: {file_path}. "
					"Expected `tree_outputs` or (`response_strings` + `reasoning_steps`)."
				)

	# Save evaluation results
	eval_output_path = os.path.join(output_dir, "evaluation_results.csv")
	df = pd.DataFrame(results_data)
	df.to_csv(eval_output_path, index=False)
	logger.info(f"Evaluation results saved to {eval_output_path}")

	# Visualization
	if not df.empty:
		create_visualizations(df, output_dir)


def create_visualizations(df, output_dir):
	"""Create visualizations for the results."""
	# 1. Correlation Matrix of Scores
	plt.figure(figsize=(10, 8))
	score_cols = [
		"relevance",
		"logical",
		"clarity",
		"emotional",
		"credibility",
		"overall",
	]
	# Ensure columns exist
	existing_cols = [c for c in score_cols if c in df.columns]
	if len(existing_cols) > 1:
		corr = df[existing_cols].corr()
		sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
		plt.title("Correlation of Argument Quality Scores")
		plt.tight_layout()
		plt.savefig(os.path.join(output_dir, "score_correlation.png"))
		plt.close()

	# 2. Distribution of Overall Scores by Stance
	if "overall" in df.columns and "stance" in df.columns:
		plt.figure(figsize=(8, 6))
		sns.boxplot(x="stance", y="overall", data=df)
		plt.title("Overall Quality Scores by Stance")
		plt.tight_layout()
		plt.savefig(os.path.join(output_dir, "score_distribution_by_stance.png"))
		plt.close()

	# 3. (Optional) Causal Impact of Actions
	logger.info("Visualizations created.")


def main():
	args = base_parser.parse_args()
	if args.mode == "generation":
		run_generation(args)
	elif args.mode == "evaluation":
		run_evaluation(args)


if __name__ == "__main__":
	main()
