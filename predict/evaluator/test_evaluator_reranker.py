"""
Tests for the reranker evaluator module.

Expected usage:
```bash
pytest predict/test_evaluator_reranker.py -vv
```
"""

# Standard library imports
import logging
import os
from typing import Literal

# Third-party imports
import dspy
import pytest
import torch

# Local imports
from constants import OpenSourceModel
from lm.scoring_local_lm import ScoringLocalVLLM
from predict.evaluator.evaluator_reranker import TreeOfThoughtRerankerEvaluator
from signatures.example_signatures import (
	QuestionAnsweringWithReasoning,
	WeightedMultiDimensionRubric,
	WeightedPRMRubric,
)
from signatures.signature import ReasoningSignature
from tree import State
from utilities_for_tests import MockScoringLocalVLLM

logger = logging.getLogger(__name__)

# =============================================================================
# GPU Skip Markers
# =============================================================================

# Check if one or more GPUs are available
if torch.cuda.is_available():
	_has_gpu = True
else:
	_has_gpu = False

# Skip GPU tests if no GPU is available
pytestmark_gpu = pytest.mark.skipif(
	not _has_gpu,
	reason="GPU tests require GPU access",
)


def _make_state_prm(
	question: str = "What is 2+2?",
	reasoning_steps: list[str] | None = None,
) -> State:
	reasoning_steps = reasoning_steps or ["I should add the numbers."]
	return State(
		input={"question": question},
		reasoning=[{"reasoning_step": step} for step in reasoning_steps],
		output={},
	)


def _make_state_orm(
	question: str = "What is 2+2?",
	reasoning_steps: list[str] | None = None,
	answer: str = "4",
) -> State:
	reasoning_steps = reasoning_steps or ["2+2=4."]
	return State(
		input={"question": question},
		reasoning=[{"reasoning_step": step} for step in reasoning_steps],
		output={"answer": answer},
	)


class TestTreeOfThoughtRerankerEvaluator:
	"""Unit tests for TreeOfThoughtRerankerEvaluator."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"generator_signature",
			"consider_reasoning_in_final_eval",
		],
		# Parameter values
		[
			pytest.param(
				QuestionAnsweringWithReasoning,  	# generator_signature
				False,  							# consider_reasoning_in_final_eval
				id="init_default_no_reasoning_in_final",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,		# generator_signature
				True,								# consider_reasoning_in_final_eval
				id="init_consider_reasoning_in_final",
			),
		],
	)
	def test_initialization(
		self,
		generator_signature: type[ReasoningSignature],
		consider_reasoning_in_final_eval: bool,
	) -> None:
		"""Evaluator initializes with signature and flags and supports LM set/get.

		Args:
			generator_signature: The task signature for ToT generation. This is used to
				provide task instructions and input/output field metadata to the scoring adapter.
			consider_reasoning_in_final_eval: Whether ORM scoring should condition on the
				reasoning trajectory in addition to the final output.
		"""
		evaluator = TreeOfThoughtRerankerEvaluator(
			generator_signature=generator_signature,
			consider_reasoning_in_final_eval=consider_reasoning_in_final_eval,
		)
		assert evaluator.generator_signature is not None
		assert evaluator.consider_reasoning_in_final_eval is consider_reasoning_in_final_eval

		with pytest.raises(ValueError, match="Language model has not been set"):
			_ = evaluator.get_lm()

		mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.5]])
		evaluator.set_lm(mock_lm)
		assert evaluator.get_lm() == mock_lm

	@pytest.mark.parametrize(
		# Parameter names
		[
			"state",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				State(input={"question": "Q"}, reasoning=[], output={}),  	# state
				ValueError,  												# expected_exception
				id="prm_empty_reasoning_raises",
			),
		],
	)
	def test_prm_requires_reasoning(
		self,
		state: State,
		expected_exception: type[BaseException],
	) -> None:
		"""PRM reranker evaluation must reject states with no reasoning steps."""
		evaluator = TreeOfThoughtRerankerEvaluator(generator_signature=QuestionAnsweringWithReasoning)
		evaluator.set_lm(MockScoringLocalVLLM(rerank_responses=[[0.5]]))

		with pytest.raises(expected_exception):
			_ = evaluator(states=state)

	@pytest.mark.parametrize(
		# Parameter names
		[
			"states",
			"rerank_scores",
			"n_samples_evaluator",
			"expected_scores",
		],
		# Parameter values
		[
			pytest.param(
				[_make_state_prm()],  		# states
				[[0.8]],  					# rerank_scores
				1,  						# n_samples_evaluator
				[0.8],  					# expected_scores
				id="single_state_prm_score",
			),
			pytest.param(
				[_make_state_orm()],  		# states
				[[0.25]],  					# rerank_scores
				1,  						# n_samples_evaluator
				[0.25],  					# expected_scores
				id="single_state_orm_score",
			),
			pytest.param(
				[							# states
					_make_state_prm(),
					_make_state_orm()
				],
				[							# rerank_scores
					[0.9],  # PRM call layer
					[0.1],  # ORM call layer
				],
				7,  						# n_samples_evaluator (ignored)
				[0.9, 0.1],  				# expected_scores
				id="mixed_prm_orm_preserves_order_single_judge_enforced",
			),
			pytest.param(
				[							# states
					_make_state_prm(reasoning_steps=["Step A."]),
					_make_state_prm(reasoning_steps=["Step B."]),
				],
				[							# rerank_scores
					[0.2, 0.7],  			# single PRM batch call with 2 pairs
				],
				3,  						# n_samples_evaluator (ignored)
				[0.2, 0.7],  				# expected_scores
				id="prm_two_states_batched_single_call",
			),
		],
	)
	def test_forward_scores_and_ordering(
		self,
		states: list[State],
		rerank_scores: list[list[float]],
		n_samples_evaluator: int,
		expected_scores: list[float],
	) -> None:
		"""Forward returns one EvaluationResult per state, preserving order."""
		evaluator = TreeOfThoughtRerankerEvaluator(generator_signature=QuestionAnsweringWithReasoning)
		evaluator.set_lm(MockScoringLocalVLLM(rerank_responses=rerank_scores))

		results = evaluator(states=states, n_samples_evaluator=n_samples_evaluator)
		assert len(results) == len(states)
		for result_list, expected_score in zip(results, expected_scores, strict=True):
			assert isinstance(result_list, list)
			assert len(result_list) == 1
			assert result_list[0].score == expected_score
			assert result_list[0].judge_evaluations
			assert result_list[0].judge_evaluations[0].normalized_scores["relevance"] == expected_score


	@pytest.mark.parametrize(
		[
			"scoring_target",
			"rubric",
			"consider_reasoning_in_output",
			"is_reasoning_empty",
			"expected_instructions",
		],
		[
			pytest.param(
				"reasoning",					# scoring_target
				None,							# rubric
				True,							# consider_reasoning_in_output
				True,							# is_reasoning_empty
				(								# expected_instructions
"""
Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Answer the provided question with step-by-step reasoning."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
""".strip()
				),
				id="build_instructions_prm_no_rubric_no_prior_context",
			),
			pytest.param(
				"output",						# scoring_target
				None,							# rubric
				True,							# consider_reasoning_in_output
				True,							# is_reasoning_empty
				(								# expected_instructions
"""
Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Answer the provided question with step-by-step reasoning."

You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs.
""".strip()
				),
				id="build_instructions_orm_consider_reasoning_no_rubric_no_reasoning",
			),
			pytest.param(
				"output",						# scoring_target
				None,							# rubric
				True,							# consider_reasoning_in_output
				False,							# is_reasoning_empty
				(								# expected_instructions
"""
Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Answer the provided question with step-by-step reasoning."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs and reasoning.
""".strip()
				),
				id="build_instructions_orm_consider_reasoning_no_rubric_with_reasoning",
			),
			pytest.param(
				"output",						# scoring_target
				None,							# rubric
				False,							# consider_reasoning_in_output
				False,							# is_reasoning_empty
				(								# expected_instructions
"""
Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Answer the provided question with step-by-step reasoning."

You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs.
""".strip()
				),
				id="build_instructions_orm_ignore_reasoning_no_rubric",
			),
			pytest.param(
				"reasoning",					# scoring_target
				dspy.ensure_signature(WeightedPRMRubric),	# rubric
				True,							# consider_reasoning_in_output
				False,							# is_reasoning_empty
				(								# expected_instructions
"""
Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Answer the provided question with step-by-step reasoning."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs, using the rubric below.

Rubric:
- soundness: Correctness of the reasoning so far
- promise: Likelihood of the reasoning to lead to a strong final answer
""".strip()
				),
				id="build_instructions_prm_with_rubric_with_prior_context",
			),
			pytest.param(
				"output",						# scoring_target
				dspy.ensure_signature(WeightedMultiDimensionRubric),	# rubric
				True,							# consider_reasoning_in_output
				False,							# is_reasoning_empty
				(								# expected_instructions
"""
Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Answer the provided question with step-by-step reasoning."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs and reasoning, using the rubric below.

Rubric:
- correctness: accuracy of facts, formulas, and calculations
- clarity: how well-explained and understandable the reasoning is
- efficiency: directness and conciseness of the approach
""".strip()
				),
				id="build_instructions_orm_with_rubric_with_reasoning",
			),
		],
	)
	def test_build_scoring_instructions(
		self,
		scoring_target: Literal["reasoning", "output"],
		rubric: dspy.Signature | None,
		consider_reasoning_in_output: bool,
		is_reasoning_empty: bool,
		expected_instructions: str,
	) -> None:
		"""build_scoring_instructions returns exact prompt instructions for reranker scoring.

		Args:
			scoring_target: The scoring target (REASONING or OUTPUT).
			rubric: The rubric for the scoring target.
			consider_reasoning_in_output: Whether to consider reasoning in the output.
			is_reasoning_empty: Whether the reasoning is empty.
			expected_instructions: The expected instructions.
		"""
		evaluator = TreeOfThoughtRerankerEvaluator(generator_signature=QuestionAnsweringWithReasoning)
		actual = evaluator.build_scoring_instructions(
			scoring_target=scoring_target,
			rubric=rubric,
			consider_reasoning_in_output=consider_reasoning_in_output,
			is_reasoning_empty=is_reasoning_empty,
		)
		assert actual == expected_instructions


# =============================================================================
# Semantic Unit Tests (GPU Required)
# =============================================================================

# Shared GPU model fixture for all GPU tests
@pytest.fixture(scope="module")
def shared_reranker_model():
	"""Shared ScoringLocalVLLM fixture for all GPU integration tests.

	This fixture loads a reranker model once and shares it across all GPU test classes
	to avoid loading multiple models and running out of GPU memory.
	"""
	if not torch.cuda.is_available():
		pytest.skip("GPU not available")

	base_path = "/projects/BSTEWART/model_storage"
	model_name = OpenSourceModel.QWEN_3_RERANKER_8B.value
	model_path = os.path.join(base_path, model_name)
	lm = None
	try:
		logger.info(f"Initializing shared reranker model from: {model_path}")
		lm = ScoringLocalVLLM(
			model=model_path,
			tensor_parallel_size=1,
			dtype="auto",
			gpu_memory_utilization=0.9,
			max_model_len=16_384,
			enforce_eager=True,
			verbosity="debug",
		)
		logger.info("Shared reranker model initialized successfully")
		yield lm
	except Exception as e:
		logger.error(f"Failed to load reranker model {model_path}: {e}")
		# Re-raise the exception so tests fail with clear error messages
		# rather than being skipped silently
		raise
	finally:
		# Cleanup after all GPU tests complete
		if lm is not None:
			logger.info("Cleaning up shared reranker model...")
			lm.kill()


@pytestmark_gpu
class TestRerankerSemantics:
	"""Semantic unit tests for the reranker evaluator (requires GPU)."""

	@pytest.fixture
	def reranker_lm(self, shared_reranker_model):
		"""Use the shared reranker model fixture."""
		return shared_reranker_model

	@pytest.fixture
	def evaluator(self, reranker_lm: ScoringLocalVLLM):
		"""Create a reranker evaluator instance with real LM."""
		evaluator = TreeOfThoughtRerankerEvaluator(
			generator_signature=QuestionAnsweringWithReasoning,
			verbosity="debug",
		)
		evaluator.set_lm(reranker_lm)
		return evaluator

	@pytest.mark.parametrize(
		# Parameter names
		[
			"worse_reasoning",
			"better_reasoning",
			"question",
			"pattern_type",
		],
		# Parameter values
		[
			# Gibberish reasoning tests
			pytest.param(
				",,,,,,,,,,,,,,",  					# worse_reasoning
				(									# better_reasoning
					"To solve 2+2, I need to add the two numbers together. 2 + 2 = 4."
				),
				"What is 2+2?",  					# question
				"gibberish",  						# pattern_type
				id="gibberish_only_commas_vs_normal",
			),
			pytest.param(
				"\n\n\n\n\n\n\n\n",  				# worse_reasoning
				(									# better_reasoning
					"The answer is obtained by adding 5 and 3. 5 + 3 = 8."
				),
				"What is 5+3?",  					# question
				"gibberish",  						# pattern_type
				id="gibberish_only_newlines_vs_normal",
			),
			pytest.param(
				"......................",  			# worse_reasoning
				(									# better_reasoning
					"Let me calculate 10 divided by 2. 10 / 2 = 5."
				),
				"What is 10/2?",  					# question
				"gibberish",  						# pattern_type
				id="gibberish_only_periods_vs_normal",
			),
			# Repeated reasoning tests
			pytest.param(
				[									# worse_reasoning
					"I need to add 2 and 2.", "I need to add 2 and 2.", "I need to add 2 and 2."
				],
				[									# better_reasoning
					"I need to add 2 and 2.", "Adding these numbers: 2 + 2.", "The result is 4."
				],
				"What is 2+2?",  					# question
				"repetition",  						# pattern_type
				id="repetition_repeated_steps_vs_varied",
			),
			pytest.param(
				[									# worse_reasoning
					"First, let's think about this problem.", "First, let's think about this problem."
				],
				[									# better_reasoning
					"First, let's think about this problem.", "We need to multiply 3 by 4 to get 12."
				],
				"What is 3*4?",  					# question
				"repetition",  						# pattern_type
				id="repetition_repeated_opening_vs_progress",
			),
			# Irrelevant reasoning tests
			pytest.param(
				[									# worse_reasoning
					"The weather is nice today. I like pizza. Dogs are great pets."
				],
				[									# better_reasoning
					"To solve 7+5, I add the numbers: 7 + 5 = 12."
				],
				"What is 7+5?",  					# question
				"irrelevance",  					# pattern_type
				id="irrelevance_vs_relevant_math",
			),
			pytest.param(
				[									# worse_reasoning
					"Paris is the capital of France. The ocean is blue."
				],
				[									# better_reasoning
					"The multiplication of 6 and 8 is 48. 6 * 8 = 48."
				],
				"What is 6*8?",  					# question
				"irrelevance",  					# pattern_type
				id="irrelevance_completely_offtopic_vs_ontopic",
			),
			# Poor rhetorical structure tests
			pytest.param(
				[									# worse_reasoning
					"For example, if we consider 2+2, the answer is 4."
				],
				[									# better_reasoning
					"Let me solve 2+2. Adding 2 and 2 gives us 4."
				],
				"What is 2+2?",  					# question
				"poor_rhetoric",  					# pattern_type
				id="rhetoric_starts_with_for_example",
			),
			pytest.param(
				[									# worse_reasoning
					"However, the sum of 5 and 3 is 8."
				],
				[									# better_reasoning
					"To find the sum of 5 and 3, I add them: 5 + 3 = 8."
				],
				"What is 5+3?",  					# question
				"poor_rhetoric",  					# pattern_type
				id="rhetoric_starts_with_however",
			),
			pytest.param(
				[									# worse_reasoning
					"On the other hand, we need to multiply 4 by 2."
				],
				[									# better_reasoning
					"Multiplying 4 by 2 gives us 8."
				],
				"What is 4*2?",  					# question
				"poor_rhetoric",  					# pattern_type
				id="rhetoric_starts_with_on_the_other_hand",
			),
			# Meta-discussion tests
			pytest.param(
				[									# worse_reasoning
					"This is a math problem. It involves addition. I should think about numbers."
				],
				[									# better_reasoning
					"Let me calculate: 3 + 4 = 7."
				],
				"What is 3+4?",  					# question
				"meta_discussion",  				# pattern_type
				id="meta_talking_about_task_vs_solving",
			),
			pytest.param(
				[									# worse_reasoning
					"I need to approach this systematically. First, I'll analyze the problem structure. "
					"This requires careful thought about the question format."
				],
				[									# better_reasoning
					"To solve 9 - 5, I subtract: 9 - 5 = 4."
				],
				"What is 9-5?",  					# question
				"meta_discussion",  				# pattern_type
				id="meta_analysis_vs_direct_solution",
			),
			pytest.param(
				[									# worse_reasoning
					"This question is asking me to perform a calculation. I should focus on accuracy. "
					"Mathematical problems like this require precision."
				],
				[									# better_reasoning
					"Dividing 12 by 3: 12 / 3 = 4."
				],
				"What is 12/3?",  					# question
				"meta_discussion",  				# pattern_type
				id="meta_describing_task_vs_executing",
			),
		],
	)
	def test_penalizes_poor_reasoning_patterns(
		self,
		evaluator: TreeOfThoughtRerankerEvaluator,
		worse_reasoning: list[str],
		better_reasoning: list[str],
		question: str,
		pattern_type: str,
	) -> None:
		"""Test that the reranker penalizes various poor reasoning patterns.

		This consolidated test covers multiple types of poor reasoning:
		- gibberish: Non-semantic or meaningless text
		- repetition: Redundant repeated statements
		- irrelevance: Off-topic reasoning
		- poor_rhetoric: Poor structural/rhetorical choices
		- meta_discussion: Talking about the task instead of solving it
		"""
		worse_state = _make_state_prm(question=question, reasoning_steps=worse_reasoning)
		better_state = _make_state_prm(question=question, reasoning_steps=better_reasoning)

		worse_results = evaluator(states=worse_state)
		better_results = evaluator(states=better_state)

		worse_score = worse_results[0][0].score
		better_score = better_results[0][0].score

		logger.info(f"[{pattern_type}] Worse score: {worse_score}, Better score: {better_score}")
		assert better_score > worse_score, (
			f"[{pattern_type}] Expected better reasoning (score={better_score}) to score higher than "
			f"worse reasoning (score={worse_score})"
		)


if __name__ == "__main__":
	pytest.main([__file__, "-vv"])

