"""
Unit tests for TreeOfThoughts.

- Initialization with various signatures
- Parameter validation
- Thought generation
- Thought evaluation
- Thought selection (GREEDY and SAMPLE strategies)
- Integration tests with real vLLM models (requires GPU)

Run with: pytest test_tree_of_thoughts.py -v
"""

# TODO[P3]: Add new tests relating to the new `n_final_responses_per_trajectory` parameter when creating final responses.

# Standard library imports
import json
import logging
import os
from collections.abc import Generator
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

# Third-party imports
import dspy
import pytest
import torch

# Local imports
from adapter.constraints import ResponseLength
from constants import OpenSourceModel, Verbosity
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.scoring_local_lm import ScoringLocalVLLM
from misc_utils import ExecutionError
from predict.controller.controller import TreeOfThoughtsController
from predict.controller.controller_constants import (
	DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	ControllerOutput,
)
from predict.controller.controller_reranker import TreeOfThoughtsControllerReranker
from predict.controller.controller_utils import (
	FINISH_TOOL,
	ReasoningIntervention,
	return_action_if_single_option,
)
from predict.evaluator.evaluator import TreeOfThoughtEvaluator
from predict.evaluator.evaluator_reranker import TreeOfThoughtRerankerEvaluator
from predict.generator.generator import TreeOfThoughtGenerator
from predict.tree_of_thoughts import (
	TreeOfThoughts,
	TreeOfThoughtsParameters,
)
from signatures import (
	ArgumentEvaluatorMultiDimensional,
	ArgumentEvaluatorSingleScore,
	GenerateArgumentWithReasoning,
	QuestionAnsweringWithReasoning,
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from tree import (
	Edge,
	Input,
	Node,
	State,
	Tree,
)
from utilities_for_tests import (
	MockGenerativeLocalVLLM,
	MockScoringLocalVLLM,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _has_at_least_2_gpus() -> bool:
	"""Check if at least 2 GPUs are available."""
	try:
		if not torch.cuda.is_available():
			return False
		device_count = torch.cuda.device_count()
		# Handle case where device_count might be a mock
		if isinstance(device_count, int):
			return device_count >= 2
		return False
	except (AttributeError, TypeError):
		# torch.cuda might not be available or might be mocked
		return False


# Skip reranker tests if less than 2 GPUs are available
# This is defined early so it can be used in parametrized tests
pytestmark_2_gpus = pytest.mark.skipif(
	not _has_at_least_2_gpus(),
	reason="Reranker tests require at least 2 GPUs",
)

# Skip all GPU tests if GPU is not available
pytestmark_gpu = pytest.mark.skipif(
	not torch.cuda.is_available(),
	reason="GPU tests require GPU access",
)

# =============================================================================
# Helper Functions
# =============================================================================


def build_tree_from_nodes(nodes: list[Node]) -> tuple[Tree, list[Node]]:
	"""
	Build a tree from a list of nodes with proper parent-child relationships.

	Args:
		nodes: List of nodes where first node is root, others have parent_id set

	Returns:
		Tuple of (tree, frontier) where frontier is nodes in deepest layer with controller_outputs
	"""
	tree = Tree.from_nodes(nodes)

	# Extract frontier - nodes in deepest layer with controller outputs
	max_layer = max(node.layer for node in tree.nodes.values())
	frontier = [
		node
		for node in tree.nodes.values()
		if node.layer == max_layer and node.state.controller_outputs
	]

	# If no controller outputs are present in the deepest layer, return all nodes in that layer
	if not frontier:
		frontier = [node for node in tree.nodes.values() if node.layer == max_layer]

	return tree, frontier


DEFAULT_TOT_PARAMS = TreeOfThoughtsParameters(
	n_samples_generation=3,
	n_samples_judge=1,
	top_k=2,
	depth=3,
	node_selection_strategy="greedy",
	do_early_stopping=False,
)


def clone_tot_params(
	base_params: TreeOfThoughtsParameters,
	**overrides: int | float | bool,
) -> TreeOfThoughtsParameters:
	"""
	Copy TreeOfThoughtsParameters with overrides for specific tests.

	Args:
		base_params: Baseline parameters to clone.
		**overrides: Parameter overrides (e.g., n_samples_generation=2).

	Returns:
		New TreeOfThoughtsParameters instance with overrides applied.
	"""
	params_dict = asdict(base_params)
	params_dict.update(overrides)
	return TreeOfThoughtsParameters(**params_dict)


def root_only_tree(input_data: Input) -> Tree:
	"""
		Create a tree with only a root node for GPU and local tests.

	Args:
		input_data: Input dictionary for the root node.

	Returns:
		Tree containing a single root node.
	"""
	return Tree(state=input_data)


def verify_selection_results(
	selected: list[Node],
	all_nodes: list[Node],
	expected_count: int,
	should_be_sorted_by_index: bool = True,
) -> None:
	"""
	Verify common selection result properties.

	Parameters:
	    selected: The nodes that were selected.
	    all_nodes: All nodes that were candidates for selection.
	    expected_count: Expected number of selected nodes.
	    should_be_sorted_by_index: Whether selected nodes should be sorted by index.
	"""
	# Verify count
	assert len(selected) == expected_count, (
		f"Expected {expected_count} selected nodes, got {len(selected)}"
	)

	# Verify all selected nodes are from the original pool
	# Use indices instead of node objects since Node is not hashable
	selected_indices = {node.index for node in selected}
	all_indices = {node.index for node in all_nodes}
	assert selected_indices.issubset(all_indices), (
		"Selected nodes should be subset of all nodes"
	)

	# Verify non-selected nodes are pruned
	non_selected = [node for node in all_nodes if node.index not in selected_indices]
	for node in non_selected:
		assert node.is_pruned, f"Non-selected node {node.index} should be pruned"

	# Verify selected nodes are not pruned
	for node in selected:
		assert not node.is_pruned, f"Selected node {node.index} should not be pruned"

	# Verify sorting by index if required
	if should_be_sorted_by_index:
		indices = [node.index for node in selected]
		assert indices == sorted(indices), "Selected nodes should be sorted by index"


# =============================================================================
# Action Space Fixtures
# =============================================================================


@pytest.fixture
def temp_action_space_styles(tmp_path: Path) -> Path:
	"""Create a temporary JSON file for styles."""
	styles_json = {
		"name": "style",
		"definition": "Forces the next reasoning step to adopt a specific rhetorical style.",
		"choices": {
			"Figurative Language": {
				"definition": "Use metaphor, simile, or analogy.",
				"internal_reasoning": "I should employ non-literal comparison.",
			},
			"Statistical & Data-Driven": {
				"definition": "Present numerical data or statistics.",
				"internal_reasoning": "I should use numbers and data.",
				"prefix": "The data shows:",
			},
		},
	}
	styles_file = tmp_path / "styles.json"
	styles_file.write_text(json.dumps(styles_json))
	return styles_file


@pytest.fixture
def temp_action_space_structures(tmp_path: Path) -> Path:
	"""Create a temporary JSON file for structures."""
	structures_json = {
		"name": "structure",
		"definition": "Control the argumentative structure.",
		"choices": {
			"Causal Reasoning": {
				"definition": "State causes and effects.",
				"internal_reasoning": "I should explain cause and effect.",
				"prefix": "Therefore,",
			},
			"Contrast": {
				"definition": "Present contrasting viewpoints.",
				"internal_reasoning": "I should present a contrasting view.",
				"prefix": "However,",
			},
		},
	}
	structures_file = tmp_path / "structures.json"
	structures_file.write_text(json.dumps(structures_json))
	return structures_file


@pytest.fixture
def temp_action_space_subtopics(tmp_path: Path) -> Path:
	"""Create a temporary JSON file for subtopics."""
	subtopics_json = {
		"name": "subtopic",
		"definition": "Focus on a specific subtopic.",
		"choices": {
			"Cost-Benefit Analysis": {
				"definition": "Analyze costs and benefits.",
				"internal_reasoning": "I should weigh the pros and cons.",
			},
			"Ethical Considerations": {
				"definition": "Discuss moral implications.",
				"internal_reasoning": "I should consider the ethics.",
			},
		},
	}
	subtopics_file = tmp_path / "subtopics.json"
	subtopics_file.write_text(json.dumps(subtopics_json))
	return subtopics_file


# =============================================================================
# Test Class 1: Initialization
# =============================================================================


class TestTreeOfThoughtsInit:
	"""Test TreeOfThoughts initialization with various configurations."""

	def test_basic_initialization(self):
		"""Test basic initialization with default parameters."""
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		assert tot.generator_signature == QuestionAnsweringWithReasoning
		# Verify real instances were created (not mocks)
		assert tot.generator is not None
		assert tot.evaluator is not None
		assert tot.reasoning_field_name == "reasoning_step"
		# Verify LM was set on components
		assert tot.generator.lm == mock_lm

	def test_initialization_with_controller(self):
		"""Test initialization with generator controller."""
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		# Verify real controller instance was created (not a mock)
		assert tot.controller is not None
		# For generator controller, LM should be set
		assert tot.controller.get_lm() == mock_lm

	def test_initialization_with_reranker(self):
		"""Test initialization with reranker controller."""
		# Use MockGenerativeLocalVLLM for generative and MockScoringLocalVLLM for reranker
		mock_generative_lm = MockGenerativeLocalVLLM([])
		mock_reranker_lm = MockScoringLocalVLLM([])
		dspy.settings.configure(lm=mock_generative_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_generative_lm,
			reranker_lm=mock_reranker_lm,
			controller_type="reranker",
		)

		# Verify real controller instance was created (not a mock)
		assert tot.controller is not None
		assert isinstance(tot.controller, TreeOfThoughtsControllerReranker)
		assert tot.controller.get_lm() == mock_reranker_lm
		assert tot.generator.lm == mock_generative_lm

	def test_initialization_with_reranker_evaluator(self) -> None:
		"""Test initialization with reranker evaluator (controller may still be generative)."""
		mock_generative_lm = MockGenerativeLocalVLLM([])
		mock_reranker_lm = MockScoringLocalVLLM([[0.5]])
		dspy.settings.configure(lm=mock_generative_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_generative_lm,
			reranker_lm=mock_reranker_lm,
			controller_type="generator",
			evaluator_type="reranker",
		)

		assert isinstance(tot.evaluator, TreeOfThoughtRerankerEvaluator)
		assert tot.evaluator.get_lm() == mock_reranker_lm
		assert tot.generator.lm == mock_generative_lm
		assert tot.controller.get_lm() == mock_generative_lm

	def test_initialization_with_reranker_evaluator_missing_reranker_lm(self) -> None:
		"""Reranker evaluator requires reranker_lm."""
		mock_generative_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_generative_lm)

		with pytest.raises(ValueError, match="reranker_lm must be provided"):
			TreeOfThoughts(
				generator_signature=QuestionAnsweringWithReasoning,
				generative_lm=mock_generative_lm,
				controller_type="generator",
				evaluator_type="reranker",
			)

	def test_initialization_with_reranker_controller_missing_reranker_lm(self):
		"""Test that initialization fails when reranker_lm is not provided for RERANKER controller."""
		mock_generative_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_generative_lm)

		with pytest.raises(ValueError, match="reranker_lm must be provided"):
			TreeOfThoughts(
				generator_signature=QuestionAnsweringWithReasoning,
				generative_lm=mock_generative_lm,
				controller_type="reranker",
			)


# =============================================================================
# Test Class 2: Parameter Validation
# =============================================================================


class TestParameterValidation:
	"""Test TreeOfThoughtsParameters validation logic."""

	@pytest.mark.parametrize(
		("params_overrides", "should_pass", "error_message_fragment"),
		[
			pytest.param({}, True, None, id="default_valid_parameters"),
			pytest.param(
				{"n_samples_generation": 5, "top_k": 3},
				True,
				None,
				id="valid_n_samples_gt_top_k",
			),
			pytest.param(
				{"n_samples_generation": 3, "top_k": 3},
				True,
				None,
				id="valid_n_samples_eq_top_k",
			),
			pytest.param(
				{"n_samples_generation": 0},
				False,
				"n_samples_generation",
				id="invalid_zero_n_samples",
			),
			pytest.param({"top_k": 0}, False, "top_k", id="invalid_zero_top_k"),
			pytest.param({"depth": 0}, False, "depth", id="invalid_zero_depth"),
			pytest.param(
				{"depth": -1},
				False,
				"depth",
				id="invalid_negative_depth",
			),
		],
	)
	def test_parameter_validation(
		self,
		params_overrides,
		should_pass,
		error_message_fragment,
	):
		"""Test parameter validation with various configurations."""
		# Create parameters with overrides
		params_dict = asdict(DEFAULT_TOT_PARAMS)
		params_dict.update(params_overrides)

		if should_pass:
			# Should not raise
			params = TreeOfThoughtsParameters(**params_dict)
			# Call assert_valid_parameters to ensure no errors
			TreeOfThoughts.assert_valid_parameters(params)
		else:
			# Should raise ValueError with specific message
			with pytest.raises((ValueError, Exception)) as exc_info:
				params = TreeOfThoughtsParameters(**params_dict)
				TreeOfThoughts.assert_valid_parameters(params)
			if error_message_fragment:
				assert error_message_fragment in str(exc_info.value).lower()

	@pytest.mark.parametrize(
		("top_k", "n_samples_generation", "initial_do_pruning", "expected_do_pruning"),
		[
			pytest.param(
				2, 5, False, True,
				id="auto_enable_when_top_k_lt_n_samples",
			),
			pytest.param(
				3, 3, False, False,
				id="no_change_when_top_k_eq_n_samples",
			),
			pytest.param(
				2, 5, True, True,
				id="stays_true_when_already_true",
			),
		],
	)
	def test_do_pruning_auto_enable(
		self,
		top_k: int,
		n_samples_generation: int,
		initial_do_pruning: bool,
		expected_do_pruning: bool,
	):
		"""Test that do_pruning is auto-enabled when top_k < n_samples_generation.

		Note: We cannot test top_k > n_samples_generation without top_k_first
		because that configuration is invalid (not enough candidates to select from).
		"""
		params = TreeOfThoughtsParameters(
			top_k=top_k,
			n_samples_generation=n_samples_generation,
			do_pruning=initial_do_pruning,
		)
		assert params.do_pruning == expected_do_pruning, (
			f"Expected do_pruning={expected_do_pruning} when top_k={top_k}, "
			f"n_samples_generation={n_samples_generation}, initial={initial_do_pruning}"
		)

	@pytest.mark.parametrize(
		("n_samples_generation", "top_k", "top_k_first", "should_pass"),
		[
			pytest.param(
				100, 250, 50, True,
				id="valid_top_k_first_bypasses_n_samples_constraint",
			),
			pytest.param(
				100, 250, None, False,
				id="invalid_n_samples_lt_top_k_without_top_k_first",
			),
			pytest.param(
				250, 250, None, True,
				id="valid_n_samples_eq_top_k_without_top_k_first",
			),
		],
	)
	def test_n_samples_top_k_constraint_with_top_k_first(
		self,
		n_samples_generation: int,
		top_k: int,
		top_k_first: int | None,
		should_pass: bool,
	):
		"""Test that n_samples_generation >= top_k constraint is conditional on top_k_first.

		Regression test for: AssertionError: `n_samples_generation` must be greater
		than or equal to `top_k`, but got 100 < 250

		When top_k_first is set, the constraint should not apply because:
		- Layer 1 uses top_k_first (not top_k)
		- Layer 2+ has n_samples * top_k_first candidates to select top_k from
		"""
		if should_pass:
			# Should not raise - create params and call assert_valid_parameters
			params = TreeOfThoughtsParameters(
				n_samples_generation=n_samples_generation,
				top_k=top_k,
				top_k_first=top_k_first,
			)
			TreeOfThoughts.assert_valid_parameters(params)
		else:
			# Should raise AssertionError
			with pytest.raises(AssertionError, match="n_samples_generation"):
				params = TreeOfThoughtsParameters(
					n_samples_generation=n_samples_generation,
					top_k=top_k,
					top_k_first=top_k_first,
				)
				TreeOfThoughts.assert_valid_parameters(params)


# =============================================================================
# Test Class 3: Generate Thoughts
# =============================================================================


class TestGenerateThoughts:
	"""Test thought generation with mocked generator."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"tot_params",
			"nodes",
			"mock_responses",
			"expected_child_count",
			"expected_child_layer",
		],
		# Parameter values
		[
			pytest.param(
				TreeOfThoughtsParameters(
					n_samples_generation=2,
					n_samples_judge=1,
					top_k=2,
					depth=2,
				),  # tot_params
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(
							input={"question": "What is the capital of France?"},
							controller_outputs=[
								ControllerOutput(
									action="continue_reasoning",
									action_arguments={},
									tool_descriptions="Action Name: continue_reasoning",
									continue_reasoning=True,
									unique_action_response_count=2,
								)
							],
						),
					),
				],
				# mock_responses: [num_layers, num_requests, num_completions] = [1, 1, 2]
				[[["## reasoning_step\nTest reasoning step"] * 2]],
				2,  # expected_child_count
				1,  # expected_child_layer
				id="root_frontier_generates_children",
			),
			pytest.param(
				TreeOfThoughtsParameters(
					n_samples_generation=1,
					n_samples_judge=1,
					top_k=1,
					depth=3,
				),  # tot_params
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(input={"question": "Who invented calculus?"}),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Who invented calculus?"},
							reasoning=[{"reasoning_step": "First step"}],
							controller_outputs=[
								ControllerOutput(
									action="continue_reasoning",
									action_arguments={},
									tool_descriptions="continue",
									continue_reasoning=True,
									unique_action_response_count=1,
								)
							],
						),
					),
					Node(
						index=2,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Who invented calculus?"},
							reasoning=[{"reasoning_step": "Second step"}],
							controller_outputs=[
								ControllerOutput(
									action="continue_reasoning",
									action_arguments={},
									tool_descriptions="continue",
									continue_reasoning=True,
									unique_action_response_count=1,
								)
							],
						),
					),
				],
				# mock_responses: [num_layers, num_requests, num_completions] = [1, 2, 1]
				[[["## reasoning_step\nGenerated child"], ["## reasoning_step\nGenerated child"]]],
				2,  # expected_child_count
				2,  # expected_child_layer
				id="layer_one_frontier_generates_layer_two",
			),
		],
	)
	def test_generate_thoughts_local(
		self,
		tot_params,
		nodes,
		mock_responses,
		expected_child_count,
		expected_child_layer,
	):
		"""Test thought generation creates correct number of child nodes."""
		# Configure mocks - pass unprocessed messages directly
		mock_lm = MockGenerativeLocalVLLM(responses=mock_responses)
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		tree, frontier = build_tree_from_nodes(nodes)

		# Generate thoughts
		child_nodes = tot.generate_thoughts(
			frontier=frontier,
			tree=tree,
			tot_parameters=tot_params,
		)

		# Verify results
		assert len(child_nodes) == expected_child_count
		assert {child.layer for child in child_nodes} == {expected_child_layer}
		assert all(
			child.parent_id in {node.index for node in frontier}
			for child in child_nodes
		)
		# Every generated node should preserve ExecutionError metadata so downstream validity
		# checks don't KeyError when a parse fails.
		for child in child_nodes:
			if child.state.output:
				assert "error" in child.state.output
			else:
				assert child.state.reasoning, "Intermediate nodes should have at least one reasoning step"
				assert "error" in child.state.reasoning[-1]


# =============================================================================
# Test Class 4: Evaluate Thoughts
# =============================================================================


class TestEvaluateThoughts:
	"""Test thought evaluation with mocked evaluator."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"tot_params",
			"nodes",
			"mock_responses",
			"expected_scored_nodes",
		],
		# Parameter values
		[
			pytest.param(
				TreeOfThoughtsParameters(
					n_samples_generation=1,
					n_samples_judge=1,
					top_k=1,
					depth=2,
				),  # tot_params
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[
								{
									"reasoning_step": "Rayleigh scattering explains it.",
									"error": ExecutionError(),
								}
							],
							controller_outputs=[
								ControllerOutput(
									action="continue_reasoning",
									action_arguments={},
									tool_descriptions="continue",
									continue_reasoning=True,
									unique_action_response_count=1,
								)
							],
						),
					),
				],
			# mock_responses: [num_layers, num_requests, num_completions] = [1, 1, 1]
			[[[
				"## soundness\n5\n## promise\n5\n## feedback\nTest feedback"
			]]],
			1,  # expected_scored_nodes
			id="intermediate_reasoning_scores_nodes",
			),
			pytest.param(
				TreeOfThoughtsParameters(
					n_samples_generation=1,
					n_samples_judge=1,
					top_k=1,
					depth=2,
				),  # tot_params
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(
							input={"question": "Provide a final answer."}
						),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Provide a final answer."},
							reasoning=[
								{"reasoning_step": "Ready to answer.", "error": ExecutionError()}
							],
							output={"answer": "Here is the answer.", "error": ExecutionError()},
							controller_outputs=[
								ControllerOutput(
									action="continue_reasoning",
									action_arguments={},
									tool_descriptions="continue",
									continue_reasoning=True,
									unique_action_response_count=1,
								)
							],
						),
					),
				],
				# mock_responses: [num_layers, num_requests, num_completions] = [1, 1, 1]
				[[["## quality\n5\n## feedback\nTest feedback"]]],
				1,  # expected_scored_nodes
				id="final_output_scores_nodes",
			),
		],
	)
	def test_evaluate_thoughts_local(
		self,
		tot_params,
		nodes,
		mock_responses,
		expected_scored_nodes,
	):
		"""Test evaluation of nodes with single evaluation type."""
		# Configure mocks - pass unprocessed messages directly
		mock_lm = MockGenerativeLocalVLLM(responses=mock_responses)
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		tree, frontier = build_tree_from_nodes(nodes)

		# Evaluate thoughts
		scored_nodes = tot.evaluate_thoughts(
			frontier=frontier,
			tree=tree,
			tot_parameters=tot_params,
		)

		# Verify scores were assigned (evaluator normalizes scores, so check they're reasonable)
		actual_scores = [node.score for node in scored_nodes]
		assert len(actual_scores) == expected_scored_nodes
		assert all(score is not None for score in actual_scores)
		assert all(
		    score is not None and 0.0 <= score <= 1.0 for score in actual_scores
		)


class TestSelectNodesCommon:
	"""Test behavior common to both GREEDY and SAMPLE selection strategies."""

	@pytest.mark.parametrize(
		"strategy",
		[
			pytest.param("greedy", id="greedy"),
			pytest.param("sample", id="sample"),
		],
	)
	def test_selection_prunes_unselected_nodes(
		self,
		strategy,
	):
		"""Test that both strategies prune non-selected nodes."""
		# Configure mocks
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		# Create nodes with scores
		nodes = [
			Node(
				index=i,
				layer=1,
				state=State(input={"question": "Who invented calculus?"}),
			)
			for i in range(5)
		]
		for node, score in zip(nodes, [5.0, 4.0, 3.0, 2.0, 1.0], strict=True):
			node.score = score

		# Update parameters
		tot_params = clone_tot_params(
			DEFAULT_TOT_PARAMS,
			node_selection_strategy=strategy,
			top_k=3,
		)

		# Select thoughts
		selected = tot.select_nodes(nodes, tot_params)

		# Verify pruning
		verify_selection_results(selected, nodes, expected_count=3)

	@pytest.mark.parametrize(
		"strategy",
		[
			pytest.param("greedy", id="greedy"),
			pytest.param("sample", id="sample"),
		],
	)
	def test_selection_maintains_index_order(
		self,
		strategy,
	):
		"""Test that both strategies sort results by node index."""
		# Configure mocks
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		nodes = [
			Node(
				index=i,
				layer=1,
				state=State(input={"question": "Who invented calculus?"}),
			)
			for i in range(5)
		]
		for node, score in zip(nodes, [3.0, 1.0, 5.0, 2.0, 4.0], strict=True):
			node.score = score

		# Update parameters
		tot_params = clone_tot_params(
			DEFAULT_TOT_PARAMS,
			node_selection_strategy=strategy,
			top_k=3,
		)

		# Select thoughts
		selected = tot.select_nodes(nodes, tot_params)

		# Verify sorting by index
		indices = [node.index for node in selected]
		assert indices == sorted(indices), "Selected nodes must be sorted by index"


# =============================================================================
# Test Class 5: top_k_first Parameter
# =============================================================================


class TestTopKFirst:
	"""Test top_k_first parameter for first-layer selection override."""

	@pytest.mark.parametrize(
		("layer", "expected_count", "description"),
		[
			pytest.param(1, 2, "layer 1 uses top_k_first", id="layer_1_uses_top_k_first"),
			pytest.param(2, 4, "layer 2 uses top_k", id="layer_2_uses_top_k"),
		],
	)
	def test_top_k_first_only_applies_to_layer_1(self, layer, expected_count, description):
		"""Test that top_k_first is used for layer 1, and top_k for layer 2+."""
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		# Create 5 nodes at the specified layer with descending scores
		nodes = [
			Node(
				index=i,
				layer=layer,
				state=State(input={"question": "Test question"}),
			)
			for i in range(5)
		]
		for node, score in zip(nodes, [5.0, 4.0, 3.0, 2.0, 1.0], strict=True):
			node.score = score

		# Set top_k_first=2, top_k=4
		# n_samples_generation=5 satisfies all constraints
		tot_params = clone_tot_params(
			DEFAULT_TOT_PARAMS,
			node_selection_strategy="greedy",
			n_samples_generation=5,
			top_k=4,
			top_k_first=2,
		)

		selected = tot.select_nodes(nodes, tot_params)

		assert len(selected) == expected_count, (
			f"Expected {expected_count} nodes ({description}), got {len(selected)}"
		)

	def test_top_k_used_when_top_k_first_is_none(self):
		"""Test that top_k is used when top_k_first is None (default behavior)."""
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		# Create 5 nodes at layer 1 with descending scores
		nodes = [
			Node(
				index=i,
				layer=1,
				state=State(input={"question": "Test question"}),
			)
			for i in range(5)
		]
		for node, score in zip(nodes, [5.0, 4.0, 3.0, 2.0, 1.0], strict=True):
			node.score = score

		# top_k_first=None (default), top_k=3
		tot_params = clone_tot_params(
			DEFAULT_TOT_PARAMS,
			node_selection_strategy="greedy",
			top_k=3,
			top_k_first=None,
		)

		selected = tot.select_nodes(nodes, tot_params)

		# Should select 3 nodes (top_k) since top_k_first is None
		assert len(selected) == 3, f"Expected 3 nodes (top_k), got {len(selected)}"


# =============================================================================
# Test Class 6: Select Thoughts - GREEDY Strategy
# =============================================================================


class TestSelectNodesGreedy:
	"""Test GREEDY node selection strategy."""

	@pytest.mark.parametrize(
		("scores", "top_k", "expected_indices"),
		[
			pytest.param([1.0, 2.0, 3.0, 4.0, 5.0], 3, [2, 3, 4], id="basic_top3"),
			pytest.param(
				[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
				5,
				[0, 1, 2, 3, 4],
				id="descending_scores",
			),
			pytest.param(
				[5.0, 4.0, 3.0, 2.0, 1.0],
				10,
				[0, 1, 2, 3, 4],
				id="k_exceeds_nodes",
			),
			pytest.param(
				[1.0, 1.0, 1.0, 1.0, 1.0],
				5,
				[0, 1, 2, 3, 4],
				id="all_equal_scores",
			),
			pytest.param([3.5, 2.1, 8.7], 1, [2], id="float_scores_single"),
			pytest.param(
				[5.0, 5.0, 3.0, 3.0, 1.0], 2, [0, 1], id="duplicate_high_scores"
			),
		],
	)
	def test_greedy_selection(
		self,
		scores,
		top_k,
		expected_indices,
	):
		"""Test GREEDY strategy selects top-k nodes by score."""
		# Configure mocks
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		nodes = [
			Node(
				index=i,
				layer=1,
				state=State(input={"question": "Who invented calculus?"}),
			)
			for i, _ in enumerate(scores)
		]
		for node, score in zip(nodes, scores, strict=True):
			node.score = score

		# Update parameters
		tot_params = clone_tot_params(
			DEFAULT_TOT_PARAMS,
			node_selection_strategy="greedy",
			top_k=top_k,
			n_samples_generation=max(top_k, DEFAULT_TOT_PARAMS.n_samples_generation),
		)

		# Select thoughts
		selected = tot.select_nodes(nodes, tot_params)

		# Verify correct nodes were selected
		selected_indices = [node.index for node in selected]
		assert selected_indices == expected_indices

		# Verify common properties
		expected_count = min(top_k, len(nodes))
		verify_selection_results(selected, nodes, expected_count)

	def test_greedy_random_tie_breaking(self):
		"""Test that GREEDY strategy randomly breaks ties among top-scoring nodes."""
		# Test with multiple runs to verify randomness
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		# Create nodes with tied scores
		nodes = [
			Node(
				index=i,
				layer=1,
				state=State(input={"question": "Test"}),
			)
			for i in range(10)
		]
		# Create tie scenario: 5 nodes with score 1.0, 5 with score 0.5
		for i in range(5):
			nodes[i].score = 1.0
		for i in range(5, 10):
			nodes[i].score = 0.5

		tot_params = clone_tot_params(
			DEFAULT_TOT_PARAMS,
			node_selection_strategy="greedy",
			top_k=3,
		)

		# Run selection multiple times
		selections = []
		for _ in range(20):
			# Copy nodes to avoid mutation between runs
			nodes_copy = [
				Node(
					index=node.index,
					layer=node.layer,
					state=node.state,
					score=node.score,
					is_pruned=False,
				)
				for node in nodes
			]
			selected = tot.select_nodes(nodes_copy, tot_params)
			selections.append(tuple(node.index for node in selected))

		# Verify all selections have correct properties
		for selection in selections:
			assert len(selection) == 3, "Should select exactly 3 nodes"
			# All selected nodes should have score 1.0 (highest)
			assert all(nodes[idx].score == 1.0 for idx in selection)

		# Verify randomness: should see multiple different selections
		unique_selections = set(selections)
		assert len(unique_selections) > 1, (
			"Should produce different selections due to random tie-breaking"
		)

	def test_greedy_no_ties_deterministic(self):
		"""Test that GREEDY without ties produces consistent results."""
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		# Create nodes with unique scores (no ties)
		nodes = [
			Node(
				index=i,
				layer=1,
				state=State(input={"question": "Test"}),
			)
			for i in range(5)
		]
		for i, node in enumerate(nodes):
			node.score = 1.0 - (i * 0.1)  # Unique scores: 1.0, 0.9, 0.8, 0.7, 0.6

		tot_params = clone_tot_params(
			DEFAULT_TOT_PARAMS,
			node_selection_strategy="greedy",
			top_k=3,
		)

		# Run selection multiple times
		selections = []
		for _ in range(10):
			# Copy nodes to avoid mutation
			nodes_copy = [
				Node(
					index=node.index,
					layer=node.layer,
					state=node.state,
					score=node.score,
					is_pruned=False,
				)
				for node in nodes
			]
			selected = tot.select_nodes(nodes_copy, tot_params)
			selections.append(tuple(node.index for node in selected))

		# All selections should be identical (no randomness when no ties)
		unique_selections = set(selections)
		assert len(unique_selections) == 1, (
			"Should produce deterministic results when there are no ties"
		)
		# Should select indices 0, 1, 2 (highest scores)
		assert selections[0] == (0, 1, 2)


# =============================================================================
# Test Class 7: Select Thoughts - SAMPLE Strategy
# =============================================================================


class TestSelectThoughtsSample:
	"""Test SAMPLE node selection strategy (weighted sampling)."""

	@pytest.mark.parametrize(
		("scores", "top_k", "test_scenario"),
		[
			pytest.param(
				[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
				3,
				"basic",
				id="sample_basic_weighted",
			),
			pytest.param([0.0] * 10, 3, "all_zeros", id="sample_all_zeros_fallback"),
			pytest.param(
				[5.0, 4.0, 3.0, 2.0, 1.0],
				10,
				"k_exceeds",
				id="sample_k_exceeds_nodes",
			),
			pytest.param(
				[10.0] + [0.0] * 9,
				1,
				"high_probability",
				id="sample_one_dominant_score",
			),
		],
	)
	def test_sample_selection(
		self,
		scores,
		top_k,
		test_scenario,
	):
		"""Test SAMPLE strategy with various scenarios."""
		# Configure mocks
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		# Create nodes
		nodes = [
			Node(
				index=i,
				layer=1,
				state=State(input={"question": "Who invented calculus?"}),
			)
			for i, _ in enumerate(scores)
		]
		for node, score in zip(nodes, scores, strict=True):
			node.score = score

		# Update parameters
		tot_params = clone_tot_params(
			DEFAULT_TOT_PARAMS,
			node_selection_strategy="sample",
			top_k=top_k,
			n_samples_generation=max(top_k, DEFAULT_TOT_PARAMS.n_samples_generation),
		)

		# Select thoughts
		selected = tot.select_nodes(nodes, tot_params)

		# Common verifications
		expected_count = min(top_k, len(nodes))
		verify_selection_results(selected, nodes, expected_count)

		# Scenario-specific verifications
		if test_scenario == "all_zeros":
			# Should fall back to taking first top_k nodes
			selected_indices = [node.index for node in selected]
			assert selected_indices == list(range(min(top_k, len(nodes))))
		elif test_scenario == "k_exceeds":
			# Should select all nodes
			assert len(selected) == len(nodes)
		elif test_scenario == "one_dominant_score":
			# Node with dominant score should be selected more frequently
			selected_indices = [node.index for node in selected]
			assert 0 in selected_indices


# =============================================================================
# Test Class 8: Plan Actions
# =============================================================================


class TestPlanActions:
	"""Test controller-based action planning for tree nodes."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"tot_params",
			"nodes",
			"controller_type",
			"max_reasoning_steps",
			"expect_finish_outputs",
			"expect_exception",
		],
		# Parameter values
		[
			pytest.param(
				TreeOfThoughtsParameters(
					n_samples_generation=2,
					n_samples_judge=1,
					top_k=1,
					depth=1,
				),  # tot_params
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(
							input={
								"question": "What is the capital of France?"
							}
						),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "What is the capital of France?"},
							reasoning=[
								{"reasoning_step": "Placeholder reasoning to allow Tree.from_nodes edge construction."},
							],
						),
					),
				],
				"generator",  # controller_type
				1,  # max_reasoning_steps
				True,  # expect_finish_outputs
				None,  # expect_exception
				id="forces_finish_action_at_max_depth",
			),
			pytest.param(
				TreeOfThoughtsParameters(
					n_samples_generation=1,
					n_samples_judge=1,
					top_k=1,
					depth=2,
				),  # tot_params
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(input={"question": "Test question"}),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Test question"},
							reasoning=[{"reasoning_step": "First step"}],
						),
					),
				],
				"generator",  # controller_type
				3,  # max_reasoning_steps
				False,  # expect_finish_outputs
				AssertionError,  # expect_exception
				id="mixed_layers_raise_assertion",
			),
		],
	)
	def test_plan_actions_local(
		self,
		tot_params,
		nodes,
		controller_type,
		max_reasoning_steps,
		expect_finish_outputs,
		expect_exception,
	):
		"""
		Plan actions on frontiers constructed from nodes-to-tree tuples.

		Ensures finish actions are forced at max depth and that mixed-layer
		frontiers raise assertions.
		"""
		mock_generative_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_generative_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_generative_lm,
			controller_type=controller_type,
			max_reasoning_steps=max_reasoning_steps,
			evaluator_type="generator",
		)

		tree, frontier = build_tree_from_nodes(nodes)
		if expect_exception:
			frontier = list(tree.nodes.values())
			with pytest.raises(expect_exception):
				tot.plan_actions(
					frontier=frontier,
					tot_parameters=tot_params,
					controller_demos=None,
				)
			return

		result_nodes = tot.plan_actions(
			frontier=frontier,
			tot_parameters=tot_params,
			controller_demos=None,
		)

		assert len(result_nodes) == len(frontier)
		if expect_finish_outputs:
			for node in result_nodes:
				assert node.state.controller_outputs
				assert all(
					not output.continue_reasoning
					for output in node.state.controller_outputs
				)


# =============================================================================
# Test Class 9: Forward
# =============================================================================


class TestForward:
	"""Test the forward method builds the initial tree from problem input."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"tot_params",
			"problem_state",
			"expected_answer",
		],
		# Parameter values
		[
			pytest.param(
				TreeOfThoughtsParameters(
					n_samples_generation=1,
					n_samples_judge=1,
					top_k=1,
					depth=1,
				),  # tot_params
				{"question": "What is 2 + 2?"},  # problem_state
				"4",  # expected_answer
				id="root_only_initialization",
			),
		],
	)
	def test_forward_initializes_tree_and_root(
		self,
		tot_params,
		problem_state,
		expected_answer,
	):
		"""Ensure forward builds a tree rooted at the provided problem."""
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		def _step(frontier, tree, tot_parameters, **kwargs):
			parent = frontier[0]
			controller_output = ControllerOutput(
				action="finish",
				action_arguments={},
				tool_descriptions="finish",
				continue_reasoning=False,
				unique_action_response_count=1,
			)
			child = tree.create_child_node(
				parent_node=parent,
				output={"answer": expected_answer},
				controller_output=controller_output,
			)
			child.state.output = {"answer": expected_answer}
			child.score = 0.5
			# Add child to tree nodes manually since create_child_node does it but we mocked step
			# Wait, create_child_node IS called on the real tree object, so it should be fine.
			# But we need to return the child.
			return [child]

		# Mock controller to avoid LM calls
		with patch.object(TreeOfThoughts, "step", side_effect=_step) as step_mock, \
			 patch.object(tot, "controller", return_value=[]):
			output = tot.forward(
				state=problem_state,
				tot_parameters=tot_params,
			)

		assert step_mock.called
		assert output.tree.root.state.input == problem_state
		assert output.tree.root.index == 0
		if tot_params.num_final_candidates > 0:
			assert any(node.state.output for node in output.tree.nodes.values())
			assert len(output.responses) == tot_params.num_final_candidates
			if tot_params.num_final_candidates == 1:
				assert output.responses[0].response_data["answer"] == expected_answer
		else:
			assert not output.responses

	def test_create_output_random_tie_breaking_greedy(self):
		"""Test that create_output with GREEDY randomly selects among tied final candidates."""
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		tot_params = TreeOfThoughtsParameters(
			n_samples_generation=1,
			n_samples_judge=1,
			top_k=1,
			depth=1,
			num_final_candidates=2,
			node_selection_strategy="greedy",
			do_pruning=False,
		)

		# Create tree with tied final outputs
		tree = Tree(state={"question": "Test"})
		candidates = []
		for i in range(5):
			node = Node(
				index=i+1,
				layer=1,
				parent_id=0,
				state=State(
					input={"question": "Test"},
					output={"answer": f"Answer {i}"},
				),
			)
			node.score = 1.0  # All tied
			tree.nodes[node.index] = node
			candidates.append(node)

		# Run create_output multiple times
		outputs = []
		for _ in range(20):
			# Copy candidates to avoid mutation
			candidates_copy = [
				Node(
					index=node.index,
					layer=node.layer,
					parent_id=node.parent_id,
					state=node.state,
					score=node.score,
					is_pruned=False,
				)
				for node in candidates
			]
			output = tot.create_output(
				candidates=candidates_copy,
				tree=tree,
				tot_parameters=tot_params,
				start_time=0.0,
			)
			selected_answers = tuple(
				resp.response_data["answer"]
				for resp in output.responses
			)
			outputs.append(selected_answers)

		# Verify correct count
		for output in outputs:
			assert len(output) == 2, "Should select exactly 2 final candidates"

		# Verify randomness
		unique_outputs = set(outputs)
		assert len(unique_outputs) > 1, (
			"Should produce different selections due to random tie-breaking"
		)


@pytest.mark.parametrize(
	"use_self_consistency",
	[
		pytest.param(False, id="no_self_consistency"),
		pytest.param(True, id="with_self_consistency"),
	],
)
def test_create_output_strips_error_metadata_from_response_strings(
	use_self_consistency: bool,
) -> None:
	"""Final response strings should not include `error` metadata appended to node outputs."""
	mock_lm = MockGenerativeLocalVLLM([])
	dspy.settings.configure(lm=mock_lm)

	tot = TreeOfThoughts(
		generator_signature=QuestionAnsweringWithReasoning,
		generative_lm=mock_lm,
		controller_type="generator",
		evaluator_type="generator",
	)

	tot_params = TreeOfThoughtsParameters(
		n_samples_generation=1,
		n_samples_judge=1,
		top_k=1,
		depth=1,
		num_final_candidates=1,
		do_pruning=False,
		use_self_consistency=use_self_consistency,
	)

	tree = Tree(state={"question": "Test"})
	node = Node(
		index=1,
		layer=1,
		parent_id=0,
		state=State(
			input={"question": "Test"},
			output={"answer": "Hello", "error": ExecutionError()},
		),
	)
	node.score = 1.0
	tree.nodes[node.index] = node

	output = tot.create_output(
		candidates=[node],
		tree=tree,
		tot_parameters=tot_params,
		start_time=0.0,
	)

	assert output.response_strings == ["Hello"]
	assert output.responses[0].to_dict() == {"answer": "Hello"}


# =============================================================================
# Test Class 10: Final Output Type Configuration
# =============================================================================


class TestFinalOutputKind:
	"""Test final_output_kind configuration."""

	@pytest.mark.parametrize(
		("final_output_kind", "expected_kind"),
		[
			pytest.param(
				"synthesis_faithful",
				"synthesis_faithful",
				id="synthesis",
			),
			pytest.param(
				"conclusion",
				"conclusion",
				id="conclusion",
			),
		],
	)
	def test_final_output_kind_passed_to_generator(
		self,
		final_output_kind,
		expected_kind,
	):
		"""Test that final_output_kind is passed to generator."""

		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			final_output_kind=final_output_kind,
			controller_type="generator",
			evaluator_type="generator",
		)

		# Verify generator was initialized with correct final_output_kind
		assert tot.generator.final_output_kind == expected_kind

	def test_default_final_output_kind(self):
		"""Test default final_output_kind is SYNTHESIS."""
		mock_lm = MockGenerativeLocalVLLM([])
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		assert tot.generator.final_output_kind == "synthesis_faithful"


# =============================================================================
# constants.GPU Integration Tests: Tree of Thoughts with Real vLLM Models
# =============================================================================


@pytest.fixture(scope="module")
def vllm_model() -> Generator[GenerativeLocalVLLM, Any, Any]:
	"""
	Initialize a generative vLLM model for testing from local storage.

	This fixture is scoped to the module so the model is loaded once
	and shared across all tests in this module for efficiency.

	The model is initialized on constants.GPU 0 only (isolated environment).

	Yields:
		Initialized GenerativeLocalVLLM instance for generative tasks
	"""
	# Set CUDA_VISIBLE_DEVICES to constants.GPU 0 ONLY for generative model
	# This ensures it can only see and use constants.GPU 0
	logger.info(
		"Initializing generative vLLM model on GPU 0"
	)
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	# Initialize GenerativeLocalVLLM with the provided config
	lm = GenerativeLocalVLLM(
		model=os.path.join("/path/to/model_storage", OpenSourceModel.QWEN_3_30B_A3B_INSTRUCT_2507.value),
		tensor_parallel_size=1,
		gpu_memory_utilization=0.9,
		max_model_len=16_384,
		verbosity=Verbosity.INFO,
	)
	logger.info("Generative VLLM model initialized successfully on constants.GPU 0")
	# Set it as the global LM for DSPy
	dspy.settings.configure(lm=lm)
	yield lm


@pytest.fixture(scope="module")
def reranker_vllm_model() -> Generator[ScoringLocalVLLM, Any, Any]:
	"""
	Initialize a reranker vLLM model for testing from local storage.

	This fixture is scoped to the module so the model is loaded once
	and shared across all tests in this module for efficiency.

	Requires at least 2 GPUs (one for generative model, one for reranker model).
	The model is initialized on constants.GPU 1 only (isolated environment).

	NOTE: Each model is initialized with isolated constants.GPU visibility to prevent conflicts.
	The generative model uses constants.GPU 0, and the reranker model uses constants.GPU 1.

	Args:
		vllm_model: The generative vLLM model fixture (ensures it's initialized first)

	Yields:
		Initialized ScoringLocalVLLM instance for reranking/scoring tasks
	"""
	# Set CUDA_VISIBLE_DEVICES to GPU 1 ONLY for reranker model
	logger.info(f"Initializing {OpenSourceModel.QWEN_3_RERANKER_8B.value} on GPU 1")
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	lm = ScoringLocalVLLM(
		model=os.path.join(
			"/path/to/model_storage", OpenSourceModel.QWEN_3_RERANKER_8B.value
		),
		tensor_parallel_size=1,
		gpu_memory_utilization=0.9,
		max_model_len=16_384,
		verbosity=Verbosity.INFO,
	)
	logger.info("Reranker VLLM model initialized successfully")
	yield lm


# =============================================================================
# constants.GPU Test Data
# =============================================================================

SHORT_REASONING_LENGTH = ResponseLength(
	granularity="sentence",
	bounds=(1, 3),
)

EXTENDED_RESPONSE_LENGTH = ResponseLength(
	granularity="sentence",
	bounds=(5, 7),
)

WORD_BASED_THOUGHT_LENGTH = ResponseLength(
	granularity="word",
	bounds=(100, 300),
)


# =============================================================================
# constants.GPU Test Helper Functions
# =============================================================================


def create_mixed_layer_tree(
	input_data: Input,
	reasoning_field_name: str,
	continued_reasoning_nodes: list[dict],
	final_response_nodes: list[dict],
) -> Tree:
	"""
	Helper function to create a tree with a mixed layer containing both nodes
	that continue reasoning and nodes with final responses.

	Args:
		input_data: The input data for the root node
		reasoning_field_name: The name of the reasoning field from the signature
		continued_reasoning_nodes: List of dicts with "reasoning" (of type list[dict]) and
			optional "controller_interventions" (of type list[dict])
		final_response_nodes: List of dicts with "reasoning" (of type list[dict]) and "output"
			(of type dict[str, str])

	Returns:
		Tree with a mixed layer containing both reasoning and final response nodes
	"""
	tree = Tree(
		root=Node(index=0, layer=0, state=State(input=input_data)),
		nodes={0: Node(index=0, layer=0, state=State(input=input_data))},
		edges={},
	)

	# Add continued reasoning nodes (nodes without final outputs)
	for node_data in continued_reasoning_nodes:
		child_state = State(input=input_data, reasoning=node_data["reasoning"])

		# Create node without final output
		child_node = Node(
			index=len(tree.nodes),
			state=child_state,
			layer=1,
			parent_id=tree.root.index,
		)

		tree.nodes[child_node.index] = child_node
		tree.root.children_ids.append(child_node.index)

		tree.add_edge(
			source_index=tree.root.index,
			target_index=child_node.index,
			edge=Edge(reasoning_step=str(node_data["reasoning"])),
		)

	# Add final response nodes (nodes with outputs)
	for node_data in final_response_nodes:
		child_state = State(input=input_data)
		child_state.reasoning = node_data["reasoning"]
		child_state.output = node_data["output"]

		child_node = Node(
			index=len(tree.nodes),
			state=child_state,
			layer=1,
			parent_id=tree.root.index,
		)

		tree.nodes[child_node.index] = child_node
		tree.root.children_ids.append(child_node.index)

		tree.add_edge(
			source_index=tree.root.index,
			target_index=child_node.index,
			edge=Edge(reasoning_step=str(node_data["reasoning"])),
		)

	return tree


def create_argument_generation_controller_output(
	tools: dict[str, dspy.Tool],
	action_name: str,
	is_reranker: bool,
	subtopic: str | None = None,
	style: str | None = None,
	structure: str | None = None,
	n_samples_generation: int = 1,
) -> ControllerOutput:
	"""
	Create a ControllerOutput for the argument generation controller.

	Args:
		tools: Dictionary of available tools.
		action_name: Name of the action (tool name for generative, specific action for reranker).
		is_reranker: Whether this is for a Reranker controller (True) or Generative (False).
		subtopic: Subtopic choice (default: None).
		style: Style choice (default: None).
		structure: Structure choice (default: None).
		n_samples_generation: Number of samples to generate for this action.

	Returns:
		ControllerOutput configured for the intervention tool.
	"""
	assert action_name, "Action name must be provided."
	assert n_samples_generation >= 1, "n_samples_generation must be at least 1."

	action_arguments = {}
	if not is_reranker:
		if subtopic is not None:
			action_arguments["stock_issue"] = subtopic
		if style is not None:
			action_arguments["style"] = style
		if structure is not None:
			action_arguments["structure"] = structure

	assert action_name in tools, f"Action {action_name} not found in tools."
	# Assert that the action arguments are exactly what the tool expects
	assert set(action_arguments.keys()) == set(tools[action_name].args.keys()), (
		f"Action arguments do not match tool arguments for {action_name}.\n"
		f"Expected {tools[action_name].args.keys()}, got {action_arguments.keys()}"
	)
	chosen_tool = tools[action_name]
	intervention = chosen_tool.func(**action_arguments)

	return ControllerOutput.from_choice_dict(
		choice_dict={
			"action": action_name,
			"arguments": action_arguments,
			"considerations": (
				f"Chose to use {action_name} with arguments {action_arguments}."
			),
			"unique_action_response_count": n_samples_generation,
			"tool_descriptions": chosen_tool.description,
		},
		intervention=intervention,
	)


def create_argumentative_reasoning_intervention_tool(
	include_finish: bool = True
) -> dict[str, dspy.Tool]:
	"""Create expected tool for all three dimensions: subtopics, styles, and structures."""

	def tool_func(
		stock_issue: Literal["General Introduction", "Economic Impact", "Social Justice & Equity"],
		style: Literal["Figurative Language", "Statistical & Data-Driven"],
		structure: Literal[
			"Causal Reasoning",
			"Evidence & Support",
			"Contrast",
			"Chronological Sequence",
			"Conjunction",
			"Condition",
			"Cause",
			"Cause-Effect",
			"Problem-Solution",
		],
	) -> ReasoningIntervention:
		# Use the actual internal_reasoning and prefix from the fixtures
		# Combine internal_reasoning from both stock_issue and style
		stock_issue_internal_reasoning_map = {
			"General Introduction": "I should provide a general introduction to the topic. ",
			"Economic Impact": "I should focus on the economic and financial implications of this issue. ",
			"Social Justice & Equity": "I should consider the fairness and equity dimensions of this issue. ",
		}
		style_internal_reasoning_map = {
			"Figurative Language": "I should employ non-literal comparison to make abstract concepts vivid. ",
			"Statistical & Data-Driven": "I should use numbers and data to provide concrete, measurable support. ",
		}
		prefix_map = {
			"Causal Reasoning": "Therefore",
			"Cause": "Therefore",
			"Evidence & Support": "According to",
			"Contrast": "However",
			"Chronological Sequence": "First",
			"Conjunction": "Moreover",
			"Condition": "If",
			"Cause-Effect": "Consequently",
			"Problem-Solution": "To solve this",
		}
		combined_internal_reasoning = (
			stock_issue_internal_reasoning_map.get(stock_issue, "") + style_internal_reasoning_map.get(style, "")
		)
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning=combined_internal_reasoning,
			prefix=prefix_map.get(structure, ""),
		)

	tools = {
		DEFAULT_REASONING_INTERVENTION_TOOL_NAME: dspy.Tool(
			name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
			func=tool_func,
			desc="Test Tool",
		)
	}

	if include_finish:
		tools["FINISH"] = FINISH_TOOL

	return tools


def create_subtopic_intervention_tool() -> dict[str, dspy.Tool]:
	"""Create expected tool for only subtopic intervention."""

	def tool_func(
		stock_issue: Literal["General Introduction", "Economic Impact", "Social Justice & Equity"],
	) -> ReasoningIntervention:
		stock_issue_internal_reasoning_map = {
			"General Introduction": "I should provide a general introduction to the topic. ",
			"Economic Impact": "I should focus on the economic and financial implications of this issue. ",
			"Social Justice & Equity": "I should consider the fairness and equity dimensions of this issue. ",
		}
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning=stock_issue_internal_reasoning_map.get(stock_issue, ""),
			prefix="",
		)

	return {
		DEFAULT_REASONING_INTERVENTION_TOOL_NAME: dspy.Tool(
			name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
			func=tool_func,
			desc="Test Tool",
		)
	}


# =============================================================================
# constants.GPU Test Class
# =============================================================================


@pytestmark_gpu
class TestTreeOfThoughtsOnGPU:
	"""Integration tests for TreeOfThoughts with real vLLM models on constants.GPU.

	These tests require constants.GPU access and will be skipped if GPUs are not available.
	For reranker controller tests, at least 2 GPUs are required.

	constants.GPU Allocation:
	- Generative model (vllm_model fixture): constants.GPU 0 with isolated visibility
	- Reranker model (reranker_vllm_model fixture): constants.GPU 1 with isolated visibility
	- Each model is initialized with CUDA_VISIBLE_DEVICES set to only its constants.GPU
	  to prevent conflicts and ensure stable operation across all tests
	"""

	def _create_tree_with_reasoning_nodes(
		self,
		input_data: Input,
		reasoning_field_name: str,
		reasoning_steps: list[list[str]],
		has_final_outputs: bool = False,
		final_outputs: list[dict[str, str]] | None = None,
	) -> Tree:
		"""
		Helper method to create a tree with existing nodes and edges.

		Notably, this method supplies existing tree-structures rather than creating
		new ones (from the root node only).

		Args:
			input_data: The input data for the root node
			reasoning_field_name: The name of the reasoning field
			reasoning_steps: List of reasoning step lists, one per node to create
			has_final_outputs: Whether the nodes should have final outputs (for ORM evaluation)
			final_outputs: Optional list of output dictionaries for each node

		Returns:
			Tree with nodes containing the specified reasoning steps
		"""
		tree = Tree(
			root=Node(index=0, layer=0, state=State(input=input_data)),
			nodes={0: Node(index=0, layer=0, state=State(input=input_data))},
			edges={},
		)

		for i, steps in enumerate(reasoning_steps):
			# Create a child node with proper state
			child_state = State(input=input_data)

			# Store reasoning steps on the state.
			child_state.reasoning = [{reasoning_field_name: step} for step in steps]

			# If final outputs are provided, add them to the state
			if has_final_outputs and final_outputs and i < len(final_outputs):
				child_state.output = final_outputs[i]

			# Create the node manually instead of using create_child_node
			child_node = Node(
				index=len(tree.nodes),
				state=child_state,
				layer=1,
				parent_id=tree.root.index,
			)

			# Add to tree
			tree.nodes[child_node.index] = child_node
			tree.root.children_ids.append(child_node.index)

			# Add edge
			tree.add_edge(
				source_index=tree.root.index,
				target_index=child_node.index,
				edge=Edge(reasoning_step=str(steps)),
			)

		return tree

	@pytest.mark.parametrize(
		"generator_signature, evaluator_signature, thought_length, response_length, early_stopping_enabled, consider_reasoning_in_final_eval, seed",
		[
			pytest.param(
				QuestionAnsweringWithReasoning,
				None,
				SHORT_REASONING_LENGTH,
				EXTENDED_RESPONSE_LENGTH,
				True,
				True,
				17,
				id="simple_signature_with_constraints_default_evaluator",
			),
			pytest.param(
				"problem: str -> reasoning_step: str -> answer: str",
				None,
				None,
				None,
				False,
				True,
				42,
				id="string_signature_without_early_stopping",
			),
			pytest.param(
				GenerateArgumentWithReasoning,
				None,
				None,
				None,
				True,
				True,
				42,
				id="argument_signature_defaults",
			),
			pytest.param(
				SolveMathProblemWithReasoning,
				None,
				WORD_BASED_THOUGHT_LENGTH,
				EXTENDED_RESPONSE_LENGTH,
				True,
				False,
				123,
				id="math_signature_with_properties",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,
				None,
				None,
				None,
				True,
				True,
				999,
				id="default_evaluator_signature",
			),
			pytest.param(
				GenerateArgumentWithReasoning,
				ArgumentEvaluatorMultiDimensional,
				None,
				None,
				True,
				True,
				42,
				id="argument_evaluator_multi_dimensional",
			),
			pytest.param(
				GenerateArgumentWithReasoning,
				ArgumentEvaluatorSingleScore,
				None,
				None,
				True,
				False,
				123,
				id="argument_evaluator_single_score",
			),
		],
	)
	@pytestmark_gpu
	def test_init(
		self,
		generator_signature: type[ReasoningSignature] | str,
		evaluator_signature: dspy.Signature | None,
		thought_length: ResponseLength | None,
		response_length: ResponseLength | None,
		early_stopping_enabled: bool,
		consider_reasoning_in_final_eval: bool,
		seed: int,
		vllm_model: GenerativeLocalVLLM,
	) -> None:
		"""
		Test TreeOfThoughts initialization with various configurations.

		This test validates that the Tree-of-Thoughts system initializes correctly
		with different parameter combinations and that the core components (generator,
		evaluator, controller) are properly instantiated.

		Args:
			generator_signature: The signature for the generator
			evaluator_signature: Custom evaluator signature (dspy.Signature) or None to use defaults
			thought_length: Response length constraints for intermediate reasoning steps
			response_length: Response length constraints for final responses
			early_stopping_enabled: Whether to enable early stopping
			consider_reasoning_in_final_eval: Whether to include reasoning in final evaluation
			seed: Random seed for reproducibility
			vllm_model: The vLLM language model fixture (shared across tests)
		"""
		# Initialize the Tree-of-Thoughts instance with explicit keyword arguments
		tot = TreeOfThoughts(
			generator_signature=generator_signature,
			evaluator_signature=evaluator_signature,
			generative_lm=vllm_model,
			thought_length=thought_length,
			response_length=response_length,
			# TODO[P3]: Test controller tools.
			controller_type="generator",  	# Use generator controller for this test
			evaluator_type="generator",		# Use generator evaluator for this test
			early_stopping_enabled=early_stopping_enabled,
			consider_reasoning_in_final_eval=consider_reasoning_in_final_eval,
			seed=seed,
			verbosity=Verbosity.INFO,
		)

		# Validate that core components exist and are properly instantiated
		assert isinstance(tot.generator, TreeOfThoughtGenerator), (
			"Generator should be a TreeOfThoughtGenerator instance"
		)

		assert isinstance(tot.evaluator, TreeOfThoughtEvaluator), (
			"Evaluator should be a TreeOfThoughtEvaluator instance"
		)

		assert isinstance(tot.controller, TreeOfThoughtsController), (
			"Controller should be a TreeOfThoughtsController instance"
		)

		logger.info(
			f"✓ Successfully initialized ToT with signature: "
			f"{generator_signature if isinstance(generator_signature, str) else generator_signature.__name__}"
		)

	@pytest.mark.parametrize(
		"tree, generator_signature, n_samples_generation, expected_number_of_nodes_in_tree, expected_number_of_nodes_in_new_layer, controller_outputs",
		[
			# All continue reasoning cases
			pytest.param(
				root_only_tree({"question": "What is the capital of France?"}),  # tree
				QuestionAnsweringWithReasoning,  	# generator_signature
				1,  # n_samples_generation
				2,  # expected_number_of_nodes_in_tree = 2 (root + 1 new reasoning step)
				1,  # expected_number_of_nodes_in_new_layer = 1 (1 new reasoning step)
				[  # controller_outputs
					ControllerOutput(
						action="continue_reasoning",
						action_arguments={},
						tool_descriptions="",
						considerations="Testing generation",
						continue_reasoning=True,
						unique_action_response_count=1,
					)
				],
				id="simple_signature_branching_1_all_continue",
			),
			pytest.param(
				root_only_tree({"question": "What is the capital of France?"}),  # tree
				QuestionAnsweringWithReasoning,  # generator_signature
				2,  # n_samples_generation
				3,  # expected_number_of_nodes_in_tree = 3 (root + 2 new reasoning steps)
				2,  # expected_number_of_nodes_in_new_layer = 2 (2 new reasoning steps)
				[  # controller_outputs
					ControllerOutput(
						action="continue_reasoning",
						action_arguments={},
						tool_descriptions="",
						considerations="Testing generation",
						continue_reasoning=True,
						unique_action_response_count=2,
					)
				],
				id="simple_signature_branching_2_all_continue",
			),
			# All finish cases
			pytest.param(
				root_only_tree({"question": "What is the capital of France?"}),
				QuestionAnsweringWithReasoning,
				1,
				2,  # root + 1 final output
				1,
				[
					ControllerOutput(
						action="finish",
						action_arguments={},
						tool_descriptions="",
						considerations="Ready to finish",
						continue_reasoning=False,
						unique_action_response_count=1,
					)
				],
				id="simple_signature_finish_only",
			),
			# Mixed cases: some continue, some finish
			pytest.param(
				root_only_tree({"question": "What is the capital of France?"}),
				QuestionAnsweringWithReasoning,
				2,
				3,  # root + 2 nodes (1 continue, 1 finish)
				2,
				[
					ControllerOutput(
						action="continue_reasoning",
						action_arguments={},
						tool_descriptions="",
						considerations="Continue reasoning",
						continue_reasoning=True,
						unique_action_response_count=1,
					),
					ControllerOutput(
						action="finish",
						action_arguments={},
						tool_descriptions="",
						considerations="Finish now",
						continue_reasoning=False,
						unique_action_response_count=1,
					),
				],
				id="simple_signature_mixed_continue_finish",
			),
			# Argument generation with style/structure tools
			pytest.param(
				root_only_tree({"topic": "Renewable energy is good", "stance": "PRO"}),
				GenerateArgumentWithReasoning,
				1,
				2,  # root + 1 new claim
				1,
				[
					ControllerOutput.from_choice_dict(
						choice_dict={
							"action": "select_style_structure",
							"arguments": {
								"style": "Knowledge",
								"structure": "Cause",
							},
							"considerations": "Using Knowledge style with Cause structure",
							"unique_action_response_count": 1,
							"tool_descriptions": "Mock tool descriptions",
						},
						intervention=ReasoningIntervention(
							continue_reasoning=True,
							internal_reasoning="",
							prefix="",
						),
					)
				],
				id="argument_style_structure_continue",
			),
			pytest.param(
				root_only_tree({"topic": "Renewable energy is good", "stance": "PRO"}),
				GenerateArgumentWithReasoning,
				2,
				3,  # root + 2 new claims
				2,
				[
					ControllerOutput.from_choice_dict(
						choice_dict={
							"action": "select_style_structure",
							"arguments": {
								"style": "Trust",
								"structure": "Contrast",
							},
							"considerations": "Using Trust style with Contrast structure",
							"unique_action_response_count": 2,
							"tool_descriptions": "Mock tool descriptions",
						},
						intervention=ReasoningIntervention(
							continue_reasoning=True,
							internal_reasoning="",
							prefix="",
						),
					)
				],
				id="argument_style_structure_multiple",
			),
			# Argument generation: mixed style/structure + finish
			pytest.param(
				root_only_tree({"topic": "Renewable energy is good", "stance": "PRO"}),
				GenerateArgumentWithReasoning,
				2,
				3,  # root + 2 nodes (1 style/structure continue, 1 finish)
				2,
				[
					ControllerOutput.from_choice_dict(
						choice_dict={
							"action": "select_style_structure",
							"arguments": {
								"style": "Power",
								"structure": "Conjunction",
							},
							"considerations": "Using Power style with Conjunction structure",
							"unique_action_response_count": 1,
							"tool_descriptions": "Mock tool descriptions",
						},
						intervention=ReasoningIntervention(
							continue_reasoning=True,
							internal_reasoning="",
							prefix="",
						),
					),
					ControllerOutput(
						action="finish",
						action_arguments={},
						tool_descriptions="",
						considerations="Ready to finish argument",
						continue_reasoning=False,
						unique_action_response_count=1,
					),
				],
				id="argument_mixed_style_structure_and_finish",
			),
			# String signature with mixed outputs
			pytest.param(
				root_only_tree({"question": "What is the capital of France?"}),
				"question: str -> reasoning_step: str -> answer: str",
				2,
				3,  # root + 2 new reasoning steps
				2,
				[
					ControllerOutput(
						action="continue_reasoning",
						action_arguments={},
						tool_descriptions="",
						considerations="Continue reasoning",
						continue_reasoning=True,
						unique_action_response_count=1,
					),
					ControllerOutput(
						action="finish",
						action_arguments={},
						tool_descriptions="",
						considerations="Finish reasoning",
						continue_reasoning=False,
						unique_action_response_count=1,
					),
				],
				id="string_signature_mixed",
			),
		],
	)
	@pytestmark_gpu
	def test_generate_thoughts(
		self,
		tree: Tree,
		generator_signature: type[ReasoningSignature] | str,
		n_samples_generation: int,
		expected_number_of_nodes_in_tree: int,
		expected_number_of_nodes_in_new_layer: int,
		controller_outputs: list[ControllerOutput],
		vllm_model: GenerativeLocalVLLM,
	) -> None:
		"""
		Test the generate_thoughts method of TreeOfThoughts.

		This test validates that new reasoning steps are generated correctly for nodes
		in the frontier, with the expected tree structure and number of nodes.

		Args:
			tree: The initial tree structure (typically just a root node)
			generator_signature: The signature for the generator
			n_samples_generation: Number of candidate thoughts to generate per frontier node
			expected_number_of_nodes_in_tree: Expected total nodes after generation
			expected_number_of_nodes_in_new_layer: Expected nodes in the newly created layer
			vllm_model: The vLLM language model fixture (shared across tests)
		"""
		# Make a deep copy of the tree to avoid modifying the original
		tree_copy = tree.model_copy(deep=True)

		# Initialize the Tree-of-Thoughts instance
		tot = TreeOfThoughts(
			generator_signature=generator_signature,
			generative_lm=vllm_model,
			controller_type="generator",
			evaluator_type="generator",
			seed=42,
			verbosity=Verbosity.INFO,
		)

		# Get the initial frontier (should be just the root for these tests)
		max_layer = max([node.layer for node in tree_copy.nodes.values()])
		frontier = [
			node
			for node in tree_copy.nodes.values()
			if node.layer == max_layer and not node.is_pruned
		]

		logger.info(f"Initial frontier size: {len(frontier)}")
		logger.info(f"Generating {n_samples_generation} thoughts per frontier node...")

		# Set up controller outputs for frontier nodes from test parameters
		# Since we're testing generation directly, we need to mimic the controller's output
		# which tells the generator how many samples to generate for each action.
		for node in frontier:
			node.state.controller_outputs = controller_outputs

		# Create TreeOfThoughtsParameters for generation
		tot_parameters = TreeOfThoughtsParameters(
			n_samples_generation=n_samples_generation,
			generator_temperature=0.7,
			top_k=1,
			n_samples_judge=1,
			judge_temperature=0.0,
			depth=1,
			num_final_candidates=1,
		)

		# Generate new thoughts
		new_nodes = tot.generate_thoughts(
			frontier=frontier,
			tree=tree_copy,
			tot_parameters=tot_parameters,
		)

		# Assertions
		assert len(new_nodes) == expected_number_of_nodes_in_new_layer, (
			f"Expected {expected_number_of_nodes_in_new_layer} new nodes, but got {len(new_nodes)}"
		)

		assert len(tree_copy.nodes) == expected_number_of_nodes_in_tree, (
			f"Expected tree to have {expected_number_of_nodes_in_tree} nodes, but got {len(tree_copy.nodes)}"
		)

		# Validate new nodes have correct structure
		for node in new_nodes:
			# Each new node should have a parent
			assert node.parent_id is not None, (
				f"New node {node.index} should have a parent"
			)

			# Each new node should be in the next layer
			assert node.layer == max_layer + 1, (
				f"New node {node.index} should be in layer {max_layer + 1}, but is in layer {node.layer}"
			)

			# Check if this is a constants.FINISH action (has output, no continue_reasoning)
			# The controller_output_trajectory contains the ControllerOutput that created this node
			has_finish_action = (
				len(node.state.controller_output_trajectory) > 0
				and not node.state.controller_output_trajectory[-1].continue_reasoning
			)

			# For constants.FINISH actions, nodes may have output but no reasoning steps
			# For CONTINUE actions, nodes should have reasoning steps
			if has_finish_action:
				# constants.FINISH action: should have output
				assert len(node.state.output) > 0, (
					f"New node {node.index} with constants.FINISH action should have output"
				)
			else:
				# CONTINUE action: should have reasoning content
				assert len(node.state.reasoning) > 0, (
					f"New node {node.index} should have non-empty reasoning"
				)

				# Nodes that are still reasoning should not contain any output
				# Note: output defaults to empty dict {}, not None
				assert not node.state.output, (
					f"New node {node.index} should not have output when it is still reasoning"
				)

			# New nodes should not have scores yet (scoring happens in evaluate_thoughts)
			assert node.score is None, (
				f"New node {node.index} should not have a score yet"
			)

		logger.info(
			f"✓ Successfully generated {len(new_nodes)} thoughts "
			f"(total tree nodes: {len(tree_copy.nodes)})"
		)

	@pytest.mark.parametrize(
		# Parameter names
		[
			"signature",
			"nodes",
			"tot_parameters",
			"expected_new_nodes",
			"expected_prefixes",
		],
		# Parameter values
		[
			# Root-only generation cases
			pytest.param(
				QuestionAnsweringWithReasoning,  # signature
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(
							input={
								"question": "What is the capital of France?"
							},
							controller_outputs=[
								ControllerOutput(
									action="continue_reasoning",
									action_arguments={},
									tool_descriptions="",
									continue_reasoning=True,
									unique_action_response_count=2,
								)
							],
						),
					),
				],
				TreeOfThoughtsParameters(  # tot_parameters
					n_samples_generation=2,
					generator_temperature=0.7,
					top_k=1,
					n_samples_judge=1,
					judge_temperature=0.0,
					depth=1,
					num_final_candidates=1,
				),
				2,  # expected_new_nodes
				["", ""],  # expected_prefixes (no structure to check)
				id="root_continue_n2",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,  # signature
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(
							input={"question": "What is 2+2?"},
							controller_outputs=[
								ControllerOutput(
									action="finish",
									action_arguments={},
									tool_descriptions="",
									continue_reasoning=False,
									unique_action_response_count=1,
								)
							],
						),
					),
				],
				TreeOfThoughtsParameters(  # tot_parameters
					n_samples_generation=1,
					generator_temperature=0.7,
					top_k=1,
					n_samples_judge=1,
					judge_temperature=0.0,
					depth=1,
					num_final_candidates=1,
				),
				1,  # expected_new_nodes
				[""],  # expected_prefixes (constants.FINISH node - no prefix)
				id="root_finish_n1",
			),
			# Single frontier node with controller interventions
			pytest.param(
				GenerateArgumentWithReasoning,  # signature
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(
							input={
								"topic": "The US should ban social media for minors.",
								"stance": "PRO",
							}
						),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={
								"topic": "The US should ban social media for minors.",
								"stance": "PRO",
							},
							reasoning=[{"claim": "Social media harms mental health in children."}],
							controller_outputs=[
								create_argument_generation_controller_output(
									tools=create_argumentative_reasoning_intervention_tool(),
									action_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
									is_reranker=False,
									subtopic="Social Impact",
									style="Trust",
									structure="Contrast",
									n_samples_generation=3,
								)
							],
						),
					),
				],
				TreeOfThoughtsParameters(  # tot_parameters
					n_samples_generation=3,
					generator_temperature=0.7,
					top_k=1,
					n_samples_judge=1,
					judge_temperature=0.0,
					depth=1,
					num_final_candidates=1,
				),
				3,  # expected_new_nodes
				["However", "However", "However"],  # expected_prefixes (Contrast structure -> "However") (x3)
				id="single_frontier_subtopic_intervention_n3",
			),
			# Multiple frontier nodes
			pytest.param(
				GenerateArgumentWithReasoning,  # signature
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(
							input={
								"topic": "Universities should eliminate standardized test requirements.",
								"stance": "ANTI",
							}
						),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={
								"topic": "Universities should eliminate standardized test requirements.",
								"stance": "ANTI",
							},
							reasoning=[
								{"claim": "Standardized tests provide objective comparison."}
							],
							controller_outputs=[
								create_argument_generation_controller_output(
									tools=create_argumentative_reasoning_intervention_tool(),
									action_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
									is_reranker=False,
									subtopic="Cost-Benefit Analysis",
									style="Knowledge",
									structure="Cause",
									n_samples_generation=2,
								)
							],
						),
					),
					Node(
						index=2,
						layer=1,
						parent_id=0,
						state=State(
							input={
								"topic": "Universities should eliminate standardized test requirements.",
								"stance": "ANTI",
							},
							reasoning=[{"claim": "Alternative assessments are less reliable."}],
							controller_outputs=[
								create_argument_generation_controller_output(
									tools=create_argumentative_reasoning_intervention_tool(),
									action_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
									is_reranker=False,
									subtopic="Implementation & Enforcement",
									style="Power",
									structure="Condition",
									n_samples_generation=2,
								)
							],
						),
					),
				],
				TreeOfThoughtsParameters(  # tot_parameters
					n_samples_generation=2,
					generator_temperature=0.7,
					top_k=1,
					n_samples_judge=1,
					judge_temperature=0.0,
					depth=1,
					num_final_candidates=1,
				),
				4,  # expected_new_nodes (2 + 2)
				[
					"Therefore", "Therefore",		# Node 1: Cause -> Therefore (x2)
					"If", "If"						# Node 2: Condition -> If    (x2)
				],  # expected_prefixes
				id="multiple_frontier_mixed_n_samples",
			),
			# Deeper tree: layer 1 -> layer 2 generation
			pytest.param(
				GenerateArgumentWithReasoning,  # signature
				[  # nodes
					Node(
						index=0,
						layer=0,
						state=State(
							input={
								"topic": "The government should impose strict AI regulations.",
								"stance": "ANTI",
							}
						),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={
								"topic": "The government should impose strict AI regulations.",
								"stance": "ANTI",
							},
							reasoning=[{"claim": "AI regulation stifles innovation."}],
							controller_outputs=[
								create_argument_generation_controller_output(
									tools=create_argumentative_reasoning_intervention_tool(),
									action_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
									is_reranker=False,
									subtopic="Cost-Benefit & Impact Analysis",
									style="Status",
									structure="Conjunction",
									n_samples_generation=2,
								)
							],
						),
					),
				],
				TreeOfThoughtsParameters(  # tot_parameters
					n_samples_generation=2,
					generator_temperature=0.7,
					top_k=1,
					n_samples_judge=1,
					judge_temperature=0.0,
					depth=1,
					num_final_candidates=1,
				),
				2,  # expected_new_nodes
				["Moreover", "Moreover"],  # expected_prefixes (Conjunction -> Moreover)
				id="deeper_tree_layer2_generation_n2",
			),
			# Comprehensive deeper tree: layer 2 -> layer 3 generation with mixed n_samples
			pytest.param(
				GenerateArgumentWithReasoning,  # signature
				[  # nodes
					# Root (layer 0)
					Node(
						index=0,
						layer=0,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							}
						),
					),
					# Layer 1 - 3 nodes (1 pruned, 2 active)
					Node(
						index=1,
						layer=1,
						parent_id=0,
						is_pruned=True,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							},
							reasoning=[{"claim": "UBI reduces poverty (pruned)."}],
						),
					),
					Node(
						index=2,
						layer=1,
						parent_id=0,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							},
							reasoning=[{"claim": "UBI stimulates economic growth."}],
						),
					),
					Node(
						index=3,
						layer=1,
						parent_id=0,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							},
							reasoning=[{"claim": "UBI provides economic security."}],
						),
					),
					# Layer 2 - 6 nodes (3 from node 2 with different n_samples, 3 from node 3 with different n_samples)
					Node(
						index=4,
						layer=2,
						parent_id=2,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							},
							reasoning=[
								{"claim": "UBI stimulates economic growth."},
								{"claim": "Increased consumer spending boosts businesses."}
							],
							controller_outputs=[
								create_argument_generation_controller_output(
									tools=create_argumentative_reasoning_intervention_tool(),
									action_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
									is_reranker=False,
									subtopic="Cost-Benefit Analysis",
									style="Trust",
									structure="Cause",
									n_samples_generation=2,
								)
							],
						),
					),
					Node(
						index=5,
						layer=2,
						parent_id=2,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							},
							reasoning=[
								{"claim": "UBI stimulates economic growth."},
								{"claim": "Entrepreneurship increases with safety nets."}
							],
							controller_outputs=[
								create_argument_generation_controller_output(
									tools=create_argumentative_reasoning_intervention_tool(),
									action_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
									is_reranker=False,
									subtopic="Cost-Benefit & Impact Analysis",
									style="Power",
									structure="Contrast",
									n_samples_generation=1,
								)
							],
						),
					),
					Node(
						index=6,
						layer=2,
						parent_id=2,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							},
							reasoning=[
								{"claim": "UBI stimulates economic growth."},
								{"claim": "Service sector jobs increase."}
							],
							controller_outputs=[
								create_argument_generation_controller_output(
									tools=create_argumentative_reasoning_intervention_tool(),
									action_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
									is_reranker=False,
									subtopic="Economic Impact",
									style="Status",
									structure="Conjunction",
									n_samples_generation=1,
								)
							],
						),
					),
					Node(
						index=7,
						layer=2,
						parent_id=3,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							},
							reasoning=[
								{"claim": "UBI provides economic security."},
								{"claim": "Families can plan for the future."}
							],
							controller_outputs=[
								create_argument_generation_controller_output(
									tools=create_argumentative_reasoning_intervention_tool(),
									action_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
									is_reranker=False,
									subtopic="Economic Impact",
									style="Formal",
									structure="Cause-Effect",
									n_samples_generation=3,
								)
							],
						),
					),
					Node(
						index=8,
						layer=2,
						parent_id=3,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							},
							reasoning=[
								{"claim": "UBI provides economic security."},
								{"claim": "Mental health improves with stability."}
							],
							controller_outputs=[
								create_argument_generation_controller_output(
									tools=create_argumentative_reasoning_intervention_tool(),
									action_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
									is_reranker=False,
									subtopic="Social Impact",
									style="Persuasive",
									structure="Problem-Solution",
									n_samples_generation=2,
								)
							],
						),
					),
					Node(
						index=9,
						layer=2,
						parent_id=3,
						state=State(
							input={
								"topic": "Universal basic income should be implemented nationwide.",
								"stance": "PRO",
							},
							reasoning=[
								{"claim": "UBI provides economic security."},
								{"claim": "Education opportunities expand."}
							],
							controller_outputs=[
								ControllerOutput(
									action="finish",
									action_arguments={},
									tool_descriptions="",
									continue_reasoning=False,
									unique_action_response_count=1,
								)
							],
						),
					),
				],
				TreeOfThoughtsParameters(  # tot_parameters
					n_samples_generation=3,
					generator_temperature=0.7,
					top_k=1,
					n_samples_judge=1,
					judge_temperature=0.0,
					depth=1,
					num_final_candidates=1,
				),
				10,  # expected_new_nodes (2+1+1+3+2+1)
				[  # expected_prefixes (one per new node, matching n_samples above)
					"Therefore",  		# Node 4, Sample 1
					"Therefore",  		# Node 4, Sample 2
					"However",    		# Node 5, Sample 1
					"Moreover",   		# Node 6, Sample 1
					"Consequently", 	# Node 7, Sample 1
					"Consequently", 	# Node 7, Sample 2
					"Consequently", 	# Node 7, Sample 3
					"To solve this",	# Node 8, Sample 1
					"To solve this",	# Node 8, Sample 2
					"",             	# Node 9, Sample 1 (FINISH -> empty prefix)
				],
				id="comprehensive_deeper_tree_layer2_to_layer3_mixed_n_samples",
			),
		],
	)
	@pytest.mark.skipif(
		not pytest.importorskip("torch").cuda.is_available(),
		reason="GPU tests require GPU access",
	)
	def test_generate_thoughts_comprehensive(
		self,
		signature: type[ReasoningSignature] | str,
		nodes: list[Node],
		tot_parameters: TreeOfThoughtsParameters,
		expected_new_nodes: int,
		expected_prefixes: list[str],
		vllm_model: GenerativeLocalVLLM,
	) -> None:
		"""
		Comprehensive test for thought generation with controller interventions.

		Tests various scenarios:
		- Root-only generation (continue vs finish)
		- Single/multiple frontier nodes
		- Various n_samples values (1, 2, 3)
		- Controller interventions (subtopic/style/structure)
		- Deeper tree structures (layer 2+ generation)

		Args:
			signature: Generator signature
			nodes: Tree nodes (frontier nodes have controller_outputs)
			tot_parameters: Tree-of-Thoughts parameters for generation
			expected_new_nodes: Expected number of new nodes generated
			expected_prefixes: List of expected structural prefixes, one per new node (use empty string to skip checking)
			vllm_model: vLLM model fixture
		"""
		# Initialize Tree-of-Thoughts
		tot = TreeOfThoughts(
			generator_signature=signature,
			generative_lm=vllm_model,
			controller_type="generator",
			evaluator_type="generator",
			seed=42,
			verbosity=Verbosity.INFO,
		)

		# Build tree and extract frontier using helper function
		tree, frontier = build_tree_from_nodes(nodes)

		# Generate new thoughts
		new_nodes = tot.generate_thoughts(
			frontier=frontier,
			tree=tree,
			tot_parameters=tot_parameters,
		)

		# Assertions: verify total number of new nodes
		assert len(new_nodes) == expected_new_nodes, (
			f"Expected {expected_new_nodes} new nodes total, got {len(new_nodes)}"
		)

		# Calculate expected layer for verification
		expected_new_layer = max(node.layer for node in frontier) + 1
		reasoning_field_name = tot.reasoning_field_name

		# Verify all new nodes are in the correct layer
		for node in new_nodes:
			assert node.layer == expected_new_layer, (
				f"Node {node.index} should be in layer {expected_new_layer}, "
				f"but is in layer {node.layer}"
			)

			# Check if this is a constants.FINISH action
			has_finish_action = (
				len(node.state.controller_output_trajectory) > 0
				and not node.state.controller_output_trajectory[-1].continue_reasoning
			)

			if has_finish_action:
				# constants.FINISH action: should have output
				assert len(node.state.output) > 0, (
					f"Node {node.index} with constants.FINISH action should have output"
				)
			else:
				# CONTINUE action: should have reasoning content
				assert len(node.state.reasoning) > 0, (
					f"Node {node.index} should have non-empty reasoning"
				)

		# Verify structural prefixes in generated text (skip if empty string)
		assert len(expected_prefixes) == len(new_nodes), (
			f"Expected {len(expected_prefixes)} prefixes but got {len(new_nodes)} new nodes"
		)
		for node, expected_prefix in zip(new_nodes, expected_prefixes, strict=True):
			# Skip verification if expected_prefix is empty string
			if expected_prefix == "":
				continue

			# Only check nodes that have reasoning (not constants.FINISH nodes)
			if node.state.reasoning:
				latest_step_dict = node.state.reasoning[-1]
				# Extract the actual string value from the dict
				if isinstance(latest_step_dict, dict):
					latest_step = latest_step_dict.get(reasoning_field_name, "")
				else:
					latest_step = str(latest_step_dict)

				# Verify prefix appears in the generated step
				# Normalize candidates to a list of strings
				candidates: list[str] = (
					[expected_prefix]
					if isinstance(expected_prefix, str)
					else expected_prefix
				)

				# Filter out empty strings (which act as wildcards)
				# If "" is present, we skip strict checking effectively (or always return true)
				# However, let's treat "" as "always matches" explicitly
				matches_any = False
				for candidate in candidates:
					if candidate == "":
						matches_any = True
						break
					if latest_step.lower().strip().startswith(candidate.lower().strip()):
						matches_any = True
						break

					assert matches_any, (
						f"Node {node.index}: Expected generated text to start with one of {candidates}.\n"
						f"Generated: {latest_step}"
					)

		logger.info(
			f"✓ Successfully generated {len(new_nodes)} thoughts "
			f"across {len(frontier)} frontier nodes"
		)

	@pytest.mark.parametrize(
		"generator_signature, evaluator_signature, reasoning_steps_good, reasoning_steps_poor, n_samples_judge, judge_temperature",
		[
			pytest.param(
				QuestionAnsweringWithReasoning,
				None,
				[["Paris is the capital of France. This is a well-known fact."]],
				[],
				1,
				0.0,
				id="simple_single_good_step_default_evaluator",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,
				None,
				[["Paris is the capital of France."]],
				[["I don't know, maybe Berlin?"]],
				1,
				0.0,
				id="simple_good_vs_poor_step",
			),
			pytest.param(
				GenerateArgumentWithReasoning,
				ArgumentEvaluatorMultiDimensional,
				[
					["Renewable energy reduces carbon emissions significantly."],
					["Solar and wind power are increasingly cost-effective."],
				],
				[["Renewable energy is bad because reasons."]],
				1,
				0.0,
				id="argument_multi_dimensional_evaluator",
			),
			pytest.param(
				SolveMathProblemWithReasoning,
				None,
				[
					["Let's solve 2+2. First, we recognize this is addition."],
					["2 + 2 = 4, which is the correct answer."],
				],
				[["2+2 equals 5, I think."]],
				2,
				0.1,  # Use non-zero temperature when n_samples_judge > 1 (vLLM constraint)
				id="math_problem_multiple_judges",
			),
		],
	)
	@pytestmark_gpu
	def test_evaluate_thoughts_process_reward_model(
		self,
		generator_signature: type[ReasoningSignature] | str,
		evaluator_signature: dspy.Signature | None,
		reasoning_steps_good: list[list[str]],
		reasoning_steps_poor: list[list[str]],
		n_samples_judge: int,
		judge_temperature: float,
		vllm_model: GenerativeLocalVLLM,
	) -> None:
		"""
		Test evaluation of intermediate reasoning steps (PRM evaluation).

		This test validates that the evaluator correctly scores reasoning steps,
		with good reasoning steps receiving higher scores than poor ones.

		Args:
			generator_signature: The signature for the generator
			evaluator_signature: Custom evaluator signature or None for default
			reasoning_steps_good: List of good reasoning step sequences
			reasoning_steps_poor: List of poor reasoning step sequences
			n_samples_judge: Number of judge samples to use
			judge_temperature: Temperature for evaluation
			vllm_model: The vLLM language model fixture
		"""
		# Initialize Tree-of-Thoughts
		# Disable default demos to avoid field mismatches with test data
		tot = TreeOfThoughts(
			generator_signature=generator_signature,
			evaluator_signature=evaluator_signature,
			generative_lm=vllm_model,
			controller_type="generator",
			evaluator_type="generator",
			seed=42,
			verbosity=Verbosity.INFO,
		)

		# Get reasoning field name
		reasoning_field_name = tot.reasoning_field_name

		# Determine input based on signature
		if isinstance(generator_signature, str):
			input_data = {"question": "What is the capital of France?"}
		elif generator_signature == QuestionAnsweringWithReasoning:
			input_data = {"question": "What is the capital of France?"}
		elif generator_signature == GenerateArgumentWithReasoning:
			input_data = {
				"topic": "Renewable energy",
				"stance": "PRO",
			}  # Use enum, not .value
		elif generator_signature == SolveMathProblemWithReasoning:
			input_data = {"math_problem": "What is 2+2?"}
		else:
			input_data = {"question": "Test question"}

		# Create tree with reasoning nodes
		all_reasoning_steps = reasoning_steps_good + reasoning_steps_poor
		tree = self._create_tree_with_reasoning_nodes(
			input_data=input_data,
			reasoning_field_name=reasoning_field_name,
			reasoning_steps=all_reasoning_steps,
			has_final_outputs=False,
		)

		# Get frontier (all nodes except root)
		frontier = [node for node in tree.nodes.values() if node.index != 0]

		logger.info(f"Evaluating {len(frontier)} nodes with PRM...")

		# Create parameters
		tot_parameters = TreeOfThoughtsParameters(
			n_samples_judge=n_samples_judge,
			judge_temperature=judge_temperature,
			n_samples_generation=1,
			top_k=1,
			depth=1,
			num_final_candidates=1,
		)

		# Evaluate thoughts
		evaluated_nodes = tot.evaluate_thoughts(
			frontier=frontier,
			tree=tree,
			tot_parameters=tot_parameters,
		)

		# Assertions
		assert len(evaluated_nodes) == len(frontier), (
			f"Expected {len(frontier)} evaluated nodes, got {len(evaluated_nodes)}"
		)

		# All nodes should have scores and reasoning
		for node in evaluated_nodes:
			assert node.score is not None, f"Node {node.index} should have a score"
			assert 0.0 <= node.score <= 1.0, (
				f"Node {node.index} score {node.score} should be in [0, 1]"
			)
			assert node.state.reasoning, f"Node {node.index} should have reasoning steps in state"
			for step in node.state.reasoning:
				assert reasoning_field_name in step, (
					f"Node {node.index} reasoning step missing '{reasoning_field_name}': {step}"
				)

		# If we have both good and poor nodes, good should score higher
		if len(reasoning_steps_poor) > 0:
			num_good = len(reasoning_steps_good)
			good_nodes = evaluated_nodes[:num_good]
			poor_nodes = evaluated_nodes[num_good:]

			for good_node in good_nodes:
				for poor_node in poor_nodes:
					assert good_node.score > poor_node.score, (
						f"Good node {good_node.index} (score={good_node.score:.3f}) should score higher "
						f"than poor node {poor_node.index} (score={poor_node.score:.3f})\n"
						f"Good reasoning: {good_node.state.reasoning}\n"
						f"Poor reasoning: {poor_node.state.reasoning}"
					)

		logger.info(
			f"✓ Successfully evaluated {len(evaluated_nodes)} thoughts with PRM "
			f"(scores: {[f'{n.score:.3f}' for n in evaluated_nodes]})"
		)

	@pytest.mark.parametrize(
		"generator_signature, evaluator_signature, reasoning_chains_good, final_outputs_good, reasoning_chains_poor, final_outputs_poor, n_samples_judge, consider_reasoning_in_final_eval",
		[
			pytest.param(
				QuestionAnsweringWithReasoning,
				None,
				[["Paris is the capital of France."]],
				[{"answer": "Paris"}],
				[],
				[],
				1,
				True,
				id="simple_single_correct_answer",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,
				None,
				[["Paris is the capital of France."]],
				[{"answer": "Paris"}],
				[["I'm not sure about this."]],
				[{"answer": "Berlin"}],
				1,
				True,
				id="simple_correct_vs_incorrect_answer",
			),
			pytest.param(
				GenerateArgumentWithReasoning,
				ArgumentEvaluatorSingleScore,
				[["Renewable energy reduces carbon emissions and combats climate change."]],
				[{"argument": "Renewable energy is essential for a sustainable future."}],
				[["Energy is important."]],
				[{"argument": "Renewable energy is bad."}],
				1,  # Changed from 2 to 1 to avoid vLLM greedy sampling constraint
				False,
				id="argument_single_score_without_reasoning_context",
			),
			pytest.param(
				SolveMathProblemWithReasoning,
				None,
				[["2 + 2 equals 4."], ["The sum of 2 and 2 is 4."]],
				[{"answer": "4"}, {"answer": "4"}],
				[["I think it's 5."]],
				[{"answer": "5"}],
				1,
				True,
				id="math_multiple_correct_vs_incorrect",
			),
		],
	)
	@pytestmark_gpu
	def test_evaluate_thoughts_outcome_reward_model(
		self,
		generator_signature: type[ReasoningSignature] | str,
		evaluator_signature: dspy.Signature | None,
		reasoning_chains_good: list[list[str]],
		final_outputs_good: list[dict[str, str]],
		reasoning_chains_poor: list[list[str]],
		final_outputs_poor: list[dict[str, str]],
		n_samples_judge: int,
		consider_reasoning_in_final_eval: bool,
		vllm_model: GenerativeLocalVLLM,
	) -> None:
		"""
		Test evaluation of final outputs with reasoning chains (ORM evaluation).

		This test validates that the evaluator correctly scores final solutions,
		with correct answers receiving higher scores than incorrect ones.

		Args:
			generator_signature: The signature for the generator
			evaluator_signature: Custom evaluator signature or None for default
			reasoning_chains_good: List of good reasoning chains
			final_outputs_good: List of correct final outputs
			reasoning_chains_poor: List of poor reasoning chains
			final_outputs_poor: List of incorrect final outputs
			n_samples_judge: Number of judge samples to use
			consider_reasoning_in_final_eval: Whether to include reasoning in evaluation
			vllm_model: The vLLM language model fixture
		"""
		# Initialize Tree-of-Thoughts
		# Disable default demos to avoid field mismatches with test data
		tot = TreeOfThoughts(
			generator_signature=generator_signature,
			evaluator_signature=evaluator_signature,
			generative_lm=vllm_model,
			consider_reasoning_in_final_eval=consider_reasoning_in_final_eval,
			controller_type="generator",
			evaluator_type="generator",
			seed=42,
			verbosity=Verbosity.INFO,
		)

		# Get reasoning field name
		reasoning_field_name = tot.reasoning_field_name

		# Determine input based on signature
		if isinstance(generator_signature, str):
			input_data = {"question": "What is the capital of France?"}
		elif generator_signature == QuestionAnsweringWithReasoning:
			input_data = {"question": "What is the capital of France?"}
		elif generator_signature == GenerateArgumentWithReasoning:
			input_data = {
				"topic": "Renewable energy",
				"stance": "PRO",
			}  # Use enum, not .value
		elif generator_signature == SolveMathProblemWithReasoning:
			input_data = {"math_problem": "What is 2+2?"}
		else:
			input_data = {"question": "Test question"}

		# Create tree with final outputs
		all_reasoning_chains = reasoning_chains_good + reasoning_chains_poor
		all_final_outputs = final_outputs_good + final_outputs_poor
		tree = self._create_tree_with_reasoning_nodes(
			input_data=input_data,
			reasoning_field_name=reasoning_field_name,
			reasoning_steps=all_reasoning_chains,
			has_final_outputs=True,
			final_outputs=all_final_outputs,
		)

		# Get frontier (all nodes except root)
		frontier = [node for node in tree.nodes.values() if node.index != 0]

		logger.info(f"Evaluating {len(frontier)} nodes with ORM...")

		# Create parameters
		tot_parameters = TreeOfThoughtsParameters(
			n_samples_judge=n_samples_judge,
			judge_temperature=0.0,
			n_samples_generation=1,
			top_k=1,
			depth=1,
			num_final_candidates=1,
		)

		# Store original tree state for comparison
		original_node_count = len(tree.nodes)
		original_edge_count = len(tree.edges)

		# Evaluate thoughts
		evaluated_nodes = tot.evaluate_thoughts(
			frontier=frontier,
			tree=tree,
			tot_parameters=tot_parameters,
		)

		# Assertions
		assert len(evaluated_nodes) == len(frontier), (
			f"Expected {len(frontier)} evaluated nodes, got {len(evaluated_nodes)}"
		)

		# Tree structure should not change
		assert len(tree.nodes) == original_node_count, (
			f"Tree node count changed from {original_node_count} to {len(tree.nodes)}"
		)
		assert len(tree.edges) == original_edge_count, (
			f"Tree edge count changed from {original_edge_count} to {len(tree.edges)}"
		)

		# All nodes should have scores and reasoning
		for node in evaluated_nodes:
			assert node.score is not None, f"Node {node.index} should have a score"
			assert 0.0 <= node.score <= 1.0, (
				f"Node {node.index} score {node.score} should be in [0, 1]"
			)
			assert node.state.reasoning, f"Node {node.index} should have reasoning steps in state"
			for step in node.state.reasoning:
				assert reasoning_field_name in step, (
					f"Node {node.index} reasoning step missing '{reasoning_field_name}': {step}"
				)

		# If we have both good and poor nodes, good should score higher
		if len(reasoning_chains_poor) > 0:
			num_good = len(reasoning_chains_good)
			good_nodes = evaluated_nodes[:num_good]
			poor_nodes = evaluated_nodes[num_good:]

			for good_node in good_nodes:
				for poor_node in poor_nodes:
					assert good_node.score > poor_node.score, (
						f"Good node {good_node.index} (score={good_node.score:.3f}) should score higher "
						f"than poor node {poor_node.index} (score={poor_node.score:.3f})\n"
						f"Good output: {good_node.state.output}\n"
						f"Poor output: {poor_node.state.output}\n"
						f"Good reasoning: {good_node.state.reasoning}\n"
						f"Poor reasoning: {poor_node.state.reasoning}"
					)

		logger.info(
			f"✓ Successfully evaluated {len(evaluated_nodes)} final outputs with ORM "
			f"(scores: {[f'{n.score:.3f}' for n in evaluated_nodes]})"
		)


	@pytestmark_2_gpus
	def test_forward_reranker_controller_no_pruning(
		self,
		vllm_model: GenerativeLocalVLLM,
		reranker_vllm_model: ScoringLocalVLLM,
	) -> None:
		"""
		Test end-to-end forward pass with reranker controller and no pruning.

		This test validates the complete workflow with:
		- Reranker controller (requires 2 GPUs)
		- No pruning (all candidates explored)
		"""
		# Initialize Tree-of-Thoughts with reranker controller
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,
			generative_lm=vllm_model,
			reranker_lm=reranker_vllm_model,
			controller_type="reranker",
			seed=42,
		)

		# Create parameters with no pruning
		tot_parameters = TreeOfThoughtsParameters(
			depth=2,
			n_samples_generation=5,
			top_k=3,
			n_samples_judge=1,
			judge_temperature=0.0,
			generator_temperature=0.7,
			num_final_candidates=1,
			do_pruning=False,
		)

		input_data = {
			"topic": "Renewable energy should recieve more government funding",
			"stance": "PRO",
		}

		logger.info("Running forward pass with reranker controller + no pruning...")

		# Run forward pass
		output = tot.forward(
			state=input_data,
			tot_parameters=tot_parameters,
		)

		# Validate output
		assert len(output.responses) > 0, "Should have at least one response"
		assert len(output.tree.nodes) > 1, "Tree should have more than just root"
		assert output.runtime > 0, "Runtime should be positive"

		logger.info(
			f"✓ Successfully completed reranker controller + no pruning: "
			f"{len(output.responses)} responses, {len(output.tree.nodes)} nodes"
		)

	@pytestmark_2_gpus
	def test_evaluate_thoughts_prm_reranker_evaluator(
		self,
		vllm_model: GenerativeLocalVLLM,
		reranker_vllm_model: ScoringLocalVLLM,
	) -> None:
		"""PRM evaluation using reranker evaluator should score good reasoning > bad reasoning."""
		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=vllm_model,
			reranker_lm=reranker_vllm_model,
			controller_type="generator",
			evaluator_type="reranker",
			seed=42,
			verbosity=Verbosity.INFO,
		)

		input_data = {"question": "What is the capital of France?"}
		nodes = [
			Node(index=0, layer=0, state=State(input=input_data)),
			Node(
				index=1,
				layer=1,
				parent_id=0,
				state=State(
					input=input_data,
					reasoning=[{"reasoning_step": "Paris is the capital of France."}],
				),
			),
			Node(
				index=2,
				layer=1,
				parent_id=0,
				state=State(
					input=input_data,
					reasoning=[{"reasoning_step": "Berlin is the capital of France."}],
				),
			),
		]
		tree, frontier = build_tree_from_nodes(nodes)

		tot_parameters = TreeOfThoughtsParameters(
			n_samples_generation=1,
			top_k=1,
			depth=1,
			n_samples_judge=7,  # should be treated as 1 for reranker evaluator
			judge_temperature=0.7,
			num_final_candidates=1,
		)
		evaluated = tot.evaluate_thoughts(frontier=frontier, tree=tree, tot_parameters=tot_parameters)
		assert len(evaluated) == 2
		assert evaluated[0].score is not None and evaluated[1].score is not None
		assert evaluated[0].score > evaluated[1].score

	@pytestmark_2_gpus
	def test_evaluate_thoughts_orm_reranker_evaluator(
		self,
		vllm_model: GenerativeLocalVLLM,
		reranker_vllm_model: ScoringLocalVLLM,
	) -> None:
		"""ORM evaluation using reranker evaluator should score correct answer > incorrect."""
		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=vllm_model,
			reranker_lm=reranker_vllm_model,
			controller_type="generator",
			evaluator_type="reranker",
			consider_reasoning_in_final_eval=True,
			seed=42,
			verbosity=Verbosity.INFO,
		)

		input_data = {"question": "What is the capital of France?"}
		nodes = [
			Node(index=0, layer=0, state=State(input=input_data)),
			Node(
				index=1,
				layer=1,
				parent_id=0,
				state=State(
					input=input_data,
					reasoning=[{"reasoning_step": "Paris is the capital of France."}],
					output={"answer": "Paris"},
				),
			),
			Node(
				index=2,
				layer=1,
				parent_id=0,
				state=State(
					input=input_data,
					reasoning=[{"reasoning_step": "Berlin is the capital of France."}],
					output={"answer": "Berlin"},
				),
			),
		]
		tree, frontier = build_tree_from_nodes(nodes)

		tot_parameters = TreeOfThoughtsParameters(
			n_samples_generation=1,
			top_k=1,
			depth=1,
			n_samples_judge=3,  # should be treated as 1 for reranker evaluator
			judge_temperature=0.0,
			num_final_candidates=1,
		)
		evaluated = tot.evaluate_thoughts(frontier=frontier, tree=tree, tot_parameters=tot_parameters)
		assert len(evaluated) == 2
		assert evaluated[0].score is not None and evaluated[1].score is not None
		assert evaluated[0].score > evaluated[1].score

	@pytestmark_2_gpus
	def test_forward_reranker_controller_and_evaluator_no_pruning(
		self,
		vllm_model: GenerativeLocalVLLM,
		reranker_vllm_model: ScoringLocalVLLM,
	) -> None:
		"""End-to-end forward pass with reranker controller + reranker evaluator enabled."""
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,
			generative_lm=vllm_model,
			reranker_lm=reranker_vllm_model,
			controller_type="reranker",
			evaluator_type="reranker",
			seed=42,
		)

		tot_parameters = TreeOfThoughtsParameters(
			depth=1,
			n_samples_generation=3,
			top_k=2,
			n_samples_judge=5,  # ignored for reranker evaluator
			judge_temperature=0.7,
			generator_temperature=0.7,
			num_final_candidates=1,
			do_pruning=False,
		)

		input_data = {
			"topic": "Renewable energy should recieve more government funding",
			"stance": "PRO",
		}

		output = tot.forward(state=input_data, tot_parameters=tot_parameters)
		assert len(output.responses) > 0
		assert len(output.tree.nodes) > 1
		assert output.runtime > 0

	@pytestmark_gpu
	def test_forward_generative_controller_no_pruning(
		self,
		vllm_model: GenerativeLocalVLLM,
	) -> None:
		"""
		Test end-to-end forward pass with generative controller and no pruning.

		This test validates the complete workflow with:
		- Generative controller
		- No pruning (all candidates explored)
		"""
		# Initialize Tree-of-Thoughts with generative controller
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,
			generative_lm=vllm_model,
			controller_type="generator",
			evaluator_type="generator",
			seed=42,
		)

		# Create parameters with no pruning
		tot_parameters = TreeOfThoughtsParameters(
			depth=2,
			n_samples_generation=5,
			n_samples_judge=1,
			judge_temperature=0.0,
			generator_temperature=0.7,
			num_final_candidates=1,
			do_pruning=False,
		)

		input_data = {
			"topic": "Renewable energy should recieve more government funding",
			"stance": "PRO",
		}

		logger.info("Running forward pass with generative controller + no pruning...")

		# Run forward pass
		output = tot.forward(
			state=input_data,
			tot_parameters=tot_parameters,
		)

		# Validate output
		assert len(output.responses) > 0, "Should have at least one response"
		assert len(output.tree.nodes) > 1, "Tree should have more than just root"
		assert output.runtime > 0, "Runtime should be positive"

		logger.info(
			f"✓ Successfully completed generative controller + no pruning: "
			f"{len(output.responses)} responses, {len(output.tree.nodes)} nodes"
		)

	@pytestmark_2_gpus
	def test_forward_reranker_controller_default_config(
		self,
		vllm_model: GenerativeLocalVLLM,
		reranker_vllm_model: ScoringLocalVLLM,
	) -> None:
		"""
		Test end-to-end forward pass with reranker controller and default configuration.

		This test validates the complete workflow with:
		- Reranker controller (requires 2 GPUs)
		- Pruning enabled
		- 2 reasoning steps
		- n_samples_generation of 3
		- top_k of 2
		"""
		# Initialize Tree-of-Thoughts with reranker controller
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,
			generative_lm=vllm_model,
			reranker_lm=reranker_vllm_model,
			controller_type="reranker",
			max_reasoning_steps=2,
			early_stopping_enabled=False,
			seed=42,
		)

		# Create parameters with default configuration
		tot_parameters = TreeOfThoughtsParameters(
			depth=2,
			n_samples_generation=3,
			top_k=2,
			n_samples_judge=1,
			judge_temperature=0.0,
			generator_temperature=0.7,
			num_final_candidates=1,
			do_pruning=True,
		)

		input_data = {
			"topic": "Renewable energy should recieve more government funding",
			"stance": "PRO",
		}

		logger.info(
			"Running forward pass with reranker controller + default config "
			"(pruning=True, depth=2, n_samples_generation=3, top_k=2)..."
		)

		# Run forward pass
		output = tot.forward(
			state=input_data,
			tot_parameters=tot_parameters,
		)

		# Validate output
		assert len(output.responses) > 0, "Should have at least one response"
		assert len(output.tree.nodes) > 1, "Tree should have more than just root"
		assert output.runtime > 0, "Runtime should be positive"

		# With pruning, tree should be constrained
		# Max nodes:
		# 1 (root)
		# + 3 (layer 1, n_samples_generation=3) +
		# + 2*3 (layer 2, n_samples_generation=3, top_k=2) +
		# + 2*3 (final, n_samples_generation=n_final_responses_per_trajectory=3)
		max_expected_nodes = 1 + 3 + (2 * 3) + (2 * 3)
		assert len(output.tree.nodes) == max_expected_nodes, (
			f"With pruning, tree should have at most {max_expected_nodes} nodes, "
			f"got {len(output.tree.nodes)}"
		)
		logger.info(
			f"✓ Successfully completed reranker controller + default config: "
			f"{len(output.responses)} responses, {len(output.tree.nodes)} nodes"
		)

	@pytestmark_gpu
	def test_forward_generative_controller_with_pruning(
		self,
		vllm_model: GenerativeLocalVLLM,
	) -> None:
		"""
		Test end-to-end forward pass with generative controller and pruning.

		This test validates the complete workflow with:
		- Generative controller
		- Pruning enabled
		"""
		# Initialize Tree-of-Thoughts with generative controller
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,
			generative_lm=vllm_model,
			controller_type="generator",
			evaluator_type="generator",
			seed=42,
			verbosity=Verbosity.INFO,
		)

		# Create parameters with pruning
		tot_parameters = TreeOfThoughtsParameters(
			depth=2,
			n_samples_generation=3,
			top_k=2,
			n_samples_judge=1,
			judge_temperature=0.0,
			generator_temperature=0.7,
			num_final_candidates=1,
			do_pruning=True,
		)

		input_data = {
			"topic": "Renewable energy should recieve more government funding",
			"stance": "PRO",
		}

		logger.info("Running forward pass with generative controller + pruning...")

		# Run forward pass
		output = tot.forward(
			state=input_data,
			tot_parameters=tot_parameters,
		)

		# Validate output
		assert len(output.responses) > 0, "Should have at least one response"
		assert len(output.tree.nodes) > 1, "Tree should have more than just root"
		assert output.runtime > 0, "Runtime should be positive"

		logger.info(
			f"✓ Successfully completed generative controller + pruning: "
			f"{len(output.responses)} responses, {len(output.tree.nodes)} nodes"
		)

	@pytestmark_gpu
	def test_forward_no_early_stopping_default_tool_pruning(
		self,
		vllm_model: GenerativeLocalVLLM,
	) -> None:
		"""
		Test end-to-end forward pass with no early stopping, default tool, and pruning.

		This test validates the vanilla tree of thought setup:
		- No early stopping (early_stopping_enabled=False)
		- Default tool only (continue reasoning)
		- Pruning enabled
		- Tests return_action_if_single_option behavior
		"""
		# Define custom forced choice function that returns n=3 samples
		def forced_choice_repeater(available_tools, state):
			choices = return_action_if_single_option(available_tools, state)
			if choices and len(choices) == 1:
				return choices * 3
			return choices

		# Initialize Tree-of-Thoughts with default tool only and no early stopping
		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			generative_lm=vllm_model,
			controller_type="generator",
			evaluator_type="generator",
			early_stopping_enabled=False,
			forced_choice_function=forced_choice_repeater,
			seed=42,
		)

		# Create parameters with pruning
		tot_parameters = TreeOfThoughtsParameters(
			depth=2,
			n_samples_generation=3,
			top_k=1,
			n_samples_judge=1,
			judge_temperature=0.7,
			generator_temperature=0.7,
			n_final_responses_per_trajectory=None,
			num_final_candidates=1,
			do_pruning=True,
		)

		input_data = {"question": "What is the capital of France?"}

		logger.info(
			"Running forward pass with no early stopping + default tool + pruning "
			"(vanilla tree of thought)..."
		)

		# Run forward pass
		output = tot.forward(
			state=input_data,
			tot_parameters=tot_parameters,
		)

		# Validate output
		assert len(output.responses) > 0, "Should have at least one response"
		assert len(output.tree.nodes) > 1, "Tree should have more than just root"
		assert output.runtime > 0, "Runtime should be positive"

		# Verify that controller was called (should use return_action_if_single_option)
		# Check that nodes have controller interventions
		nodes_with_interventions = [
			node
			for node in output.tree.nodes.values()
			if len(node.state.controller_outputs) > 0
		]
		assert len(nodes_with_interventions) > 0, (
			"Should have nodes with controller interventions "
			"(return_action_if_single_option should be triggered)"
		)

		logger.info(
			f"✓ Successfully completed no early stopping + default tool + pruning: "
			f"{len(output.responses)} responses, {len(output.tree.nodes)} nodes"
		)

	@pytestmark_gpu
	def test_forward_structure_only_interventions(
		self,
		vllm_model: GenerativeLocalVLLM,
		temp_action_space_structures: Path,
	) -> None:
		"""
		Test end-to-end forward pass with structure-only interventions.

		This test validates the workflow with:
		- Structure-only action space
		- Default configuration (pruning, depth=2, n_samples_generation=3, top_k=2)
		"""
		# Initialize Tree-of-Thoughts with structure action space
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,
			generative_lm=vllm_model,
			controller_type="generator",
			evaluator_type="generator",
			action_space_paths=[temp_action_space_structures],
			max_reasoning_steps=2,
			seed=42,
			verbosity=Verbosity.INFO,
		)

		# Create parameters with default configuration
		tot_parameters = TreeOfThoughtsParameters(
			depth=2,
			n_samples_generation=3,
			top_k=2,
			n_samples_judge=1,
			judge_temperature=0.0,
			generator_temperature=0.7,
			num_final_candidates=1,
			do_pruning=True,
		)

		input_data = {
			"topic": "Renewable energy should recieve more government funding",
			"stance": "PRO",
		}

		logger.info(
			"Running forward pass with structure-only interventions + default config..."
		)

		# Run forward pass
		output = tot.forward(
			state=input_data,
			tot_parameters=tot_parameters,
		)

		# Validate output
		assert len(output.responses) > 0, "Should have at least one response"
		assert len(output.tree.nodes) > 1, "Tree should have more than just root"
		assert output.runtime > 0, "Runtime should be positive"

		# Verify that structure interventions were applied
		nodes_with_interventions = [
			node
			for node in output.tree.nodes.values()
			if len(node.state.controller_outputs) > 0
		]
		assert len(nodes_with_interventions) > 0, (
			"Should have nodes with structure-only controller interventions"
		)

		logger.info(
			f"✓ Successfully completed structure-only interventions: "
			f"{len(output.responses)} responses, {len(output.tree.nodes)} nodes"
		)

	@pytestmark_gpu
	def test_forward_topic_only_interventions(
		self,
		vllm_model: GenerativeLocalVLLM,
		temp_action_space_subtopics: Path,
	) -> None:
		"""
		Test end-to-end forward pass with topic-only (subtopic) interventions.

		This test validates the workflow with:
		- Subtopic-only action space
		- Default configuration (pruning, depth=2, n_samples_generation=3, top_k=2)
		"""
		# Initialize Tree-of-Thoughts with subtopic action space
		tot = TreeOfThoughts(
			generator_signature=GenerateArgumentWithReasoning,
			generative_lm=vllm_model,
			controller_type="generator",
			evaluator_type="generator",
			action_space_paths=[temp_action_space_subtopics],
			max_reasoning_steps=2,
			seed=42,
			verbosity=Verbosity.INFO,
		)

		# Create parameters with default configuration
		tot_parameters = TreeOfThoughtsParameters(
			depth=2,
			n_samples_generation=3,
			top_k=2,
			n_samples_judge=1,
			judge_temperature=0.0,
			generator_temperature=0.7,
			num_final_candidates=1,
			do_pruning=True,
		)

		input_data = {
			"topic": "Renewable energy should recieve more government funding",
			"stance": "PRO",
		}

		logger.info(
			"Running forward pass with topic-only (subtopic) interventions + default config..."
		)

		# Run forward pass
		output = tot.forward(
			state=input_data,
			tot_parameters=tot_parameters,
		)

		# Validate output
		assert len(output.responses) > 0, "Should have at least one response"
		assert len(output.tree.nodes) > 1, "Tree should have more than just root"
		assert output.runtime > 0, "Runtime should be positive"

		# Verify that subtopic interventions were applied
		nodes_with_interventions = [
			node
			for node in output.tree.nodes.values()
			if len(node.state.controller_outputs) > 0
		]
		assert len(nodes_with_interventions) > 0, (
			"Should have nodes with subtopic-only controller interventions"
		)

		logger.info(
			f"✓ Successfully completed topic-only (subtopic) interventions: "
			f"{len(output.responses)} responses, {len(output.tree.nodes)} nodes"
		)

# =============================================================================
# Test Class 6: Failed Parsing Handling
# =============================================================================


class TestFailedParsingHandling:
	"""Test handling of nodes with failed parsing."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"scenario",
			"reasoning_step",
			"existing_reasoning_steps",
			"expected_score",
			"expected_pruned",
			"expected_reasoning_fragment",
		],
		[
			(
				"failed_parsing",
				[],  # Empty reasoning steps for failed case (or dummy)
				[{"error": ExecutionError(error_type="parsing", error_message="Error")}],
				0.0,
				True,
				"failed to parse",
			),
			(
				"valid_parsing",
				[{"reasoning_step": "Valid step", "error": ExecutionError()}],
				[{"reasoning_step": "Valid step", "error": ExecutionError()}],
				0.8,  # Mocked score
				False,
				None,
			),
		],
	)
	def test_parsing_handling(
		self,
		scenario,
		reasoning_step,
		existing_reasoning_steps,
		expected_score,
		expected_pruned,
		expected_reasoning_fragment,
	):
		"""Test that nodes are handled correctly based on parsing status."""
		# Configure mocks
		mock_lm = MockGenerativeLocalVLLM()
		dspy.settings.configure(lm=mock_lm)

		tot = TreeOfThoughts(
			generator_signature=QuestionAnsweringWithReasoning,
			evaluator_signature=ArgumentEvaluatorSingleScore,
			generative_lm=mock_lm,
			controller_type="generator",
			evaluator_type="generator",
		)

		# Create node
		node = Node(
			index=0,
			layer=1,
			state=State(
				input={"question": "Question"},
				reasoning=existing_reasoning_steps or reasoning_step,
			),
		)

		# Mock evaluator response
		# If valid, we expect a score. If failed, evaluator shouldn't be called for this node,
		# but we mock it just in case logic is wrong and it IS called.
		mock_responses = [[["## overall_quality\n4"]]]
		mock_lm.set_responses(mock_responses)

		# Evaluate thoughts
		tot_params = DEFAULT_TOT_PARAMS
		scored_nodes = tot.evaluate_thoughts(
			frontier=[node],
			tree=Tree(state={"question": "Question"}),  # Dummy tree
			tot_parameters=tot_params,
		)

		# Verify results
		assert len(scored_nodes) == 1
		scored_node = scored_nodes[0]

		if expected_score == 0.0:
			assert scored_node.score == 0.0
		else:
			assert scored_node.score > 0

		assert scored_node.is_pruned == expected_pruned

		if expected_reasoning_fragment:
			assert expected_reasoning_fragment in scored_node.reasoning.lower()


if __name__ == "__main__":
	gpu_available = torch.cuda.is_available()
	if not gpu_available:
		# Run all tests except constants.GPU-specific ones
		logger.warning(
			"GPU not available. Running unit tests only. "
			"GPU integration tests will be skipped."
		)
		pytest.main(
		[
			__file__,
				"-k",
				"not TestTreeOfThoughtsOnGPU",
				"-v",  # Verbose test output
				"-s",  # Disable output capturing (show failures/errors immediately)
				"--tb=short",  # Shorter traceback format
				"--showlocals",  # Show local variables in tracebacks
				"--log-cli-level=INFO",  # Show INFO logs during test execution
			]
		)
	else:
		# If constants.GPU is available, run tests in a specific order to optimize constants.GPU usage:
		# 1. First run all unit tests (no constants.GPU models loaded)
		# 2. Then run single-constants.GPU tests (only generative model loaded)
		# 3. Finally run dual-constants.GPU tests (both models loaded)
		logger.info("=" * 80)
		logger.info("RUNNING UNIT TESTS (No constants.GPU models)")
		logger.info("=" * 80)
		pytest.main(
			[
				__file__,
				"-k",
				"not TestTreeOfThoughtsOnGPU",
			"-v",
			"-s",
			"--tb=short",
			"--showlocals",
			"--log-cli-level=INFO",
		]
		)
		logger.info("\n" + "=" * 80)
		logger.info("RUNNING SINGLE-constants.GPU TESTS (Generative model only)")
		logger.info("=" * 80)
		pytest.main(
			[
				__file__,
				"-k",
				"TestTreeOfThoughtsOnGPU and not reranker",
				"-v",
				"-s",
				"--tb=short",
				"--showlocals",
				"--log-cli-level=INFO",
			]
		)
		if _has_at_least_2_gpus():
			logger.info("\n" + "=" * 80)
			logger.info(
				"RUNNING DUAL-constants.GPU TESTS (Generative + Reranker models)"
			)
			logger.info("=" * 80)
			pytest.main(
				[
					__file__,
					"-k",
					"TestTreeOfThoughtsOnGPU and reranker",
					"-v",
					"-s",
					"--tb=short",
					"--showlocals",
					"--log-cli-level=INFO",
				]
			)
