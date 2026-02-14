from typing import Any

import pytest
from pydantic import ValidationError

from predict.controller.controller_constants import ControllerOutput
from tree import (
	Edge,
	Node,
	ReasoningChain,
	State,
	Tree,
)
from tree.tree_constants import NodeField


def create_tree_from_parent_node(parent_node: Node) -> Tree:
	"""Create a tree with a parent node and all necessary ancestor nodes.

	If the parent node is the root (layer 0), creates a tree with just that node.
	If the parent node is an intermediate node (layer > 0), creates all ancestor
	nodes from root up to the parent's layer and links them properly.

	Args:
		parent_node: Parent node to include in the tree. If layer > 0, ancestor
			nodes will be created automatically.

	Returns:
		Tree instance with the parent node and all necessary ancestors properly
		linked.
	"""
	if parent_node.layer == 0:
		return Tree(root=parent_node)

	# Create ancestor nodes from root up to parent's layer
	root = Node(
		index=0,
		layer=0,
		state=State(input=parent_node.state.input),
	)
	nodes = [root]
	current_parent = root

	for layer_idx in range(1, parent_node.layer):
		ancestor = Node(
			index=layer_idx,
			layer=layer_idx,
			parent_id=current_parent.index,
			state=State(
				input=parent_node.state.input,
				reasoning=[
						{"reasoning_step": f"Step {layer_idx}"},
					],
			),
		)
		nodes.append(ancestor)
		current_parent = ancestor

	# Add the parent node with proper parent_id
	parent_node_copy = parent_node.model_copy(deep=True)
	if parent_node_copy.parent_id is None:
		parent_node_copy.parent_id = current_parent.index
	nodes.append(parent_node_copy)

	return Tree.from_nodes(nodes)


class TestStateModelOutputSoFar:
	"""Tests for State.model_output_so_far()."""

	@pytest.mark.parametrize(
		[
			"state",
			"expected",
		],
		[
			pytest.param(
				State(input={"question": "What is 2+2?"}),
				"",
				id="empty_state_returns_empty_string",
			),
			pytest.param(
				State(
					input={"question": "What is 2+2?"},
					reasoning=[{"reasoning_step": "First step"}],
				),
				(
					"""
<thinking>
<step>
## reasoning_step
First step
</step>
""".strip()
				),
				id="single_step_intermediate_output",
			),
			pytest.param(
				State(
					input={"question": "What is 2+2?"},
					reasoning=[
						{
							"reasoning_step": "First step",
							"error": {"error_type": "parsing", "error_message": "bad", "raw_output": ""},
						}
					],
				),
				(
					"""
<thinking>
<step>
## reasoning_step
First step
</step>
""".strip()
				),
				id="skips_error_metadata_in_reasoning_steps",
			),
			pytest.param(
				State(
					input={"question": "What is 2+2?"},
					controller_output_trajectory=[
						ControllerOutput(
							action="continue_reasoning",
							action_arguments={},
							tool_descriptions="Action Name: continue_reasoning",
							continue_reasoning=True,
							considerations="Test",
							internal_reasoning="Think carefully",
							prefix="",
						),
					],
					reasoning=[{"reasoning_step": "First step"}],
				),
				(
					"""
<thinking>
<step>
## internal_reasoning
Think carefully
## reasoning_step
First step
</step>
""".strip()
				),
				id="injects_internal_reasoning_from_controller_trajectory",
			),
			pytest.param(
				State(
					input={"question": "What is 2+2?"},
					controller_output_trajectory=[
						ControllerOutput(
							action="continue_reasoning",
							action_arguments={},
							tool_descriptions="Action Name: continue_reasoning",
							continue_reasoning=True,
							considerations="Test",
							internal_reasoning="Think carefully",
							prefix="",
						),
						ControllerOutput(
							action="finish",
							action_arguments={},
							tool_descriptions="Action Name: finish",
							continue_reasoning=False,
							considerations="Test",
							internal_reasoning="Now answer",
							prefix="",
						),
					],
					reasoning=[
						{"reasoning_step": "Step 1"},
						{"reasoning_step": "Step 2"},
					],
					output={"answer": "4"},
				),
				(
					"""
<thinking>
<step>
## internal_reasoning
Think carefully
## reasoning_step
Step 1
</step>
<step>
## internal_reasoning
Now answer
## reasoning_step
Step 2
</step>
</thinking>
<answer>
## answer
4
</answer>
""".strip()
				),
				id="final_output_includes_answer_section",
			),
		],
	)
	def test_model_output_so_far(self, state: State, expected: str) -> None:
		"""Ensure model_output_so_far() deterministically reconstructs assistant output."""
		assert state.model_output_so_far() == expected


class TestReasoningChainControllerTrajectory:
	"""Tests for ReasoningChain.controller_trajectory()."""

	@pytest.mark.parametrize(
		[
			"nodes",
			"expected_trajectory",
		],
		[
			pytest.param(
				[],
				[],
				id="empty_nodes_returns_empty_list",
			),
			pytest.param(
				[
					Node(
						index=0,
						layer=0,
						state=State(
							input={"topic": "Test"},
							controller_output_trajectory=[],
						),
					),
				],
				[],
				id="root_only_no_controller_outputs",
			),
			pytest.param(
				[
					Node(
						index=0,
						layer=0,
						state=State(input={"topic": "Test"}),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"topic": "Test"},
							controller_output_trajectory=[
								ControllerOutput(
									action="Causal Structures:causal_reasoning",
									action_arguments={"structure": "Causal Structures", "subtopic": "causal_reasoning"},
									tool_descriptions="",
									continue_reasoning=True,
								),
							],
							reasoning=[{"claim": "First claim"}],
						),
					),
				],
				[
					(
						"Causal Structures:causal_reasoning",
						{"structure": "Causal Structures", "subtopic": "causal_reasoning"},
					),
				],
				id="single_step_with_action_arguments",
			),
			pytest.param(
				[
					Node(
						index=0,
						layer=0,
						state=State(input={"topic": "Test"}),
					),
					Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"topic": "Test"},
							controller_output_trajectory=[
								ControllerOutput(
									action="step1",
									action_arguments={"a": 1},
									tool_descriptions="",
									continue_reasoning=True,
								),
								ControllerOutput(
									action="step2",
									action_arguments={"b": "x"},
									tool_descriptions="",
									continue_reasoning=False,
								),
							],
							reasoning=[{"claim": "1"}, {"claim": "2"}],
							output={"argument": "Final"},
						),
					),
				],
				[
					("step1", {"a": 1}),
					("step2", {"b": "x"}),
				],
				id="multi_step_trajectory",
			),
		],
	)
	def test_controller_trajectory(
		self,
		nodes: list[Node],
		expected_trajectory: list[tuple[str, dict[str, Any]]],
	) -> None:
		"""Verify controller_trajectory returns tool names and arguments per step."""
		chain = ReasoningChain.from_node_path(nodes)
		assert chain.controller_trajectory() == expected_trajectory


@pytest.mark.parametrize(
	# Parameter names
	[
		"init_kwargs",
		"expected_edges_reasoning",
		"expected_nodes_per_layer",
		"expected_exception",
		"expected_message_fragment",
	],
	# Parameter values
	[
		# ========== State-only initialization ==========
		pytest.param(
			{									# init_kwargs
				"state": {"question": "Why is the sky blue?"}
			},
			{},  								# expected_edges_reasoning
			[1],  								# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="state_only_creates_one_layer_tree_with_root_on_layer_0",
		),
		pytest.param(
			{  									# init_kwargs
				"state": {
					"claim": "Tree of thoughts is the best prompting method for LLMs.",
					"stance": "PRO",
				},
			},
			{},  								# expected_edges_reasoning
			[1],  								# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="state_only_complex_input",
		),
		# ========== Root-only initialization ==========
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
			},
			{},  								# expected_edges_reasoning
			[1],  								# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="root_only_valid_root",
		),
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=1,  					# Wrong index
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"index 0",  						# expected_message_fragment
			id="root_node_with_an_index_different_from_0_raises_validation_error",
		),
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=0,
					parent_id=1,  				# Has parent
					state=State(input={"question": "Why is the sky blue?"}),
				),
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"not have a parent",  				# expected_message_fragment
			id="root_node_with_a_parent_id_raises_validation_error",
		),
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step"}]
					),
				),
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"not contain reasoning",  			# expected_message_fragment
			id="root_with_reasoning_raises_validation_error",
		),
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						output={"answer": "Because"},
					),
				),
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"not contain a final response",  	# expected_message_fragment
			id="root_with_output_raises_validation_error",
		),
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=1,  					# Wrong layer
					state=State(input={"question": "Why is the sky blue?"}),
				),
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"layer 0",  						# expected_message_fragment
			id="root_with_a_layer_different_from_0_raises_validation_error",
		),
		# ========== Root + Nodes + Edges initialization ==========
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
				"edges": {},
			},
			{},  								# expected_edges_reasoning
			[1],  								# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="root_nodes_edges_empty_edges",
		),
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					1: Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[{"reasoning_step": "First step"}],
						),
					),
				},
				"edges": {
					0: {1: Edge(reasoning_step="First step")},
				},
			},
			{0: {1: "First step"}},  			# expected_edges_reasoning
			[1, 1],  							# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="root_nodes_edges_single_child",
		),
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					1: Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[{"reasoning_step": "Step 1"}],
						),
					),
					2: Node(
						index=2,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[{"reasoning_step": "Step 2"}],
						),
					),
					3: Node(
						index=3,
						layer=2,
						parent_id=1,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[
								{"reasoning_step": "Step 1"},
								{"reasoning_step": "Step 3"},
							],
						),
					),
				},
				"edges": {
					0: {
						1: Edge(reasoning_step="Step 1"),
						2: Edge(reasoning_step="Step 2"),
					},
					1: {3: Edge(reasoning_step="Step 3")},
				},
			},
			{  									# expected_edges_reasoning
				0: {1: "Step 1", 2: "Step 2"},
				1: {3: "Step 3"},
			},
			[1, 2, 1],  						# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="root_nodes_edges_multi_layer_tree",
		),
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					1: Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							output={"answer": "Because"},
						),
					),
				},
				"edges": {
					0: {1: Edge(reasoning_step="Final answer")},
				},
			},
			{0: {1: "Final answer"}},  			# expected_edges_reasoning
			[1, 1],  							# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="root_nodes_edges_with_final_output",
		),
		# ========== Nodes-only initialization ==========
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
			},
			{},  								# expected_edges_reasoning
			[1],  								# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="nodes_only_single_root",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					1: Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[{"reasoning_step": "Step"}],
						),
					),
				},
			},
			{},  								# expected_edges_reasoning
			[1, 1],  							# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="nodes_only_with_child",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					1: Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[{"reasoning_step": "Step 1"}],
						),
					),
					2: Node(
						index=2,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[{"reasoning_step": "Step 2"}],
						),
					),
					3: Node(
						index=3,
						layer=2,
						parent_id=1,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[
								{"reasoning_step": "Step 1"},
								{"reasoning_step": "Step 3"},
							],
						),
					),
				},
			},
			{},  								# expected_edges_reasoning
			[1, 2, 1],  						# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="nodes_only_complex_tree",
		),
		# ========== Nodes + Edges initialization ==========
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					1: Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[{"reasoning_step": "Step"}],
						),
					),
				},
				"edges": {
					0: {1: Edge(reasoning_step="Step")},
				},
			},
			{0: {1: "Step"}},  					# expected_edges_reasoning
			[1, 1],  							# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="nodes_edges_single_edge",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					1: Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[{"reasoning_step": "Step 1"}],
						),
					),
					2: Node(
						index=2,
						layer=1,
						parent_id=0,
						state=State(
							input={"question": "Why is the sky blue?"},
							reasoning=[{"reasoning_step": "Step 2"}],
						),
					),
				},
				"edges": {
					0: {
						1: Edge(reasoning_step="Step 1"),
						2: Edge(reasoning_step="Step 2"),
					},
				},
			},
			{  									# expected_edges_reasoning
				0: {1: "Step 1", 2: "Step 2"},
			},
			[1, 2],  							# expected_nodes_per_layer
			None,  								# expected_exception
			None,  								# expected_message_fragment
			id="nodes_edges_multiple_children",
		),
		# ========== State-only initialization error cases ==========
		pytest.param(
			{},  								# init_kwargs
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValueError,  						# expected_exception
			"must be provided",  				# expected_message_fragment
			id="state_only_empty_kwargs_raises",
		),
		pytest.param(
			{NodeField.STATE: {}},  						# init_kwargs
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValueError,  						# expected_exception
			"must be provided",  				# expected_message_fragment
			id="state_only_empty_dict_raises",
		),
		pytest.param(
			{NodeField.STATE: None},  					# init_kwargs
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValueError,  						# expected_exception
			"must be provided",  				# expected_message_fragment
			id="state_only_none_raises",
		),
		# ========== Root + Nodes + Edges initialization error cases ==========
		pytest.param(
			{  									# init_kwargs
				"root": Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				"nodes": "not_a_dict",  			# Invalid type
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"dictionary",  						# expected_message_fragment
			id="root_nodes_edges_invalid_nodes_type_raises",
		),
		# ========== Nodes-only initialization error cases ==========
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					1: Node(
						index=1,
						layer=1,
						parent_id=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"exactly one root node",  			# expected_message_fragment
			id="nodes_only_without_root_raises",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {},
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValueError,  						# expected_exception
			"must be provided",  				# expected_message_fragment
			id="nodes_only_empty_nodes_raises",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					1: Node(
						index=1,
						layer=0,  				# Multiple roots
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"exactly one root node",  			# expected_message_fragment
			id="nodes_only_multiple_roots_raises",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
					1: Node(
						index=1,
						layer=1,
						parent_id=None,  		# Missing parent
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
			},
			{},  								# expected_edges_reasoning
			[1, 1],  							# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"must have a parent_id",  			# expected_message_fragment
			id="non_root_node_missing_parent_raises_error",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					"not_int": Node(  			# Invalid key type
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"integer keys",  					# expected_message_fragment
			id="nodes_only_non_integer_keys_raises",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: "not_a_node",  			# Invalid value type
				},
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"Node instances",  					# expected_message_fragment
			id="nodes_only_non_node_values_raises",
		),
		# ========== Nodes + Edges initialization error cases ==========
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
				"edges": "not_a_dict",  			# Invalid type
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"dictionary",  						# expected_message_fragment
			id="nodes_edges_invalid_edges_type_raises",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
				"edges": {
					"not_int": {1: Edge(reasoning_step="step")},
				},
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"integer keys",  					# expected_message_fragment
			id="nodes_edges_non_integer_source_keys_raises",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
				"edges": {
					0: "not_a_dict",
				},
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"nested dictionary",  				# expected_message_fragment
			id="nodes_edges_non_dict_targets_raises",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
				"edges": {
					0: {"not_int": Edge(reasoning_step="step")},
				},
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			ValidationError,  					# expected_exception
			"integer keys",  					# expected_message_fragment
			id="nodes_edges_non_integer_target_keys_raises",
		),
		pytest.param(
			{  									# init_kwargs
				"nodes": {
					0: Node(
						index=0,
						layer=0,
						state=State(input={"question": "Why is the sky blue?"}),
					),
				},
				"edges": {
					0: {1: "not_an_edge"},
				},
			},
			None,  								# expected_edges_reasoning
			None,  								# expected_nodes_per_layer
			KeyError,  							# expected_exception
			"0",  								# expected_message_fragment
			id="nodes_edges_non_edge_values_raises",
		),
	],
)
def test_initialize_tree(
	init_kwargs: dict[str, object],
	expected_edges_reasoning: dict[int, dict[int, str]] | None,
	expected_nodes_per_layer: list[int] | None,
	expected_exception: type[Exception] | None,
	expected_message_fragment: str | None,
) -> None:
	"""
	Tree initializes from state or components and validates inputs comprehensively.

	Parameters:
		init_kwargs: Dictionary containing the initialization parameters.
			- 'state': Dictionary mapping input field names to values.
			- 'root': Root Node object (for root-only initialization).
			- 'nodes': Dictionary mapping node indices to Node objects.
			- 'edges': Optional nested dictionary of Edge objects between nodes.
		expected_edges_reasoning: Dictionary containing the expected edges and reasoning steps.
		expected_nodes_per_layer: List containing the expected number of nodes per layer.
		expected_exception: Type of exception to be raised.
		expected_message_fragment: Message fragment to be in the exception.
	"""
	if expected_exception:
		with pytest.raises(expected_exception) as exc_info:
			Tree(**init_kwargs)  # type: ignore[arg-type]
		assert expected_message_fragment is not None and expected_message_fragment in str(exc_info.value)
		return

	tree = Tree(**init_kwargs)  # type: ignore[arg-type]
	assert tree.root.layer == 0
	assert tree.root.index == 0
	if expected_edges_reasoning is not None:
		assert {
			src: {tgt: edge.reasoning_step for tgt, edge in targets.items()}
			for src, targets in tree.edges.items()
		} == expected_edges_reasoning
	if expected_nodes_per_layer is not None:
		assert len(tree.layers) == len(expected_nodes_per_layer)
		for layer_idx, expected_count in enumerate(expected_nodes_per_layer):
			assert len(tree.layers[layer_idx]) == expected_count
			for node in tree.layers[layer_idx]:
				assert node.layer == layer_idx


@pytest.mark.parametrize(
	# Parameter names
	[
		"parent_node",
		"controller_output",
		"generator_output",
		"expect_output",
		"expected_exception",
	],
	# Parameter values
	[
		# ========== Root node as parent ==========
		pytest.param(
			Node(								# parent_node
				index=0,
				layer=0,
				state=State(input={"question": "Why is the sky blue?"}),
			),
			ControllerOutput(  					# controller_output
				action="continue",
				action_arguments={},
				tool_descriptions="",
				continue_reasoning=True,
				unique_action_response_count=1,
			),
			{									# generator_output
				"reasoning_step": "First step"
			},
			False,  							# expect_output
			None,  								# expected_exception
			id="root_node_continues_reasoning_adds_step",
		),
		pytest.param(
			Node(								# parent_node
				index=0,
				layer=0,
				state=State(input={"question": "Why is the sky blue?"}),
			),
			ControllerOutput(  					# controller_output
				action="finish",
				action_arguments={},
				tool_descriptions="",
				continue_reasoning=False,
				unique_action_response_count=1,
			),
			{									# generator_output
				"answer": "Because."
			},
			True,  								# expect_output
			None,  								# expected_exception
			id="root_node_finish_stores_output",
		),
		pytest.param(
			Node(								# parent_node
				index=0,
				layer=0,
				state=State(
					input={"question": "Why is the sky blue?"},
					output={"answer": "Done"},
				),
			),
			ControllerOutput(  					# controller_output
				action="finish",
				action_arguments={},
				tool_descriptions="",
				continue_reasoning=False,
				unique_action_response_count=1,
			),
			{									# generator_output
				"answer": "New"
			},
			True,  								# expect_output
			ValidationError,  					# expected_exception
			id="root_node_with_output_raises",
		),
		# ========== Intermediate node as parent ==========
		pytest.param(
			Node(								# parent_node
				index=1,
				layer=1,
				parent_id=0,
				state=State(
					input={"question": "Why is the sky blue?"},
					reasoning=[{"reasoning_step": "Previous step"}],
				),
			),
			ControllerOutput(  					# controller_output
				action="continue",
				action_arguments={},
				tool_descriptions="",
				continue_reasoning=True,
				unique_action_response_count=1,
			),
			{									# generator_output
				"reasoning_step": "Second step"
			},
			False,  							# expect_output
			None,  								# expected_exception
			id="intermediate_node_continues_reasoning_adds_step",
		),
		pytest.param(
			Node(								# parent_node
				index=2,
				layer=2,
				parent_id=1,
				state=State(
					input={"question": "Why is the sky blue?"},
					reasoning=[{"reasoning_step": "Step 1"}, {"reasoning_step": "Step 2"}],
				),
			),
			ControllerOutput(  					# controller_output
				action="finish",
				action_arguments={},
				tool_descriptions="",
				continue_reasoning=False,
				unique_action_response_count=1,
			),
			{									# generator_output
				"answer": "Final answer"
			},
			True,  								# expect_output
			None,  								# expected_exception
			id="intermediate_node_finish_stores_output",
		),
		pytest.param(
			Node(								# parent_node
				index=1,
				layer=1,
				parent_id=0,
				state=State(
					input={"question": "Why is the sky blue?"},
					reasoning=[{"reasoning_step": "Step"}],
					output={"answer": "Done"},
				),
			),
			ControllerOutput(  					# controller_output
				action="finish",
				action_arguments={},
				tool_descriptions="",
				continue_reasoning=False,
				unique_action_response_count=1,
			),
			{									# generator_output
				"answer": "New"
			},
			True,  								# expect_output
			AssertionError,  					# expected_exception
			id="intermediate_node_with_output_raises",
		),
	],
)
def test_create_child_node(
	parent_node: Node,
	controller_output: ControllerOutput,
	generator_output: dict[str, Any],
	expect_output: bool,
	expected_exception: type[Exception] | None,
) -> None:
	"""create_child_node creates children from root or intermediate nodes and validates state.

	Tests that create_child_node correctly:
	- Links parent and child nodes (parent_id, children_ids, edges)
	- Sets child layer to parent layer + 1
	- Stores reasoning steps in State.reasoning when continue_reasoning=True
	- Stores final output when continue_reasoning=False
	- Adds controller_output to child's controller_output_trajectory
	- Raises AssertionError when parent already has output

	Works with both root nodes (layer 0) and intermediate nodes (layer > 0).
	"""
	if expected_exception:
		# Exception may occur during tree creation (e.g., ValidationError for invalid root nodes)
		# or during create_child_node (e.g., AssertionError for intermediate nodes with output)
		with pytest.raises(expected_exception):
			tree = create_tree_from_parent_node(parent_node)
			# If tree creation succeeded, exception should occur during create_child_node
			tree.create_child_node(
				parent_node=parent_node,
				output=generator_output,
				controller_output=controller_output,
			)
		return

	tree = create_tree_from_parent_node(parent_node)

	child = tree.create_child_node(
		parent_node=parent_node,
		output=generator_output,
		controller_output=controller_output,
	)

	assert child.parent_id == parent_node.index
	assert child.layer == parent_node.layer + 1
	assert child.index in tree.nodes
	assert parent_node.children_ids is not None
	assert child.index in parent_node.children_ids
	assert tree.edges[parent_node.index][child.index].reasoning_step
	if expect_output:
		assert child.state.output == generator_output
		assert controller_output in child.state.controller_output_trajectory
	else:
		reasoning_steps = child.state.reasoning
		assert any(
			step.get("reasoning_step") == generator_output["reasoning_step"]
			for step in reasoning_steps
		)


@pytest.mark.parametrize(
	# Parameter names
	[
		"nodes",
		"target_index",
		"expected_path_indices",
		"expected_exception",
	],
	# Parameter values
	[
		# ========== Root node path ==========
		pytest.param(
			{  									# nodes
				0: Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
			},
			0,  								# target_index
			[0],  								# expected_path_indices
			None,  								# expected_exception
			id="root_node_returns_single_node_path",
		),
		# ========== Linear chain paths ==========
		pytest.param(
			{  									# nodes
				# Root node
				0: Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				# Layer 1 nodes
				1: Node(
					index=1,
					layer=1,
					parent_id=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step1"}],
					),
				),
			},
			1,  								# target_index
			[0, 1],  							# expected_path_indices
			None,  								# expected_exception
			id="single_child_returns_root_to_child",
		),
		pytest.param(
			{  									# nodes
				# Root node
				0: Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				# Layer 1 nodes
				1: Node(
					index=1,
					layer=1,
					parent_id=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step1"}],
					),
				),
				# Layer 2 nodes
				2: Node(
					index=2,
					layer=2,
					parent_id=1,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[
								{"reasoning_step": "step1"},
								{"reasoning_step": "step2"},
							],
					),
				),
			},
			2,  								# target_index
			[0, 1, 2],  						# expected_path_indices
			None,  								# expected_exception
			id="two_level_chain_returns_full_path",
		),
		pytest.param(
			{  									# nodes
				# Root node
				0: Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				# Layer 1 nodes
				1: Node(
					index=1,
					layer=1,
					parent_id=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step1"}],
					),
				),
				# Layer 2 nodes
				2: Node(
					index=2,
					layer=2,
					parent_id=1,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step1"}, {"reasoning_step": "step2"}],
					),
				),
				# Layer 3 nodes
				3: Node(
					index=3,
					layer=3,
					parent_id=2,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[
							{"reasoning_step": "step1"},
							{"reasoning_step": "step2"},
							{"reasoning_step": "step3"},
						],
					),
				),
				# Layer 4 nodes
				4: Node(
					index=4,
					layer=4,
					parent_id=3,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[
							{"reasoning_step": "step1"},
							{"reasoning_step": "step2"},
							{"reasoning_step": "step3"},
							{"reasoning_step": "step4"},
						],
					),
				),
			},
			4,  									# target_index
			[0, 1, 2, 3, 4],  						# expected_path_indices
			None,  								# expected_exception
			id="deep_chain_returns_complete_path",
		),
		# ========== Intermediate node in chain ==========
		pytest.param(
			{  									# nodes
				0: Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				1: Node(
					index=1,
					layer=1,
					parent_id=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step1"}],
					),
				),
				2: Node(
					index=2,
					layer=2,
					parent_id=1,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[
							{"reasoning_step": "step1"},
							{"reasoning_step": "step2"},
						],
					),
				),
				3: Node(
					index=3,
					layer=3,
					parent_id=2,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[
							{"reasoning_step": "step1"},
							{"reasoning_step": "step2"},
							{"reasoning_step": "step3"},
						],
					),
				),
			},
			2,  									# target_index
			[0, 1, 2],  							# expected_path_indices
			None,  								# expected_exception
			id="intermediate_node_returns_path_to_intermediate",
		),
		# ========== Branching tree paths ==========
		pytest.param(
			{  									# nodes
				0: Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				1: Node(
					index=1,
					layer=1,
					parent_id=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step1"}],
					),
				),
				2: Node(
					index=2,
					layer=1,
					parent_id=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step2"}],
					),
				),
			},
			2,  									# target_index
			[0, 2],  								# expected_path_indices
			None,  								# expected_exception
			id="branching_tree_returns_path_to_sibling",
		),
		pytest.param(
			{  									# nodes
				0: Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
				1: Node(
					index=1,
					layer=1,
					parent_id=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step1"}],
					),
				),
				2: Node(
					index=2,
					layer=1,
					parent_id=0,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[{"reasoning_step": "step2"}],
					),
				),
				3: Node(
					index=3,
					layer=2,
					parent_id=1,
					state=State(
						input={"question": "Why is the sky blue?"},
						reasoning=[
							{"reasoning_step": "step1"},
							{"reasoning_step": "step1a"},
						],
					),
				),
			},
			3,  									# target_index
			[0, 1, 3],  							# expected_path_indices
			None,  								# expected_exception
			id="branching_tree_returns_path_to_nested_child",
		),
		# ========== Error cases ==========
		pytest.param(
			{  									# nodes
				0: Node(
					index=0,
					layer=0,
					state=State(input={"question": "Why is the sky blue?"}),
				),
			},
			99,  									# target_index
			[],  									# expected_path_indices
			KeyError,  							# expected_exception
			id="missing_target_raises_keyerror",
		),
	],
)
def test_get_path_to_node(
	nodes: dict[int, Node],
	target_index: int,
	expected_path_indices: list[int],
	expected_exception: type[Exception] | None,
) -> None:
	"""get_path_to_node returns root-to-target path and validates path structure.

	Tests that get_path_to_node correctly:
	- Returns path from root to target node (including both endpoints)
	- Handles root node (returns single-node path)
	- Handles linear chains of arbitrary depth
	- Handles intermediate nodes in chains
	- Handles branching trees (siblings and nested children)
	- Validates path structure (consecutive parent-child relationships, layer ordering)
	- Raises KeyError when target node doesn't exist in tree

	Args:
		nodes: Dictionary mapping node indices to Node objects forming a complete tree.
		target_index: Index of the target node to get path to.
		expected_path_indices: Expected list of node indices in path order.
		expected_exception: Expected exception type if error case.
	"""
	tree = Tree(nodes=nodes)  # type: ignore[call-arg]

	if expected_exception:
		with pytest.raises(expected_exception):
			tree.get_path_to_node(tree.nodes[target_index])
		return

	path = tree.get_path_to_node(tree.nodes[target_index])

	# Validate path indices
	assert [node.index for node in path] == expected_path_indices

	# Validate path structure: consecutive parent-child relationships
	for i in range(len(path) - 1):
		current = path[i]
		next_node = path[i + 1]
		assert next_node.parent_id == current.index, (
			f"Path broken at index {i}: node {current.index} -> "
			f"node {next_node.index} (parent_id={next_node.parent_id})"
		)
		assert next_node.layer == current.layer + 1, (
			f"Layer mismatch at index {i}: node {current.index} (layer "
			f"{current.layer}) -> node {next_node.index} (layer "
			f"{next_node.layer})"
		)

	# Validate path starts at root
	assert path[0].index == 0, f"Path should start at root (index 0), got {path[0].index}"
	assert path[0].layer == 0, f"Path should start at layer 0, got {path[0].layer}"

	# Validate path ends at target
	assert path[-1].index == target_index, (
		f"Path should end at target index {target_index}, got {path[-1].index}"
	)


@pytest.mark.parametrize(
	# Parameter names
	[
		"nodes",
		"expected_edges",
		"expected_children",
		"expected_exception",
		"expected_message",
	],
	# Parameter values
	[
		pytest.param(
			[										# nodes
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
							{"reasoning_step": "First step"},
							{"reasoning_step": "Second step"},
						]
					),
				),
			],
			{										# expected_edges
				0: {1: "Second step"},
			},
			{										# expected_children
				0: [1],
			},
			None,  								# expected_exception
			None,  								# expected_message
			id="reasoning_step_edge_and_children",
		),
		pytest.param(
			[									# nodes
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
							{"reasoning_step": "Prior step"},
							{"reasoning_step": "Latest step"},
						],
					),
				),
			],
			{									# expected_edges
				0: {1: "Latest step"},
			},
			{									# expected_children
				0: [1],
			},
			None,  								# expected_exception
			None,  								# expected_message
			id="existing_reasoning_edge",
		),
		pytest.param(
			[									# nodes
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
						output={"answer": "Because."},
					),
				),
			],
			{0: {1: "Because."}},				# expected_edges
			{0: [1]},							# expected_children
			None,  								# expected_exception
			None,  								# expected_message
			id="edge_from_output",
		),
		pytest.param(
			[									# nodes
				Node(index=0, layer=0, state=State(input={"question": "Why is the sky blue?"})),
				Node(
					index=1,
					layer=1,
					parent_id=None,
					state=State(input={"question": "Why is the sky blue?"}),
				),
			],
			None,  								# expected_edges
			None,  								# expected_children
			AssertionError,  						# expected_exception
			"must have a parent_id",  				# expected_message
			id="missing_parent_id_raises",
		),
		pytest.param(
			[										# nodes
				Node(index=0, layer=0, state=State(input={"question": "Why is the sky blue?"})),
				Node(
					index=1,
					layer=1,
					parent_id=42,
					state=State(input={"question": "Why is the sky blue?"}),
				),
			],
			None,  								# expected_edges
			None,  								# expected_children
			AssertionError,  					# expected_exception
			"not found",  						# expected_message
			id="missing_parent_node_raises",
		),
		pytest.param(
			[									# nodes
				Node(index=0, layer=0, state=State(input={"question": "Why is the sky blue?"})),
				Node(index=1, layer=0, state=State(input={"question": "Another?"})),
			],
			None,  								# expected_edges
			None,  								# expected_children
			AssertionError,  					# expected_exception
			"one root node",  					# expected_message
			id="multiple_roots_raise",
		),
	],
)
def test_from_nodes(
	nodes: list[Node],
	expected_edges: dict[int, dict[int, str]] | None,
	expected_children: dict[int, list[int]] | None,
	expected_exception: type[Exception] | None,
	expected_message: str | None,
) -> None:
	"""Tree.from_nodes assembles trees and validates inputs."""
	original_children = {node.index: list(node.children_ids or []) for node in nodes}

	if expected_exception:
		with pytest.raises(expected_exception) as exc_info:
			Tree.from_nodes(nodes)
		assert expected_message is not None and expected_message in str(exc_info.value)
		return

	tree = Tree.from_nodes(nodes)

	assert expected_edges is not None
	for parent, targets in expected_edges.items():
		for target, expected_reason in targets.items():
			assert tree.edges[parent][target].reasoning_step == expected_reason

	assert expected_children is not None
	for parent, children in expected_children.items():
		assert tree.nodes[parent].children_ids == children

	for node in nodes:
		assert node.children_ids == original_children[node.index]

if __name__ == "__main__":
	pytest.main(
		[
			__file__,
			"-v",  # Verbose test output
			"-s",  # Disable output capturing (show failures/errors immediately)
			"--tb=short",  # Shorter traceback format
			"--showlocals",  # Show local variables in tracebacks
			"--log-cli-level=INFO",  # Show INFO logs during test execution
		]
	)
