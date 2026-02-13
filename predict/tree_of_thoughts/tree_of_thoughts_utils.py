"""Utility functions for formatting Tree of Thoughts logging output."""

# Standard library imports
from typing import Any

# Local imports
from misc_utils import LogColor


def format_controller_output(output: Any, index: int) -> str:
	"""Format a single controller output for logging.

	Parameters:
		output: ControllerOutput to format.
		index: Index of this output (1-based).

	Returns:
		Formatted string for logging.
	"""
	parts = []
	parts.append(f"{LogColor.CYAN}Output {index}{LogColor.RESET}:")

	# Always show action
	action_str = str(output.action)
	if output.continue_reasoning:
		parts.append(f"{LogColor.GREEN}action={action_str}{LogColor.RESET}")
	else:
		parts.append(f"{LogColor.YELLOW}action={action_str} (finish){LogColor.RESET}")

	# Show action_arguments if not empty
	if output.action_arguments:
		parts.append(f"args={output.action_arguments}")

	# Show considerations if meaningful (not "N/A" or empty)
	if output.considerations and output.considerations != "N/A":
		# Truncate long considerations
		considerations = output.considerations
		if len(considerations) > 150:
			considerations = considerations[:147] + "..."
		parts.append(f"{LogColor.DIM}considerations={considerations}{LogColor.RESET}")

	# Show internal_reasoning if present
	if output.internal_reasoning:
		reasoning = output.internal_reasoning
		if len(reasoning) > 100:
			reasoning = reasoning[:97] + "..."
		parts.append(f"{LogColor.DIM}reasoning={reasoning}{LogColor.RESET}")

	# Show prefix if present
	if output.prefix:
		parts.append(f"{LogColor.DIM}prefix='{output.prefix}'{LogColor.RESET}")

	# Show unique_action_response_count if > 1
	if output.unique_action_response_count > 1:
		parts.append(f"{LogColor.DIM}count={output.unique_action_response_count}{LogColor.RESET}")

	# ALWAYS show errors prominently if present
	has_error = bool(output.tool_execution_error or output.failed_tool)
	if has_error:
		# Join main parts first, then add error on new line
		main_line = " ".join(parts)
		error_parts = [f"{LogColor.RED}{LogColor.BOLD}ERROR:{LogColor.RESET}"]
		if output.failed_tool:
			error_parts.append(f"{LogColor.RED}  failed_tool={output.failed_tool}{LogColor.RESET}")
		if output.tool_execution_error:
			error_parts.append(f"{LogColor.RED}  error={output.tool_execution_error}{LogColor.RESET}")
		return main_line + "\n\t  " + " ".join(error_parts)

	return " ".join(parts)


def format_step_info(
	layer: int,
	node_index: int | None = None,
	message: str = "",
	score: float | None = None,
) -> str:
	"""Format a standardized step/log message.

	Format: [Step] Layer {layer}[, Node {node_index}]: {message} [score={score}]

	Parameters:
		layer: Current tree layer depth.
		node_index: Optional index of the node.
		message: The main log message.
		score: Optional score to display.

	Returns:
		Formatted log string.
	"""
	parts = [f"{LogColor.BOLD}[Step] Layer {layer}{LogColor.RESET}"]

	if node_index is not None:
		parts.append(f", Node {node_index}")

	parts.append(": ")
	parts.append(message)

	if score is not None:
		parts.append(f" score={score:.4f}")

	return "".join(parts)

