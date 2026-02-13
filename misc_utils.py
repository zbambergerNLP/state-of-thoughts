# Standard library imports
import ast
import enum
import logging
import os
import shutil
import subprocess
import types
from pathlib import Path
from dataclasses import is_dataclass, asdict
from datetime import datetime
from typing import Any, Literal, Union, get_args, get_origin

# Third-party imports
import pydantic
import torch


class ExecutionError(pydantic.BaseModel):
	"""
	Represents an error that occurred during execution.

	This model encapsulates errors from both generation (e.g., model refusals,
	context length exceeded) and parsing (e.g., invalid output format, missing fields).

	Attributes:
		error_type: The type of error ("generation" or "parsing"). None indicates no error.
		raw_output: The raw text output from the model (None if no output was generated).
		error_message: Human-readable description of the error. None indicates no error.
	"""
	error_type: Literal["generation", "parsing"] | None = None
	raw_output: str | None = None
	error_message: str | None = None

	def has_error(self) -> bool:
		"""Check if this represents an actual error (vs. successful completion)."""
		return self.error_type is not None

	def is_generation_error(self) -> bool:
		"""Check if this is a generation error."""
		return self.error_type == "generation"

	def is_parsing_error(self) -> bool:
		"""Check if this is a parsing error."""
		return self.error_type == "parsing"


# ANSI color codes for logging
class LogColor:
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def is_list_of_lists(lst: Any, inner_type: type[Any] | None = None) -> bool:
    """
    Check if the input is a list of lists.

    Parameters:
        lst (Any): The input to check.
        inner_type (type[Any]): The type of the inner elements of the lists.

    Returns:
        bool: True if the input is a list of lists, False otherwise.
    """
    if not isinstance(lst, list):
        return False
    # Empty list is not a list of lists
    if len(lst) == 0:
        return False
    # First check if all items are lists
    if not all(isinstance(item, list) for item in lst):
        return False
    # If no inner_type is specified, we just need all items to be lists
    if inner_type is None:
        return True
    # If inner_type is specified, check that all elements within the inner lists are of that type
    for item in lst:
        if not all(isinstance(element, inner_type) for element in item):
            return False
    return True


def format_list_of_fields(field_names: list[str]) -> str:
    """
    Format a list of field names into a single string.

    Parameters:
        field_names (List[str]): A list of field names.

    Returns:
        str: A single string that concatenates the list of strings.
    """
    assert len(field_names) > 0, "The list of strings must not be empty."
    # If the list is 1 item long, return the item as a string
    if len(field_names) == 1:
        return f"`{field_names[0]}`"
    # If the list is 2 items long, return the items as a string separated by "and"
    elif len(field_names) == 2:
        return f"`{field_names[0]}` and `{field_names[1]}`"
    # If the list is 3 or more items long, return the items as a string separated by commas and "and"
    else:
        return (
            ", ".join([f"`{field}`" for field in field_names[:-1]])
            + f", and `{field_names[-1]}`"
        )


def safe_parse_dict(value):
    """
    Recursively parses dictionary-like string values into dictionaries.

    Args:
        value (Any): A value that may be a dictionary, a dictionary-like
            string, or another type.

    Returns:
        Any: A dictionary with parsed sub-dictionaries if the input or any of
        its values are dicts or dictionary-like strings. If the input is not
        a dictionary or cannot be parsed as one, returns the original value.
    """
    if isinstance(value, dict):
        # Recursively parse all values in the dictionary
        return {k: safe_parse_dict(v) for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                # Recursively parse the parsed dictionary
                return safe_parse_dict(parsed)
        except (ValueError, SyntaxError):
            pass
    return value  # Return as-is if not a dict or parsable dict-like string


def parse_literal(text: str) -> str:
	"""
	Clean a string literal from model output by removing common prefixes and quotes.

	Removes:
	- Enumeration prefixes (e.g., "1.", "1)", "(1)", "1. ")
	- Bullet points (e.g., "-", "*", "•")

	Args:
		text: The raw text to clean.

	Returns:
		str: The cleaned string.
	"""
	if not isinstance(text, str):
		return text

	# Strip whitespace first
	text = text.strip()

	# Remove enumeration prefixes like "1. ", "(1) ", "1) ", "[1]"
	# Also handles bullet points like "-", "*", "•"
	if not text:
		return text

	# Check for bullet points at the start
	if text[0] in "-*•":
		text = text[1:].lstrip()
		return text

	# Check for number patterns at the start
	# Pattern: optional "(" or "[", followed by digits, followed by optional ")", "]", or "."
	prefix_end = 0

	# Check if starts with opening bracket or paren
	if text[0] in "([":
		prefix_end = 1

	# Find where digits end
	digit_start = prefix_end
	digit_end = digit_start
	while digit_end < len(text) and text[digit_end].isdigit():
		digit_end += 1

	# If we found digits, check for closing pattern
	if digit_end > digit_start:
		# Check if there's a closing character: ")", "]", or "."
		if digit_end < len(text) and text[digit_end] in ")].":
			# Found a match like "(1)", "[1]", or "1."
			prefix_end = digit_end + 1
		elif digit_end < len(text) and text[digit_end].isspace():
			# Found pattern like "1 " or "(1 " (number followed by space)
			prefix_end = digit_end

		# Remove the prefix and any trailing whitespace
		if prefix_end > 0:
			text = text[prefix_end:].lstrip()

	return text


def parse_reasoning_signature(
    input_field_names: list[str],
    output_field_names: list[str],
    reasoning_field_names: list[str],
) -> tuple[str, str, str]:
    """
    Parse the reasoning signature to extract the input, output, and reasoning fields.

    Args:
        input_field_names (List[str]): A list of input field names.
        output_field_names (List[str]): A list of output field names.
        reasoning_field_names (List[str]): A list of reasoning field names.

    Returns:
        tuple[str, str, str]: A tuple of the form `[input_fields, reasoning_fields, output_fields]`, where:
            - `input_fields` is a string (comma separated) representation of the input fields with backticks.
            - `reasoning_fields` is a string (comma separated) representation of the reasoning fields with backticks.
            - `output_fields` is a string (comma separated) representation of the output fields with backticks.
    """
    input_fields, output_fields = parse_base_signature(
        input_field_names=input_field_names,
        output_field_names=output_field_names,
    )
    reasoning_fields = format_list_of_fields(field_names=reasoning_field_names)
    return input_fields, reasoning_fields, output_fields


def parse_base_signature(
    input_field_names: list[str],
    output_field_names: list[str],
) -> tuple[str, str]:
    """
    Parse the base (generation) signature to extract the input, output, and instructions for the task.

    Create a formatted string representation of the input fields, output fields, and instructions from the base
    (generation) signature.

    Returns:
        Tuple[str, str]: A tuple of the form `[input_fields, output_fields]`, where:
        - `input_fields` is a string (comma separated) representation of the input fields with backticks.
        - `output_fields` is a string (comma separated) representation of the output fields with backticks.
    """
    input_fields = format_list_of_fields(field_names=input_field_names)
    output_fields = format_list_of_fields(field_names=output_field_names)
    return input_fields, output_fields


def create_model_config(**kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """
    Optimize model configuration based on GPU type and model size.

    Potential kwargs:
        * model_directory: The directory where the model is stored.
        * model_provider: The provider of the open source model (e.g., "unsloth", "Qwen", etc.).
        * model: The HuggingFace model name to use.
        * gpu: The GPU type to use for the model (e.g., "A100_40GB", "A100_80GB", "T4", etc.).
        * trust_remote_code: Whether to trust remote code when loading the model.
        * tensor_parallel_size: The number of GPUs to use for tensor parallelism (default is 1).
        * dtype: The data type to use for the model (e.g., "auto", "bfloat16", "float16").
        * gpu_memory_utilization: A float between 0 (low) and 1 (high) denoting how much GPU memory
            to use for the model.
        * task: The task to use the model for (e.g., "generate", "auto").
        * max_model_len: The maximum sequence length to capture for the model.
        * swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
        * cpu_offload_gb: The size (GiB) of CPU memory to use for offloading the model weights.
        * enforce_eager: Whether to enable eager execution mode in vLLM to avoid compilation issues.

    Returns:
        A dictionary of optimized configuration parameters
    """
    model_directory: str = kwargs.get("model_directory", "")  # type: ignore
    model_provider: str = kwargs.get("model_provider", "")  # type: ignore
    assert model_directory or model_provider, (
        "We require users to specify EITHER the path to a locally stored model "
        "OR the model provider that supplies the desired open-source model"
    )
    model_name: str = kwargs.get("model")  # type: ignore
    model_prefix: str = model_directory if model_directory else model_provider
    model_path = os.path.join(model_prefix, model_name)

    config = {
        "model": model_path,
        "gpu_memory_utilization": kwargs.get("gpu_memory_utilization"),
        "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
        "task": kwargs.get("task", "generate"),
        "max_model_len": kwargs.get("max_model_len", 2_048),
        "swap_space": kwargs.get("swap_space", 4),
        "cpu_offload_gb": kwargs.get("cpu_offload_gb", 2),
        "trust_remote_code": kwargs.get("trust_remote_code", True),
        "enforce_eager": kwargs.get("enforce_eager", False),
    }
    # TODO[P2]: Determine ideal memory utilization based on expected sequence lengths (input and
    # output), constants.GPU, and model choice.

    # Adjust based on model name and constants.GPU type
    dtype = kwargs.get("dtype")
    gpu: str = kwargs.get("gpu")  # type: ignore
    if dtype:
        config["dtype"] = dtype
    elif "A" in gpu:  # Ampere series Nvidia GPUs (A100-40GB, A100-80GB, A6000, etc.)
        # Only Ampere GPUs or newer allow bf16 quantization
        config["dtype"] = torch.bfloat16  # type: ignore
    elif "H" in gpu:  # Hopper series Nvidia GPUs (H100, H200, etc.)
        # Hopper GPUs support not only bf16 but also fp8 quantization.
        # We let vLLM choose the best dtype based on the model name (mapping to whether it was trained with fp8 or bf16).
        config["dtype"] = "auto"
    elif gpu == "T4" or "awq" in model_name.lower():
        # Older GPUs (before the Ampere generation) should use fp16 instead of bf16
        # Using AWQ quantization in vLLM requires fp16 quantization -- bf16 is not supported
        config["dtype"] = torch.float16  # type: ignore
    return config


def log_gpu_state(context: str, logger: logging.Logger | None = None) -> None:
    """Log GPU memory stats and nvidia-smi snapshot for debugging."""
    log = logger or logging.getLogger(__name__)
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        log.info("[%s] GPU not available; skipping GPU state log", context)
        return

    try:
        device = torch.cuda.current_device()
    except Exception:
        device = 0

    try:
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        log.info(
            "[%s] GPU memory (device %s): allocated=%.2fGB, reserved=%.2fGB, total=%.2fGB",
            context,
            device,
            allocated,
            reserved,
            total,
        )
    except Exception as mem_error:  # pragma: no cover - best effort logging
        log.warning("[%s] Unable to read torch GPU memory stats: %s", context, mem_error)

    smi_path = shutil.which("nvidia-smi")
    if not smi_path:
        log.debug("[%s] nvidia-smi not found on PATH", context)
        return

    try:
        result = subprocess.run(
            [
                smi_path,
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("[%s] nvidia-smi (total,used,free MB): %s", context, result.stdout.strip())
    except Exception as smi_error:  # pragma: no cover - best effort logging
        log.warning("[%s] Unable to run nvidia-smi: %s", context, smi_error)


def resolve_model_path(path: str | None, base_path: str | Path) -> str | None:
    """Resolve a possibly-relative model path against a base directory."""
    if not path:
        return None

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base_path).expanduser() / candidate

    candidate = candidate.resolve(strict=False)
    if not candidate.exists():
        raise FileNotFoundError(f"Model path not found: {candidate}")

    return str(candidate)


def stringify(obj: Any) -> str:
    """
    Convert an object to a string.

    Parameters:
        obj (Any): The object to convert.

    Returns:
        str: The string representation of the object.
    """
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        return "[\n" + "\n".join([f"\t{stringify(item)}," for item in obj]) + "\n]"
    elif isinstance(obj, set):
        if not obj:
            return r"{}"
        return "{\n" + "\n".join([f"\t{stringify(item)}," for item in obj]) + "\n}"
    elif isinstance(obj, tuple):
        if not obj:
            return "()"
        return "(\n" + "\n".join([f"\t{stringify(item)}," for item in obj]) + "\n)"
    elif isinstance(obj, dict):
        return _stringify_dict(obj)
    elif isinstance(obj, enum.Enum):
        return stringify(obj.value)
    elif isinstance(obj, pydantic.BaseModel):
        return _stringify_dict(obj.model_dump())
    else:  # Fallback for unexpected types
        return str(obj)


def _stringify_dict(obj: dict[str, Any]) -> str:
    """
    Convert a dictionary to a string.

    Parameters:
        obj (Dict[str, Any]): The dictionary to convert.

    Returns:
        str: The string representation of the dictionary.
    """
    if len(obj) == 1:  # Only one key-value pair in the dictionary
        return str(list(obj.values())[0])
    elif len(obj) > 1:
        # Ensure that if "reasoning" is present in the dictionary,
        # that it and its value are represented first in the string
        if "reasoning" in obj:
            reasoning: list[Any] = obj.pop("reasoning")
        else:
            reasoning = None
        result = "\n".join(
            [f"{stringify(key)}: {stringify(value)}" for key, value in obj.items()]
        )
        if reasoning:
            result = f"{"reasoning"}: {stringify(reasoning)}\n{result}"
        return result
    else:
        return ""


def extract_expected_types(type_annotation: type) -> list[type]:
    """
    Extracts all possible expected types from a type annotation, excluding None.

    Args:
        type_annotation (type): The type annotation to extract types from. Can be a single type or a Union.

    Returns:
        List[type]: A list of types extracted from the annotation, excluding NoneType.
    """
    if get_origin(type_annotation) is Union:
        args = get_args(type_annotation)
        return [arg for arg in args if arg is not type(None)]
    return [type_annotation]



def serialize_object(value: Any) -> Any:
    """
    Recursively serialize a given object into a JSON-compatible format.

    Handles: Path, datetime, Enum, dataclasses, pydantic models, custom objects with __dict__,
    lists, tuples, sets, and dicts.

    Args:
        value: Value to serialize

    Returns:
        JSON-serializable equivalent
    """
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump()
    elif isinstance(value, dict):
        return {k: serialize_object(v) for k, v in value.items()}
    elif isinstance(value, list | tuple):
        return [serialize_object(item) for item in value]
    elif isinstance(value, set):
        return [serialize_object(item) for item in value]
    elif isinstance(value, Path):
        return str(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, enum.Enum):
        return str(value)
    elif is_dataclass(value):
        return serialize_object(asdict(value))
    elif hasattr(value, "__dict__"):
        return serialize_object(vars(value))
    else:
        return value


def strip_metadata_from_obj(
    obj: Any,
    metadata_keys: set[str] | None = None,
) -> Any:
    """
    Recursively remove metadata keys from nested structures.

    This is used to ensure internal bookkeeping fields (e.g., `error`) are not included in
    human-facing string outputs (e.g., saved generations) while still being retained on nodes.

    Args:
        obj: Object to strip. Supports dict/list/tuple, and Pydantic models.
        metadata_keys: Keys to remove from dicts. Defaults to {"error"}.

    Returns:
        A copy of `obj` with metadata keys removed everywhere they appear.
    """
    if metadata_keys is None:
        metadata_keys = {"error"}

    if isinstance(obj, pydantic.BaseModel):
        return strip_metadata_from_obj(obj.model_dump(), metadata_keys=metadata_keys)

    if isinstance(obj, dict):
        return {
            k: strip_metadata_from_obj(v, metadata_keys=metadata_keys)
            for k, v in obj.items()
            if k not in metadata_keys
        }

    if isinstance(obj, list):
        return [strip_metadata_from_obj(v, metadata_keys=metadata_keys) for v in obj]

    if isinstance(obj, tuple):
        return tuple(strip_metadata_from_obj(v, metadata_keys=metadata_keys) for v in obj)

    return obj


def stringify_without_metadata(
    obj: Any,
    metadata_keys: set[str] | None = None,
) -> str:
    """
    Stringify an object after removing metadata keys (e.g., `error`).

    Args:
        obj: Object to stringify.
        metadata_keys: Keys to remove from dicts. Defaults to {"error"}.

    Returns:
        A string representation without metadata fields.
    """
    return stringify(strip_metadata_from_obj(obj, metadata_keys=metadata_keys))

def _is_optional_type(type_annotation: type) -> bool:
    """
    Check if a type annotation is optional (allows None).

    An optional type is a Union that includes NoneType, such as:
    - str | None (Python 3.10+ syntax)
    - Optional[str] (typing.Optional)
    - Union[str, None] (typing.Union)

    Args:
        type_annotation: The type annotation to check

    Returns:
        bool: True if the type annotation accepts None, False otherwise

    Examples:
        _is_optional_type(str) -> False
        _is_optional_type(str | None) -> True
        _is_optional_type(Optional[str]) -> True
        _is_optional_type(list[str] | None) -> True
        _is_optional_type(list[str]) -> False
    """
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)

    # Check if this is a Union type (including | syntax)
    if (
        origin is Union
        or isinstance(type_annotation, types.UnionType)
    ):
        # Check if NoneType is in the union
        return type(None) in args

    return False


def _is_instance_of_type(value: Any, type_annotation: type) -> bool:
    """
    Check if a value matches a type annotation, handling parameterized generics.

    This function recursively validates values against type annotations including:
    - Simple types: str, int, float, bool
    - Parameterized generics: list[str], dict[str, int], list[list[str]]
    - Union types: str | int, Optional[str]

    Args:
        value: The value to check
        type_annotation: The type annotation to validate against

    Returns:
        bool: True if value matches the type annotation, False otherwise

    Examples:
        _is_instance_of_type("hello", str) -> True
        _is_instance_of_type(["a", "b"], list[str]) -> True
        _is_instance_of_type([1, 2], list[str]) -> False
        _is_instance_of_type([["a"], ["b"]], list[list[str]]) -> True
    """
    # Handle NoneType
    if type_annotation is type(None):
        return value is None

    origin = get_origin(type_annotation)
    args = get_args(type_annotation)

    # Handle Literal (e.g., Literal["PRO", "ANTI"])
    if origin is Literal:
        return value in args

    # Handle Unions (including | syntax in Python 3.10+)
    if (
        origin is Union
        or isinstance(type_annotation, types.UnionType)
    ):
        return any(_is_instance_of_type(value, arg) for arg in args)

    # Handle parameterized collections
    if origin:
        if not isinstance(value, origin):
            return False

        if origin is list:
            return not args or all(_is_instance_of_type(v, args[0]) for v in value)

        if origin is dict:
            if len(args) == 2:
                kt, vt = args
                return all(
                    _is_instance_of_type(k, kt) and _is_instance_of_type(v, vt)
                    for k, v in (value or {}).items()
                )
            return isinstance(value, dict)

        if origin is tuple:
            if len(args) != len(value):
                return False
            return all(_is_instance_of_type(v, t) for v, t in zip(value, args, strict=True))

        # Default: check instance of origin only
        return isinstance(value, origin)

    # Handle typing aliases like List, Dict (no origin but still subscriptable)
    try:
        return isinstance(value, type_annotation)
    except TypeError:
        return False


def create_tool_descriptions(options_dict: dict[str, dict[str, str]]) -> dict[str, str]:
    """
    Create a dictionary of tool descriptions from a dictionary of tool options.

    This function extracts only the constants.DEFINITION values from tool option dictionaries.

    Args:
        options_dict: Dictionary with option names mapping to dictionaries containing
            constants.DEFINITION (or "definition") keys

    Returns:
        Dict mapping tool names to their descriptions
    """
    return {
        option_name: option_data["definition"] for option_name, option_data in options_dict.items()
    }


def format_controller_output(output: Any, index: int) -> str:
    """Format a single controller output for logging.

    Parameters:
        output: ControllerOutput to format.
        index: Index of this output (1-based).

    Returns:
        Formatted string for logging.
    """
    parts = [f"{LogColor.CYAN}Output {index}{LogColor.RESET}:"]

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


def format_generated_node_log(
    layer: int,
    node_index: int,
    parent_id: int | None,
    reasoning_steps_count: int,
    has_output: bool,
    reasoning_content: list[str] | None = None,
    output_text: str | None = None,
) -> str:
    """Format log for a generated node with syntax highlighting.

    Args:
        layer: Current layer number
        node_index: Index of the node
        parent_id: ID of parent node (or None)
        reasoning_steps_count: Number of reasoning steps
        has_output: Whether the node has a final output
        reasoning_content: Optional list of reasoning strings to display
        output_text: Optional final output text to display

    Returns:
        Formatted string for logging
    """
    parent_info = f"parent={parent_id}" if parent_id is not None else "parent=N/A"

    # Header line with bold step info
    header_parts = [
        f"{LogColor.BOLD}[Step] Layer {layer}, Node {node_index} ({parent_info}){LogColor.RESET}:",
        f"reasoning_steps={reasoning_steps_count},",
        f"has_output={has_output}"
    ]
    header = " ".join(header_parts)

    body = []

    # Format reasoning steps
    if reasoning_content:
        for i, step in enumerate(reasoning_content, 1):
            # Cyan for "Reasoning Step X" label
            body.append(f"\t{LogColor.CYAN}Reasoning Step {i}{LogColor.RESET}: {step}")

    # Format output
    if output_text:
        # Green for "Output:" label
        body.append(f"\t{LogColor.GREEN}Output:{LogColor.RESET}\n{output_text}")

    if not body:
        return header

    return header + "\n" + "\n".join(body)
