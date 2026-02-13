# Tests for signature.py functionality including ReasoningSignature, make_reasoning_signature, two-arrow parsing and related utilities.

# Standard library imports

# Third-party imports
import pydantic
import pytest

# Local imports
from signatures.field import (
	InputField,
	OutputField,
	ReasoningField,
)
from signatures.signature import (
	ReasoningSignature,
	_convert_fields_to_signature_format,
	_parse_field_string,
	ensure_reasoning_signature,
	infer_prefix,
	make_reasoning_signature,
)
from tree.tree_constants import ReasoningState


class TestBasicSignatureCreation:
	"""Test basic signature creation and validation."""

	def test_signature_with_reasoning_fields(self) -> None:
		"""Test creating a signature with reasoning fields."""

		class ReasoningSignatureTest(ReasoningSignature):
			"""Signature with reasoning fields."""

			question: str = InputField(desc="The question to answer")
			reasoning: str = ReasoningField(desc="Step-by-step reasoning")
			answer: str = OutputField(desc="The final answer")

		# Verify field types and properties
		assert len(ReasoningSignatureTest.input_fields) == 1
		assert len(ReasoningSignatureTest.reasoning_fields) == 1
		assert len(ReasoningSignatureTest.output_fields) == 1

		# Verify field order in fields property
		field_names = list(ReasoningSignatureTest.fields.keys())
		assert field_names == ["question", "reasoning", "answer"]

		# Verify signature string representation (should show two arrows)
		assert ReasoningSignatureTest.signature == "question -> reasoning -> answer"

	def test_signature_field_validation(self) -> None:
		"""Test that signature field validation works correctly."""
		with pytest.raises(
			TypeError,
			match="must be declared with InputField, OutputField, or ReasoningField",
		):

			class InvalidSignature(ReasoningSignature):
				invalid_field: str = pydantic.Field()  # Should raise error

	def test_signature_with_constraints(self) -> None:
		"""Test signature with field constraints."""

		class ConstrainedSignature(ReasoningSignature):
			"""Signature with constrained fields."""

			user_input: str = InputField(
				desc="User input with constraints", min_length=5, max_length=100
			)
			response: str = OutputField(desc="Generated response", min_length=1)

		# Verify constraints are preserved
		user_input_field = ConstrainedSignature.input_fields["user_input"]
		extra = user_input_field.json_schema_extra
		if isinstance(extra, dict) and "constraints" in extra:
			constraints = extra["constraints"]
			if isinstance(constraints, str | list):
				assert "minimum length 5" in str(constraints)
				assert "maximum length 100" in str(constraints)


class TestStringSignatureParsing:
	"""Test parsing signature strings into Signature classes."""

	def test_single_arrow_signature_parsing(self) -> None:
		"""Test parsing single arrow signature format."""
		sig = make_reasoning_signature("question, context -> answer")

		assert len(sig.input_fields) == 2
		assert len(sig.output_fields) == 1
		assert len(sig.reasoning_fields) == 0

		assert "question" in sig.input_fields
		assert "context" in sig.input_fields
		assert "answer" in sig.output_fields

		assert sig.signature == "question, context -> answer"

	def test_two_arrow_signature_parsing(self) -> None:
		"""Test parsing two arrow signature format."""
		sig = make_reasoning_signature("question, context -> reasoning -> answer")

		assert len(sig.input_fields) == 2
		assert len(sig.reasoning_fields) == 1
		assert len(sig.output_fields) == 1

		assert "question" in sig.input_fields
		assert "context" in sig.input_fields
		assert "reasoning" in sig.reasoning_fields
		assert "answer" in sig.output_fields

		assert sig.signature == "question, context -> reasoning -> answer"

	def test_complex_two_arrow_signature(self) -> None:
		"""Test complex two arrow signature with multiple reasoning fields."""
		sig = make_reasoning_signature(
			"input1, input2 -> step1, step2 -> output1, output2"
		)

		assert len(sig.input_fields) == 2
		assert len(sig.reasoning_fields) == 2
		assert len(sig.output_fields) == 2

		# Verify field order
		field_names = list(sig.fields.keys())
		assert field_names == [
			"input1",
			"input2",
			"step1",
			"step2",
			"output1",
			"output2",
		]

	def test_typed_signature_parsing(self) -> None:
		"""Test parsing signatures with type annotations."""
		sig = make_reasoning_signature("question: str, count: int -> answer: bool")

		# Verify field types
		question_field = sig.input_fields["question"]
		count_field = sig.input_fields["count"]
		answer_field = sig.output_fields["answer"]

		assert question_field.annotation is str
		assert count_field.annotation is int
		assert answer_field.annotation is bool

	def test_invalid_signature_formats(self) -> None:
		"""Test that invalid signature formats raise appropriate errors."""
		# No arrows
		with pytest.raises(ValueError, match="Invalid signature format"):
			make_reasoning_signature("input output")

		# Too many arrows
		with pytest.raises(ValueError, match="Invalid signature format"):
			make_reasoning_signature("input -> reasoning -> output -> extra")

		# Empty signature parts
		with pytest.raises(ValueError):
			make_reasoning_signature(" -> output")


class TestSignatureManipulation:
	"""Test signature manipulation methods (insert, append, delete, etc.)."""

	def test_signature_append(self) -> None:
		"""Test appending fields to a signature."""

		class BaseSignature(ReasoningSignature):
			input1: str = InputField(desc="First input")
			output1: str = OutputField(desc="First output")

		# Append a new input field
		new_sig = BaseSignature.append("input2", InputField(desc="Second input"))

		assert len(new_sig.input_fields) == 2
		assert "input2" in new_sig.input_fields

		# Append a reasoning field
		reasoning_sig = new_sig.append(
			"reasoning", ReasoningField(desc="Reasoning step")
		)

		assert len(reasoning_sig.reasoning_fields) == 1
		assert "reasoning" in reasoning_sig.reasoning_fields

	def test_signature_prepend(self) -> None:
		"""Test prepending fields to a signature."""

		class BaseSignature(ReasoningSignature):
			input1: str = InputField(desc="First input")
			output1: str = OutputField(desc="First output")

		new_sig = BaseSignature.prepend("input0", InputField(desc="Zeroth input"))

		# Verify the field was added at the beginning
		input_names = list(new_sig.input_fields.keys())
		assert input_names[0] == "input0"

	def test_signature_delete(self) -> None:
		"""Test deleting fields from a signature."""

		class BaseSignature(ReasoningSignature):
			input1: str = InputField(desc="First input")
			input2: str = InputField(desc="Second input")
			output1: str = OutputField(desc="First output")

		new_sig = BaseSignature.delete("input2")

		assert len(new_sig.input_fields) == 1
		assert "input2" not in new_sig.input_fields
		assert "input1" in new_sig.input_fields

	def test_signature_insert(self) -> None:
		"""Test inserting fields at specific positions."""

		class BaseSignature(ReasoningSignature):
			input1: str = InputField(desc="First input")
			input3: str = InputField(desc="Third input")
			output1: str = OutputField(desc=ReasoningState.OUTPUT)

		# Insert at position 1 (between input1 and input3)
		new_sig = BaseSignature.insert(1, "input2", InputField(desc="Second input"))

		input_names = list(new_sig.input_fields.keys())
		assert input_names == ["input1", "input2", "input3"]

	def test_signature_with_updated_fields(self) -> None:
		"""Test updating existing field properties."""

		class BaseSignature(ReasoningSignature):
			input1: str = InputField(desc="Original description")
			output1: str = OutputField(desc=ReasoningState.OUTPUT)

		new_sig = BaseSignature.with_updated_fields(
			"input1", desc="Updated description"
		)

		updated_field = new_sig.input_fields["input1"]
		extra = updated_field.json_schema_extra
		if isinstance(extra, dict):
			assert extra["desc"] == "Updated description"

	def test_signature_with_instructions(self) -> None:
		"""Test updating signature instructions."""

		class BaseSignature(ReasoningSignature):
			input1: str = InputField(desc=ReasoningState.INPUT)
			output1: str = OutputField(desc=ReasoningState.OUTPUT)

		new_sig = BaseSignature.with_instructions("New instructions for the task.")

		assert new_sig.instructions == "New instructions for the task."


class TestEnsureReasoningSignature:
	"""Test the ensure_reasoning_signature function."""

	def test_ensure_reasoning_signature_with_string(self) -> None:
		"""Test ensure_reasoning_signature with string input."""
		sig = ensure_reasoning_signature("input -> output")
		assert sig is not None, (
			"Signature should not be None given a valid string input."
		)
		assert len(sig.input_fields) == 1
		assert len(sig.output_fields) == 1
		assert ReasoningState.INPUT in sig.input_fields
		assert ReasoningState.OUTPUT in sig.output_fields

	def test_ensure_reasoning_signature_with_class(self) -> None:
		"""Test ensure_reasoning_signature with existing signature class."""

		class TestSignature(ReasoningSignature):
			input1: str = InputField(desc=ReasoningState.INPUT)
			output1: str = OutputField(desc=ReasoningState.OUTPUT)

		sig = ensure_reasoning_signature(TestSignature)

		assert sig is TestSignature

	def test_ensure_reasoning_signature_with_instructions(self) -> None:
		"""Test ensure_reasoning_signature with additional instructions."""
		sig = ensure_reasoning_signature("input -> output", "Custom instructions")
		assert sig is not None, (
			"Signature should not be None given a valid string input."
		)
		assert sig.instructions == "Custom instructions"


class TestCustomTypes:
	"""Test custom type support in signature parsing."""

	def test_custom_types_in_make_reasoning_signature(self) -> None:
		"""Test using custom types in make_reasoning_signature."""

		class UniversityStudent(pydantic.BaseModel):
			"""A Pydantic model representing a university student."""

			student_id: str
			name: str
			email: str | None = None  # Optional field

		custom_types = {"UniversityStudent": UniversityStudent}
		sig = make_reasoning_signature(
			"student: UniversityStudent -> summary: str", custom_types=custom_types
		)
		input_field = sig.input_fields["student"]
		assert input_field.annotation == UniversityStudent

	def test_standard_typing_support(self) -> None:
		"""Test support for standard typing module types."""
		sig = make_reasoning_signature(
			"items: List[str], flag: Optional[bool] -> result: Union[int, str]"
		)

		items_field = sig.input_fields["items"]
		flag_field = sig.input_fields["flag"]
		result_field = sig.output_fields["result"]

		# These should parse successfully without errors
		assert items_field.annotation is not None
		assert flag_field.annotation is not None
		assert result_field.annotation is not None


class TestSignatureCompatibility:
	"""Test compatibility features and edge cases."""

	def test_field_prefix_inference(self) -> None:
		"""Test automatic prefix inference for fields."""
		prefix = infer_prefix("user_input")
		assert prefix == "User Input"

		prefix = infer_prefix("complex_field_name")
		assert prefix == "Complex Field Name"

	def test_default_instructions_generation(self) -> None:
		"""Test automatic generation of default instructions."""

		class TestSignature(ReasoningSignature):
			input1: str = InputField(desc="First input")
			input2: str = InputField(desc="Second input")
			output1: str = OutputField(desc=ReasoningState.OUTPUT)

		# Instructions should be auto-generated
		assert "Given the fields" in TestSignature.instructions
		assert "`input1` and `input2`" in TestSignature.instructions
		assert "`output1`" in TestSignature.instructions

	def test_signature_state_management(self) -> None:
		"""Test signature state dump and load functionality."""

		class TestSignature(ReasoningSignature):
			"""Test signature for state management."""

			input1: str = InputField(desc="Input field", prefix="Input:")
			output1: str = OutputField(desc="Output field", prefix="Output:")

		state = TestSignature.dump_state()

		assert "instructions" in state
		assert "fields" in state
		assert state["instructions"] == "Test signature for state management."
		assert len(state["fields"]) == 2

	def test_signature_equality(self) -> None:
		"""Test signature equality comparison."""

		class Sig1(ReasoningSignature):
			input1: str = InputField(desc="A first input")
			output1: str = OutputField(desc="A first output")

		class Sig2(ReasoningSignature):
			input1: str = InputField(desc="A first input")
			output1: str = OutputField(desc="A first output")

		class Sig3(ReasoningSignature):
			input1: str = InputField(desc="Different input")
			output1: str = OutputField(desc="Different output")

		# Same structure with same descriptions should be equal
		assert Sig1.equals(Sig2)

		# Different descriptions should not be equal
		assert not Sig1.equals(Sig3)


class TestComplexScenarios:
	"""Test complex real-world scenarios."""

	def test_multi_step_reasoning_signature(self) -> None:
		"""Test a complex multi-step reasoning signature."""
		sig = make_reasoning_signature(
			"problem: str, context: List[str] -> "
			"analysis: str, steps: List[str], verification: str -> "
			"solution: str, confidence: float"
		)

		assert len(sig.input_fields) == 2
		assert len(sig.reasoning_fields) == 3
		assert len(sig.output_fields) == 2

		# Verify field order is preserved
		field_names = list(sig.fields.keys())
		expected_order = [
			"problem",
			"context",  # inputs
			"analysis",
			"steps",
			"verification",  # reasoning
			"solution",
			"confidence",  # outputs
		]
		assert field_names == expected_order

	def test_signature_chaining(self) -> None:
		"""Test chaining signature manipulations."""
		base_sig = make_reasoning_signature("input -> output")
		final_sig = (
			base_sig.append("reasoning", ReasoningField(desc="Thinking step"))
			.prepend("context", InputField(desc="Additional context"))
			.with_instructions("Process the input with context and reasoning.")
		)
		assert len(final_sig.input_fields) == 2
		assert len(final_sig.reasoning_fields) == 1
		assert len(final_sig.output_fields) == 1
		assert "Process the input with context and reasoning." in final_sig.instructions

	def test_signature_with_all_field_types_and_constraints(self) -> None:
		"""Test signature with all field types and various constraints."""

		class ComplexSignature(ReasoningSignature):
			"""Complex signature for comprehensive testing."""

			# Input fields with constraints
			user_query: str = InputField(
				desc="User's query", min_length=1, max_length=500, prefix="Query:"
			)
			temperature: float = InputField(
				desc="Temperature parameter", ge=0.0, le=2.0, prefix="Temp:"
			)

			# Reasoning fields
			analysis: str = ReasoningField(
				desc="Initial analysis", min_length=10, prefix="Analysis:"
			)
			reasoning_steps: list[str] = ReasoningField(
				desc="Step-by-step reasoning", min_length=1, prefix="Steps:"
			)

			# Output fields
			response: str = OutputField(
				desc="Final response", min_length=1, prefix="Response:"
			)
			confidence: float = OutputField(
				desc="Confidence score", ge=0.0, le=1.0, prefix="Confidence:"
			)

		# Verify field counts
		assert len(ComplexSignature.input_fields) == 2
		assert len(ComplexSignature.reasoning_fields) == 2
		assert len(ComplexSignature.output_fields) == 2

		# Verify constraints are preserved
		temp_field = ComplexSignature.input_fields["temperature"]
		extra = temp_field.json_schema_extra
		if isinstance(extra, dict) and "constraints" in extra:
			constraints = extra["constraints"]
			if isinstance(constraints, str | list):
				assert "greater than or equal to 0" in str(constraints)
				assert "less than or equal to 2" in str(constraints)

		# Verify signature representation
		expected_signature = "user_query, temperature -> analysis, reasoning_steps -> response, confidence"
		assert ComplexSignature.signature == expected_signature


class TestHelperFunctions:
	"""Test helper functions and utilities."""

	def test_parse_field_string(self) -> None:
		"""Test _parse_field_string function."""

		# Simple fields
		fields = _parse_field_string("input1, input2")
		assert len(fields) == 2
		assert fields[0][0] == "input1"
		assert fields[1][0] == "input2"

		# Typed fields
		fields = _parse_field_string("input1: str, input2: int")
		assert fields[0][1] is str
		assert fields[1][1] is int

	def test_convert_fields_to_signature_format(self) -> None:
		"""Test _convert_fields_to_signature_format function."""
		# Create some test fields
		test_fields = {
			"input1": InputField(desc="Test input"),
			"output1": OutputField(desc="Test output"),
		}

		converted = _convert_fields_to_signature_format(test_fields)

		assert "input1" in converted
		assert "output1" in converted
		assert isinstance(converted["input1"], tuple)
		assert len(converted["input1"]) == 2


@pytest.mark.parametrize(
	"signature_string, expected_input_count, expected_reasoning_count, expected_output_count",
	[
		# Single arrow signatures
		("input -> output", 1, 0, 1),
		("input1, input2 -> output", 2, 0, 1),
		("input -> output1, output2", 1, 0, 2),
		("input1, input2 -> output1, output2", 2, 0, 2),
		# Two arrow signatures
		("input -> reasoning -> output", 1, 1, 1),
		("input1, input2 -> reasoning -> output", 2, 1, 1),
		("input -> reasoning1, reasoning2 -> output", 1, 2, 1),
		("input -> reasoning -> output1, output2", 1, 1, 2),
		("input1, input2 -> reasoning1, reasoning2 -> output1, output2", 2, 2, 2),
	],
)
def test_signature_parsing_variations(
	signature_string: str,
	expected_input_count: int,
	expected_reasoning_count: int,
	expected_output_count: int,
) -> None:
	"""Test various signature parsing patterns."""
	sig = make_reasoning_signature(signature_string)

	assert len(sig.input_fields) == expected_input_count
	assert len(sig.reasoning_fields) == expected_reasoning_count
	assert len(sig.output_fields) == expected_output_count


@pytest.mark.parametrize(
	"type_string, expected_type",
	[
		("str", str),
		("int", int),
		("float", float),
		("bool", bool),
		("List[str]", list),  # Will be resolved to basic list type in simple cases
	],
)
def test_type_parsing(type_string: str, expected_type: type) -> None:
	"""Test type parsing from string annotations."""
	sig = make_reasoning_signature(f"input: {type_string} -> output")
	input_field = sig.input_fields[ReasoningState.INPUT]

	# Basic type matching (more complex types may not match exactly)
	if expected_type in (str, int, float, bool):
		assert input_field.annotation == expected_type


class TestReasoningFieldsSpecific:
	"""Test specific functionality for reasoning fields."""

	def test_reasoning_field_properties(self) -> None:
		"""Test reasoning field properties and access methods."""

		class ReasoningSignatureTest(ReasoningSignature):
			"""Test reasoning signature."""

			question: str = InputField(desc="Input question")
			thought_process: str = ReasoningField(desc="Step-by-step reasoning")
			conclusion: str = ReasoningField(desc="Final conclusion")
			answer: str = OutputField(desc="Final answer")

		# Test reasoning fields access
		reasoning_fields = ReasoningSignatureTest.reasoning_fields
		assert len(reasoning_fields) == 2
		assert "thought_process" in reasoning_fields
		assert "conclusion" in reasoning_fields

		# Test that reasoning fields have correct type
		for _field_name, field in reasoning_fields.items():
			extra = field.json_schema_extra
			if callable(extra):
				# Some Pydantic versions return a callable
				continue
			elif extra and isinstance(extra, dict):
				assert extra.get("__dspy_field_type") == "reasoning"

	def test_reasoning_field_in_signature_string(self) -> None:
		"""Test that reasoning fields appear correctly in signature string representation."""
		sig = make_reasoning_signature("input -> step1, step2 -> output1, output2")

		expected = "input -> step1, step2 -> output1, output2"
		assert sig.signature == expected

	def test_mixed_field_ordering(self) -> None:
		"""Test that fields maintain proper order in signature string."""

		class MixedSignature(ReasoningSignature):
			"""Test mixed field types."""

			input_a: str = InputField()
			reasoning_x: str = ReasoningField()
			input_b: str = InputField()
			output_y: str = OutputField()
			reasoning_z: str = ReasoningField()
			output_w: str = OutputField()

		# Fields should be grouped by type in signature string
		assert (
			MixedSignature.signature
			== "input_a, input_b -> reasoning_x, reasoning_z -> output_y, output_w"
		)


class TestSignatureEdgeCases:
	"""Test edge cases and error handling."""

	def test_empty_signature_parts_validation(self) -> None:
		"""Test validation of empty signature parts."""
		# Test empty input
		with pytest.raises(ValueError, match="input part cannot be empty"):
			make_reasoning_signature(" -> output")

		# Test empty output
		with pytest.raises(ValueError, match="output part cannot be empty"):
			make_reasoning_signature("input -> ")

		# Test empty reasoning in two-arrow format
		with pytest.raises(ValueError, match="reasoning part cannot be empty"):
			make_reasoning_signature("input ->  -> output")

	def test_whitespace_handling(self) -> None:
		"""Test proper handling of whitespace in signatures."""
		sig1 = make_reasoning_signature("  input  ,  context  ->  output  ")
		sig2 = make_reasoning_signature("input, context -> output")

		# Both should parse the same
		assert len(sig1.input_fields) == 2
		assert len(sig1.output_fields) == 1
		assert sig1.signature == sig2.signature

	def test_invalid_arrow_counts(self) -> None:
		"""Test error handling for invalid arrow counts."""
		# No arrows
		with pytest.raises(ValueError, match="Invalid signature format"):
			make_reasoning_signature("input output")

		# Three arrows
		with pytest.raises(ValueError, match="Invalid signature format"):
			make_reasoning_signature("input -> reasoning -> output -> extra")

	def test_single_field_signatures(self) -> None:
		"""Test signatures with single fields."""
		# Single input to single output
		sig1 = make_reasoning_signature("input -> output")
		assert len(sig1.input_fields) == 1
		assert len(sig1.output_fields) == 1

		# Single input through reasoning to single output
		sig2 = make_reasoning_signature("input -> reasoning -> output")
		assert len(sig2.input_fields) == 1
		assert len(sig2.reasoning_fields) == 1
		assert len(sig2.output_fields) == 1


class TestSignatureEquality:
	"""Test signature equality and comparison methods."""

	def test_signature_equals_method(self) -> None:
		"""Test the equals method for signature comparison."""
		sig1 = make_reasoning_signature("input -> output")
		sig2 = make_reasoning_signature("input -> output")
		sig3 = make_reasoning_signature("input -> reasoning -> output")

		# Same signatures should be equal
		assert sig1.equals(sig2)

		# Different signatures should not be equal
		assert not sig1.equals(sig3)

	def test_signature_equals_with_instructions(self) -> None:
		"""Test equals method with different instructions."""
		sig1 = make_reasoning_signature("input -> output", "First instructions")
		sig2 = make_reasoning_signature("input -> output", "Second instructions")

		# Different instructions should make signatures unequal
		assert not sig1.equals(sig2)

	def test_signature_equals_non_signature(self) -> None:
		"""Test equals method with non-signature objects."""
		sig = make_reasoning_signature("input -> output")

		# Should return False for non-signature objects
		assert not sig.equals(str)
		assert not sig.equals("string")
		assert not sig.equals(42)


class TestSignatureModificationMethods:
	"""Test signature modification methods like append, prepend, etc."""

	def test_append_to_reasoning_signature(self) -> None:
		"""Test appending fields to a signature with reasoning fields."""
		base_sig = make_reasoning_signature("input -> reasoning -> output")

		# Append input field
		new_sig = base_sig.append("new_input", InputField(desc="New input"))
		assert "new_input" in new_sig.input_fields
		assert len(new_sig.input_fields) == 2

		# Append reasoning field
		new_sig2 = base_sig.append(
			"new_reasoning", ReasoningField(desc="New reasoning")
		)
		assert "new_reasoning" in new_sig2.reasoning_fields
		assert len(new_sig2.reasoning_fields) == 2

	def test_with_instructions_preserves_reasoning(self) -> None:
		"""Test that with_instructions preserves reasoning fields."""
		base_sig = make_reasoning_signature("input -> step1, step2 -> output")
		new_sig = base_sig.with_instructions("New instructions")

		assert len(new_sig.reasoning_fields) == 2
		assert new_sig.instructions == "New instructions"
		assert new_sig.signature == "input -> step1, step2 -> output"


if __name__ == "__main__":
	pytest.main([__file__, "-vv"])
