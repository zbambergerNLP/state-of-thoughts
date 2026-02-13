"""
Comprehensive tests for LocalPredict batching detection with type-based logic.

Tests cover:
1. Simple types (str, int)
2. List types (list[str])
3. Nested list types (list[list[str]])
4. Dict types (dict[str, int])
5. Mixed batching scenarios
6. Edge cases
"""

import dspy
import pytest

from predict.local_predict import LocalPredict

# ============================================================================
# Test Batching Detection Logic
# ============================================================================


class TestBatchingDetection:
	"""Test batching detection with various field types."""

	def test_simple_string_field_single(self):
		"""Test single value for string field."""
		# Create a simple signature
		sig = dspy.Signature({"question": (str, dspy.InputField())})

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(sig, question="Q1")

		assert len(batch_inputs) == 1
		assert batch_inputs[0]["question"] == "Q1"

	def test_simple_string_field_batch(self):
		"""Test batch values for string field."""
		sig = dspy.Signature({"question": (str, dspy.InputField())})

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(
			sig, question=["Q1", "Q2", "Q3"]
		)

		assert len(batch_inputs) == 3
		assert batch_inputs[0]["question"] == "Q1"
		assert batch_inputs[1]["question"] == "Q2"
		assert batch_inputs[2]["question"] == "Q3"

	def test_list_field_single(self):
		"""Test single value for list[str] field."""
		sig = dspy.Signature({"reasoning_steps": (list[str], dspy.InputField())})

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(
			sig, reasoning_steps=["step1", "step2", "step3"]
		)

		# This should be treated as a SINGLE input (not batching)
		assert len(batch_inputs) == 1
		assert batch_inputs[0]["reasoning_steps"] == ["step1", "step2", "step3"]

	def test_list_field_batch(self):
		"""Test batch values for list[str] field."""
		sig = dspy.Signature({"reasoning_steps": (list[str], dspy.InputField())})

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(
			sig,
			reasoning_steps=[
				["step1a", "step1b"],
				["step2a", "step2b", "step2c"],
			],
		)

		# This should be treated as BATCHING (list of list[str])
		assert len(batch_inputs) == 2
		assert batch_inputs[0]["reasoning_steps"] == ["step1a", "step1b"]
		assert batch_inputs[1]["reasoning_steps"] == ["step2a", "step2b", "step2c"]

	def test_mixed_batching(self):
		"""Test mixed batching: some fields batched, others broadcasted."""
		sig = dspy.Signature(
			{
				"question": (str, dspy.InputField()),
				"context": (str, dspy.InputField()),
			}
		)

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(
			sig, question=["Q1", "Q2"], context="shared context"
		)

		# question is batched, context is broadcasted
		assert len(batch_inputs) == 2
		assert batch_inputs[0]["question"] == "Q1"
		assert batch_inputs[0]["context"] == "shared context"
		assert batch_inputs[1]["question"] == "Q2"
		assert batch_inputs[1]["context"] == "shared context"

	def test_multiple_batched_fields_same_length(self):
		"""Test multiple fields batched with same length."""
		sig = dspy.Signature(
			{
				"question": (str, dspy.InputField()),
				"context": (str, dspy.InputField()),
			}
		)

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(
			sig, question=["Q1", "Q2"], context=["C1", "C2"]
		)

		assert len(batch_inputs) == 2
		assert batch_inputs[0]["question"] == "Q1"
		assert batch_inputs[0]["context"] == "C1"
		assert batch_inputs[1]["question"] == "Q2"
		assert batch_inputs[1]["context"] == "C2"

	def test_multiple_batched_fields_different_length_error(self):
		"""Test error when batched fields have different lengths."""
		sig = dspy.Signature(
			{
				"question": (str, dspy.InputField()),
				"context": (str, dspy.InputField()),
			}
		)

		predictor = LocalPredict(sig)

		with pytest.raises(AssertionError, match="same length"):
			predictor._detect_and_prepare_batching(
				sig,
				question=["Q1", "Q2"],
				context=["C1", "C2", "C3"],  # Different length!
			)

	def test_dict_field_single(self):
		"""Test single value for dict field."""
		sig = dspy.Signature({"metadata": (dict[str, int], dspy.InputField())})

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(
			sig, metadata={"key1": 1, "key2": 2}
		)

		assert len(batch_inputs) == 1
		assert batch_inputs[0]["metadata"] == {"key1": 1, "key2": 2}

	def test_dict_field_batch(self):
		"""Test batch values for dict field."""
		sig = dspy.Signature({"metadata": (dict[str, int], dspy.InputField())})

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(
			sig, metadata=[{"key1": 1}, {"key2": 2, "key3": 3}]
		)

		assert len(batch_inputs) == 2
		assert batch_inputs[0]["metadata"] == {"key1": 1}
		assert batch_inputs[1]["metadata"] == {"key2": 2, "key3": 3}

	def test_invalid_type_error(self):
		"""Test error when value doesn't match expected type."""
		sig = dspy.Signature({"question": (str, dspy.InputField())})

		predictor = LocalPredict(sig)

		with pytest.raises(TypeError, match="expects type"):
			predictor._detect_and_prepare_batching(sig, question=42)

	def test_invalid_list_elements_error(self):
		"""Test error when list elements don't match expected type."""
		sig = dspy.Signature({"question": (str, dspy.InputField())})

		predictor = LocalPredict(sig)

		with pytest.raises(TypeError, match="expects type"):
			# Trying to batch with wrong element types
			predictor._detect_and_prepare_batching(sig, question=[1, 2, 3])

	def test_empty_list_for_non_optional_list_field_error(self):
		"""Test empty list for non-optional list[str] field raises error."""
		sig = dspy.Signature({"reasoning_steps": (list[str], dspy.InputField())})

		predictor = LocalPredict(sig)

		with pytest.raises(TypeError, match="not optional but received empty list"):
			# Empty list not allowed for non-optional fields
			predictor._detect_and_prepare_batching(sig, reasoning_steps=[])

	def test_empty_list_for_optional_list_field(self):
		"""Test empty list for optional list[str] field is valid."""
		sig = dspy.Signature({"reasoning_steps": (list[str] | None, dspy.InputField())})

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(sig, reasoning_steps=[])

		# Empty list is valid for optional fields
		assert len(batch_inputs) == 1
		assert batch_inputs[0]["reasoning_steps"] == []

	def test_empty_list_for_string_field_error(self):
		"""Test empty list for string field raises error."""
		sig = dspy.Signature({"question": (str, dspy.InputField())})

		predictor = LocalPredict(sig)

		with pytest.raises(TypeError, match="not optional but received empty list"):
			# Empty list not allowed for non-optional fields
			predictor._detect_and_prepare_batching(sig, question=[])

	def test_none_for_non_optional_field_error(self):
		"""Test None for non-optional field raises error."""
		sig = dspy.Signature({"question": (str, dspy.InputField())})

		predictor = LocalPredict(sig)

		with pytest.raises(TypeError, match="not optional but received None"):
			predictor._detect_and_prepare_batching(sig, question=None)

	def test_none_for_optional_field(self):
		"""Test None for optional field is valid."""
		sig = dspy.Signature({"question": (str | None, dspy.InputField())})

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(sig, question=None)

		# None is valid for optional fields
		assert len(batch_inputs) == 1
		assert batch_inputs[0]["question"] is None

	def test_none_for_optional_list_field(self):
		"""Test None for optional list[str] field is valid."""
		sig = dspy.Signature({"reasoning_steps": (list[str] | None, dspy.InputField())})

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(sig, reasoning_steps=None)

		# None is valid for optional fields
		assert len(batch_inputs) == 1
		assert batch_inputs[0]["reasoning_steps"] is None

	def test_empty_list_and_none_for_optional_list_field(self):
		"""Test both empty list and None work for optional list[str] field."""
		sig = dspy.Signature({"reasoning_steps": (list[str] | None, dspy.InputField())})

		predictor = LocalPredict(sig)

		# Test empty list
		batch_inputs = predictor._detect_and_prepare_batching(sig, reasoning_steps=[])
		assert len(batch_inputs) == 1
		assert batch_inputs[0]["reasoning_steps"] == []

		# Test None
		batch_inputs = predictor._detect_and_prepare_batching(sig, reasoning_steps=None)
		assert len(batch_inputs) == 1
		assert batch_inputs[0]["reasoning_steps"] is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestBatchingIntegration:
	"""Integration tests combining multiple scenarios."""

	def test_complex_mixed_batching(self):
		"""Test complex scenario with multiple field types."""
		sig = dspy.Signature(
			{
				"question": (str, dspy.InputField()),
				"reasoning_steps": (list[str], dspy.InputField()),
				"temperature": (float, dspy.InputField()),
			}
		)

		predictor = LocalPredict(sig)
		batch_inputs = predictor._detect_and_prepare_batching(
			sig,
			question=["Q1", "Q2"],  # Batched
			reasoning_steps=[["step1a"], ["step2a", "step2b"]],  # Batched
			temperature=0.7,  # Broadcasted
		)

		assert len(batch_inputs) == 2

		assert batch_inputs[0]["question"] == "Q1"
		assert batch_inputs[0]["reasoning_steps"] == ["step1a"]
		assert batch_inputs[0]["temperature"] == 0.7

		assert batch_inputs[1]["question"] == "Q2"
		assert batch_inputs[1]["reasoning_steps"] == ["step2a", "step2b"]
		assert batch_inputs[1]["temperature"] == 0.7

	def test_deeply_nested_lists(self):
		"""Test deeply nested list types."""
		sig = dspy.Signature({"data": (list[list[list[str]]], dspy.InputField())})

		predictor = LocalPredict(sig)

		# Single value: list[list[list[str]]]
		single_value = [[["a", "b"], ["c"]], [["d"]]]
		batch_inputs = predictor._detect_and_prepare_batching(sig, data=single_value)

		assert len(batch_inputs) == 1
		assert batch_inputs[0]["data"] == single_value

		# Batch value: list of list[list[list[str]]]
		batch_value = [[[["a", "b"]]], [[["c"]], [["d", "e"]]]]
		batch_inputs = predictor._detect_and_prepare_batching(sig, data=batch_value)

		assert len(batch_inputs) == 2
		assert batch_inputs[0]["data"] == [[["a", "b"]]]
		assert batch_inputs[1]["data"] == [[["c"]], [["d", "e"]]]


if __name__ == "__main__":
	pytest.main([__file__, "-v"])
