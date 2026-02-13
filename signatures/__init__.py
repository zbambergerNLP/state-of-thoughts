"""Signatures module for reasoning-based DSPy signatures."""

from signatures.example_signatures import (
	AnalyzeTextWithReasoning,
	ArgumentEvaluatorMultiDimensional,
	ArgumentEvaluatorSingleScore,
	BalancedArgumentRubric,
	GenerateArgumentWithReasoning,
	QuestionAnsweringWithReasoning,
	SolveMathProblemWithReasoning,
	WeightedMultiDimensionRubric,
	WeightedPRMRubric,
)
from signatures.field import (
	InputField,
	OutputField,
	ReasoningField,
)
from signatures.signature import (
	ReasoningSignature,
	ensure_reasoning_signature,
	make_reasoning_signature,
)

__all__ = [
	# Example signatures
	"AnalyzeTextWithReasoning",
	"ArgumentEvaluatorMultiDimensional",
	"ArgumentEvaluatorSingleScore",
	"BalancedArgumentRubric",
	"GenerateArgumentWithReasoning",
	"GenerateArgumentWithReasoningAndPersona",
	"QuestionAnsweringWithReasoning",
	"SolveMathProblemWithReasoning",
	"WeightedMultiDimensionRubric",
	"WeightedPRMRubric",
	# Field types
	"InputField",
	"OutputField",
	"ReasoningField",
	# Signature classes and functions
	"ReasoningSignature",
	"ensure_reasoning_signature",
	"make_reasoning_signature",
]

