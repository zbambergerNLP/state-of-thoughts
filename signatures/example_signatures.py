"""
Examples of reasoning-based signatures used across both unit tests and experiments.
These signatures are meant to express relalistic tasks that may benefit from reasoning.
"""

# Standard library imports
import enum
from typing import Literal

# Third-party imports
import dspy

# Local imports
from signatures.field import InputField, OutputField, ReasoningField
from signatures.signature import ReasoningSignature


class ArgumentField(enum.StrEnum):
	"""Enum for argument-related fields."""
	TOPIC = "topic"
	STANCE = "stance"
	CLAIM = "claim"
	ARGUMENT = "argument"


class QuestionField(enum.StrEnum):
	"""Enum for question-answering fields."""
	QUESTION = "question"
	REASONING_STEP = "reasoning_step"
	ANSWER = "answer"


class TextAnalysisField(enum.StrEnum):
	"""Enum for text analysis fields."""
	INPUT_TEXT = "input_text"
	SUMMARY = "summary"
	SENTIMENT = "sentiment"
	KEYWORDS = "keywords"


ArgumentStance = Literal["PRO", "ANTI"]

class QuestionAnsweringWithReasoning(ReasoningSignature):
	"""Answer the provided question with step-by-step reasoning."""

	question: str = InputField(desc="The question to answer")
	reasoning_step: str = ReasoningField(desc="Step-by-step reasoning")
	answer: str = OutputField(desc="The answer to the question")


class SolveMathProblemWithReasoning(ReasoningSignature):
	"""
	Solve the provided math problem and return its answer.
	"""

	math_problem: str = InputField(desc="The math problem to solve")
	math_operation: str = ReasoningField(desc="A math operation towards solving the math problem")
	answer: str = OutputField(desc="The answer to the math problem")

class AnalyzeTextWithReasoning(ReasoningSignature):
	"""
	Analyze the inputted text and perform the following tasks:
	- Summarize the text
	- Determine the sentiment of the text
	- Extract key words from the text
	"""

	input_text: str = InputField(desc="Input text to process")
	reasoning_step: str = ReasoningField(desc="Step-by-step reasoning")
	summary: str = OutputField(desc="A summary of the input")
	sentiment: str = OutputField(desc="The sentiment of the input")
	keywords: list[str] = OutputField(desc="Key words from the input")

class GenerateArgumentWithReasoning(ReasoningSignature):
	"""
	Generate an argument which takes the provided stance towards the provided topic.
	"""

	topic: str = InputField(desc="The topic to generate an argument about")
	stance: ArgumentStance = InputField(desc="The stance to take on the topic")
	claim: str = ReasoningField(
		desc="A component of the argument that advocates for the given stance towards the topic"
	)
	argument: str = OutputField(desc="The generated argument")

class GenerateArgumentWithReasoningAndPersona(ReasoningSignature):
	"""
	Generate an argument which takes the provided stance towards the provided topic.
	"""

	topic: str = InputField(desc="The topic to generate an argument about")
	stance: ArgumentStance = InputField(desc="The stance to take on the topic")
	persona: str = InputField(
		desc="Optional persona or characteristic to influence your thinking. Empty string means no specific persona.",
		default="The average person, who is not necessarily a domain-expert in the topic.",
	)
	claim: str = ReasoningField(desc="A component of the argument that advocates for the given stance towards the topic")
	argument: str = OutputField(desc="The generated argument")


# Evaluator Signatures for Argument Generation
# Note: Evaluator signatures include:
# - constants.INPUT fields: the generator's input fields + output fields (what we're evaluating)
# - constants.OUTPUT fields: at least one numeric scoring field + exactly one feedback field


class ArgumentEvaluatorMultiDimensional(dspy.Signature):
	"""Evaluate the generated argument along three dimensions:

	PERSUASIVENESS (1-7): How convincing and compelling the argument is
	- 7 = Highly persuasive with strong evidence and reasoning
	- 5-6 = Moderately persuasive with reasonable support
	- 3-4 = Somewhat persuasive but lacks strong support
	- 1-2 = Weak or unconvincing argument

	COHERENCE (1-7): How well-structured and logically organized the argument is
	- 7 = Perfectly coherent with clear logical flow
	- 5-6 = Generally coherent with minor organizational issues
	- 3-4 = Somewhat coherent but with notable structural problems
	- 1-2 = Poorly organized or incoherent

	RELEVANCE (1-7): How well the argument addresses the topic and stance
	- 7 = Perfectly aligned with topic and stance
	- 5-6 = Generally relevant with minor deviations
	- 3-4 = Partially relevant but misses key aspects
	- 1-2 = Off-topic or misaligned with stance

	Provide separate numeric scores for each dimension.

	Return your response using exactly these headers and numeric values only:

	## persuasiveness
	<INTEGER 1-7>

	## coherence
	<INTEGER 1-7>

	## relevance
	<INTEGER 1-7>
	"""

	# Input fields (from generator)
	topic: str = dspy.InputField(desc="The topic the argument is about")
	stance: ArgumentStance = dspy.InputField(desc="The stance taken on the topic")
	argument: str = dspy.InputField(desc="The generated argument to evaluate")

	# Output fields (evaluation scores)
	persuasiveness: int = OutputField(desc="Persuasiveness score", ge=1, le=7)
	coherence: int = OutputField(desc="Coherence score", ge=1, le=7)
	relevance: int = OutputField(desc="Relevance score", ge=1, le=7)


class ArgumentEvaluatorSingleScore(dspy.Signature):
	"""Evaluate the overall quality of the generated argument:

	10 = Exceptional: Highly persuasive, perfectly coherent, and fully addresses the topic/stance
	8-9 = Strong: Very good argument with only minor weaknesses
	6-7 = Good: Solid argument but with some notable issues
	4-5 = Adequate: Acceptable but with significant room for improvement
	2-3 = Weak: Poor quality with major flaws in persuasiveness, coherence, or relevance
	1 = Very Poor: Fails to meet basic standards for an argument

	Provide a single overall quality score.
	"""

	# Input fields (from generator)
	topic: str = dspy.InputField(desc="The topic the argument is about")
	stance: ArgumentStance = dspy.InputField(desc="The stance taken on the topic")
	argument: str = dspy.InputField(desc="The generated argument to evaluate")

	# Output fields (evaluation score)
	overall_quality: int = OutputField(desc="Overall argument quality score", ge=1, le=10)


# Evaluator Rubric Examples

class WeightedPRMRubric(dspy.Signature):
	"""Evaluate quality of intermediate reasoning steps towards addressing the user's task."""

	soundness: int = OutputField(
		desc="Correctness of the reasoning so far",
		ge=1,
		le=7,
		rubric_weight=0.7,
	)
	promise: int = OutputField(
		desc="Likelihood of the reasoning to lead to a strong final answer",
		ge=1,
		le=7,
		rubric_weight=0.3,
	)


class WeightedMultiDimensionRubric(dspy.Signature):
	"""Assess the correctness, clarity, and efficiency of the provided solution."""

	correctness: float = OutputField(
		desc="accuracy of facts, formulas, and calculations",
		ge=1.0,
		le=5.0,
		rubric_weight=0.5,
	)
	clarity: float = OutputField(
		desc="how well-explained and understandable the reasoning is",
		ge=1.0,
		le=5.0,
		rubric_weight=0.3,
	)
	efficiency: float = OutputField(
		desc="directness and conciseness of the approach",
		ge=1.0,
		le=5.0,
		rubric_weight=0.2,
	)


class BalancedArgumentRubric(dspy.Signature):
	"""Evaluate arguments across persuasiveness, coherence, and relevance."""

	persuasiveness: int = OutputField(
		desc="How convincing the argument is", ge=1, le=7
		# No rubric_weight specified = equal weights (33.3% each)
	)
	coherence: int = OutputField(
		desc="How well-structured and logically organized the argument is", ge=1, le=7
		# No rubric_weight specified = equal weights
	)
	relevance: int = OutputField(
		desc="How well the argument addresses the topic and stance", ge=1, le=7
		# No rubric_weight specified = equal weights
	)
