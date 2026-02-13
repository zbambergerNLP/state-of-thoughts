"""Shared DSPy signatures for simulation and benchmark experiments."""

# Third-party imports
import dspy

# Local imports
from signatures.field import InputField, OutputField, ReasoningField
from signatures.signature import ReasoningSignature

####################################################
### Single-Turn Instruction-Following signatures ###
####################################################

class InstructionFollowing(dspy.Signature):
	"""
	Follow the instructions and abide by the stated specifications and constraints.
	"""
	instruction: str = dspy.InputField(desc="The instruction from the user.")
	result: str = dspy.OutputField(desc="The result of following the instruction.")

class InstructionFollowingCoT(dspy.Signature):
	"""
	Follow the instructions and abide by the stated specifications and constraints.
	"""
	instruction: str = dspy.InputField(desc="The instruction from the user.")
	rationale: str = dspy.OutputField(desc="Reasoning steps towards following the instruction.")
	result: str = dspy.OutputField(desc="The result of following the instruction.")

class InstructionFollowingWithTools(dspy.Signature):
	"""
	Follow the instructions and abide by the stated specifications and constraints.
	"""
	instruction: str = dspy.InputField(desc="The instruction from the user.")
	action_space: str = dspy.InputField(desc="Considerations for reasoning about the problem.")
	result: str = dspy.OutputField(desc="The result of following the instruction.")

class InstructionFollowingWithToolsCoT(dspy.Signature):
	"""
	Follow the instructions and abide by the stated specifications and constraints.
	"""
	instruction: str = dspy.InputField(desc="The instruction from the user.")
	action_space: str = dspy.InputField(desc="Considerations for reasoning about the problem.")
	rationale: str = dspy.OutputField(desc="Reasoning steps towards following the instruction.")
	result: str = dspy.OutputField(desc="The result of following the instruction.")

class InstructionFollowingWithReasoning(ReasoningSignature):
    """
    Follow the instructions and abide by the stated specifications and constraints.
    """
    instruction: str = dspy.InputField(desc="The instruction from the user.")
    reasoning_step: str = ReasoningField(
        desc="A sentence, paragraph, or other textual addition that will be part of the final response. This is a partial component (e.g., header, sub-header, bullet point, etc.) of the final response, and not the final response itself. This addition must not violate the constraints and specifications in `instruction` and must not include planning, reflecting, or meta-cognitive reasoning about the problem or how to solve it."
    )
    result: str = dspy.OutputField(desc="The result of following the instruction.")

class InstructionFollowingWithReasoningAndTools(ReasoningSignature):
    """
    Follow the instructions and abide by the stated specifications and constraints.
    """
    instruction: str = dspy.InputField(desc="The instruction from the user.")
    action_space: str = dspy.InputField(desc="Considerations for reasoning about the problem.")
    reasoning_step: str = ReasoningField(
        desc="A sentence, paragraph, or other textual addition that will be part of the final response. This is a partial component (e.g., header, sub-header, bullet point, etc.) of the final response, and not the final response itself. This addition must not violate the constraints and specifications in `instruction` and must not include planning, reflecting, or meta-cognitive reasoning about the problem or how to solve it."
    )
    result: str = dspy.OutputField(desc="The result of following the instruction.")

###################################################
### Multi-Turn Instruction-Following signatures ###
###################################################

# TODO[P2]: Use DSPy's History field type instead of list[str].

class MultiTurnInstructionFollowing(dspy.Signature):
	"""
	Follow the user's instructions (from the conversation history) in a multi-turn conversation.
	"""

	conversation_history: list[str] = dspy.InputField(
		desc="A list of messages between a human user and an LLM assistant. The most recent message includes the user's instructions, which you must follow."
	)
	result: str = dspy.OutputField(
		desc="The result of following the user's instructions (from the conversation history)."
	)

class MultiTurnInstructionFollowingCoT(dspy.Signature):
	"""
	Follow the instructions in the user's most recent message in a multi-turn conversation.
	"""

	conversation_history: list[str] = dspy.InputField(
		desc="A list of messages between a human user and an LLM assistant. The most recent message includes the user's instructions, which you must follow."
	)
	rationale: str = dspy.OutputField(desc="Reasoning steps towards following the instruction.")
	result: str = dspy.OutputField(
		desc="The result of following the user's instructions (from the conversation history)."
	)

class MultiTurnInstructionFollowingWithTools(dspy.Signature):
	"""
	Follow the instructions in the user's most recent message in a multi-turn conversation.
	"""

	conversation_history: list[str] = dspy.InputField(
		desc="A list of messages between a human user and an LLM assistant. The most recent message includes the user's instruction, which you must follow."
	)
	action_space: str = dspy.InputField(desc="Considerations for reasoning about the problem.")
	result: str = dspy.OutputField(
		desc="The result of following the user's instructions (from the conversation history)."
	)

class MultiTurnInstructionFollowingWithToolsCoT(dspy.Signature):
	"""
	Follow the instructions in the user's most recent message in a multi-turn conversation.
	"""

	conversation_history: list[str] = dspy.InputField(
		desc="A list of messages between a human user and an LLM assistant. The most recent message includes the user's instructions, which you must follow."
	)
	action_space: str = dspy.InputField(desc="Considerations for how to reason about the user's instructions.")
	rationale: str = dspy.OutputField(desc="Reasoning steps towards following the user's instructions.")
	result: str = dspy.OutputField(
		desc="The result of following the user's instructions (from the conversation history)."
	)

class InstructionFollowingMultiTurnWithReasoning(ReasoningSignature):
	"""
	Follow the user's instructions (from the conversation history) in a multi-turn conversation.
	"""

	conversation_history: list[str] = InputField(
		desc="A list of messages between a human user and an LLM assistant. The most recent message includes the user's instructions, which you must follow."
	)
	reasoning_step: str = ReasoningField(
		desc="A single reasoning step planning or preparing the response, not a final response to the user's instructions."
	)
	result: str = OutputField(desc="The result of following the user's instructions (from the conversation history).")

class InstructionFollowingMultiTurnWithReasoningAndWithTools(ReasoningSignature):
	"""
	Follow the user's instructions (from the conversation history) in a multi-turn conversation.
	"""

	conversation_history: list[str] = InputField(
		desc="A list of messages between a human user and an LLM assistant. The most recent message includes the user's instructions, which you must follow."
	)
	action_space: str = InputField(desc="Considerations for how to reason about the user's instructions.")
	reasoning_step: str = ReasoningField(
		desc="A single reasoning step planning or preparing the response, not a final response to the user's instructions."
	)
	result: str = OutputField(desc="The result of following the user's instructions (from the conversation history).")


###############################################################
### Evaluator Signatures for Instruction-Following ToT ###
###############################################################


class InstructionFollowingReasoningRubric(dspy.Signature):
	"""Evaluate how well a reasoning trajectory follows the user's instructions.

	Given a reasoning trajectory, score each dimension based on how well the reasoning
	addresses and advances toward fulfilling the user's request:

	INSTRUCTION_ADHERENCE (0-10): How well the reasoning step addresses the user's instructions
	- 10 = Perfectly addresses and advances toward fulfilling the user's request
	- 7-9 = Strongly aligned with instructions, making clear progress
	- 4-6 = Partially relevant but may miss key aspects of the instructions
	- 1-3 = Largely ignores or misinterprets the user's instructions
	- 0 = Completely ignores or contradicts the user's instructions

	REASONING_QUALITY (0-10): Logical soundness, accuracy, and coherence of the step
	- 10 = Flawless logic, accurate facts, excellent coherence with context
	- 7-9 = Sound reasoning with minor imperfections that don't affect validity
	- 4-6 = Acceptable but with notable gaps in logic or accuracy
	- 1-3 = Significant logical errors, inaccuracies, or incoherence
	- 0 = Completely illogical, factually wrong, or incoherent

	PROGRESS (0-10): How much the step advances toward a high-quality solution
	- 10 = Optimal advancement, clear path to excellent solution
	- 7-9 = Strong progress, high likelihood of quality outcome
	- 4-6 = Some progress but suboptimal or meandering
	- 1-3 = Little to no progress, stuck or counterproductive
	- 0 = No progress or actively harmful to solution
	"""

	instruction_adherence: float = OutputField(
		desc="How well the reasoning step addresses and follows the user's instructions (0-10)",
		ge=0,
		le=10,
		rubric_weight=0.4,
	)
	reasoning_quality: float = OutputField(
		desc="Logical soundness, factual accuracy, and coherence of the reasoning step (0-10)",
		ge=0,
		le=10,
		rubric_weight=0.35,
	)
	progress: float = OutputField(
		desc="How much the step advances toward a high-quality solution (0-10)",
		ge=0,
		le=10,
		rubric_weight=0.25,
	)
	feedback: str = OutputField(
		desc="Detailed feedback explaining the scores for each dimension"
	)


class InstructionFollowingResponseRubric(dspy.Signature):
	"""Evaluate how well a response follows the user's instructions.

	Given a response to the user's request, score each dimension based on how well
	the response fulfills the user's instructions:

	HELPFULNESS (0-10): How useful and actionable the response is
	- 10 = Exceptionally helpful, provides exactly what the user needs
	- 7-9 = Very helpful with clear practical value
	- 4-6 = Moderately helpful but could be more useful
	- 1-3 = Minimally helpful or unhelpful
	- 0 = Completely unhelpful or harmful

	RELEVANCE (0-10): How well the response addresses the user's actual request
	- 10 = Perfectly addresses all aspects of the user's question/instruction
	- 7-9 = Addresses most aspects with minor omissions
	- 4-6 = Partially relevant but misses key aspects
	- 1-3 = Largely off-topic or misses the point
	- 0 = Completely off-topic or unrelated

	ACCURACY (0-10): Correctness of facts, claims, and reasoning
	- 10 = Completely accurate with no errors
	- 7-9 = Highly accurate with only trivial imperfections
	- 4-6 = Mostly accurate but with some notable errors
	- 1-3 = Contains significant factual errors or flawed reasoning
	- 0 = Completely incorrect or fabricated

	COMPLETENESS (0-10): Depth, thoroughness, and level of detail
	- 10 = Comprehensive coverage with appropriate depth and detail
	- 7-9 = Thorough with good detail, minor gaps acceptable
	- 4-6 = Adequate but lacks depth or misses important details
	- 1-3 = Superficial or incomplete treatment
	- 0 = No meaningful content or entirely missing
	"""

	helpfulness: float = OutputField(
		desc="How useful and actionable the response is for the user (0-10)",
		ge=0,
		le=10,
		rubric_weight=0.3,
	)
	relevance: float = OutputField(
		desc="How well the response addresses the user's actual request (0-10)",
		ge=0,
		le=10,
		rubric_weight=0.25,
	)
	accuracy: float = OutputField(
		desc="Correctness of facts, claims, and reasoning in the response (0-10)",
		ge=0,
		le=10,
		rubric_weight=0.3,
	)
	completeness: float = OutputField(
		desc="Depth, thoroughness, and level of detail in the response (0-10)",
		ge=0,
		le=10,
		rubric_weight=0.15,
	)
	feedback: str = OutputField(
		desc="Detailed feedback explaining the scores, covering helpfulness, relevance, accuracy, and completeness"
	)
