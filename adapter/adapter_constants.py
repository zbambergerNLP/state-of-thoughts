"""Constants specific to adapters."""
# Standard library imports
import re
from typing import Literal

# FINAL_OUTPUT_KIND_TYPE refects the various synthesis methods for producing a final output given
# a sequence of reasoning steps.
FINAL_OUTPUT_KIND_TYPE = Literal[
	# Synthesis Strict: Very faithful to the reasoning steps.
	# Preserve content, structure, ordering, and phrasing as closely as possible.
	"synthesis_strict",
	# Synthesis Faithful: Faithful to the ideas and reasoning steps, but allows light rephrasing.
	# Ordering and structure should remain the same, with minimal stylistic edits.
	"synthesis_faithful",
	# Synthesis Restructured: Maintains the same broad ideas and reasoning,
	# but allows rephrasing and restructuring for clarity and coherence.
	"synthesis_restructured",
	# Conclusion: Allows the model to produce the best possible final answer based on the reasoning.
	# Prioritizes clarity and quality over strict faithfulness to structure or phrasing.
	"conclusion",
]

# Adapter configuration constant
GENERATOR_ADAPTER_NAME = "VLLMGeneratorAdapter"

# Pattern constants
FIELD_HEADER_PATTERN = re.compile(r"##\s+(\w+)")
