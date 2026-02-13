"""Prompts for the LLM adapters

The prompts in this file are meant to evoke multi-step reasoning within a single LLM reasoning
trajectory (i.e., between <thinking>...</thinking> tags). Each step in the reasoning process
requires an LLM call.

Definitions for variables in templates:
* `task_instructions`: The task instructions provided by the user (i.e., the signature
  instructions).
* `field_descriptions`: The descriptions of the input and output fields from the signature.
* `thought_length_instruction_formatted`: Formatted thought length instructions.
* `response_length_instruction_formatted`: Formatted response length instructions.
* `reasoning_field_name`: The name of the reasoning field (i.e., the field that contains a sequence
  of reasoning steps).
* `reasoning_field_type`: The type of the reasoning field (typically a string).
* `reasoning_field_description`: The description of the reasoning field (i.e., what a single step
  in the reasoning trajectory entails).
* `output_fields`: The output fields from the signature
* `input_fields`: The input fields from the signature
* `final_output_description`: A block of instructions for how to produce an output to the user's
  request. See `FinalOutputKind` for the different types of final output instructions (i.e.,
  `synthesis_strict`, `synthesis_faithful`, `synthesis_restructured`, `conclusion`).
* `output_field_sections`: The sections of the output fields from the signature.
* `formatted_input_fields`: The formatted input fields from an in-context example.
* `formatted_reasoning_steps`: The formatted reasoning steps from an in-context example.
* `formatted_answer`: The formatted answer from an in-context example.

NOTE: Adapters convert DSPy signatures and their corresponding inputs/in-context examples into
messages for the LM call.
"""

######################
### System Prompts ###
######################

SIMPLE_MAIN_TEMPLATE = """
{task_instructions}

{field_descriptions}{response_length_instruction_formatted}

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
"""

GENERATOR_SYSTEM_PROMPT_VANILLA = """
# Instructions
{task_instructions}

{field_descriptions}

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within <step>...</step> tags, and contain a `{reasoning_field_name}` that advances the solution towards producing {output_fields}.
{final_output_description}

Your reasoning process should follow the rules below:
- Each `{reasoning_field_name}` (of type `{reasoning_field_type}`) entails {reasoning_field_description}.{thought_length_instruction_formatted}{response_length_instruction_formatted}

## Response Format
Once a user provides {input_fields}, your response must follow this exact template:

<thinking>
<step>
## {reasoning_field_name}
The first reasoning step towards producing {output_fields}
</step>
<step>
## {reasoning_field_name}
The second reasoning step towards producing {output_fields}
</step>
...
<step>
## {reasoning_field_name}
The final reasoning step towards producing {output_fields}
</step>
</thinking>
<answer>
{output_field_sections}
</answer>
"""

GENERATOR_SYSTEM_PROMPT_INTERNAL_REASONING = """
# Instructions

{task_instructions}

{field_descriptions}

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within <step>...</step> tags, and contain a `{reasoning_field_name}` that advances the solution towards producing {output_fields}.
{final_output_description}

Your reasoning process should follow the rules below:
- Each `{reasoning_field_name}` (of type `{reasoning_field_type}`) entails {reasoning_field_description}.
- Before writing a new `{reasoning_field_name}`, start with some internal reasoning which discusses and guides what to do with the next `{reasoning_field_name}`.{thought_length_instruction_formatted}{response_length_instruction_formatted}

## Response Format

Once a user provides {input_fields}, your response must follow this exact template:

<thinking>
<step>
## internal_reasoning
Your internal reasoning about the first `{reasoning_field_name}`
## {reasoning_field_name}
The first reasoning step towards producing {output_fields}
</step>
<step>
## internal_reasoning
Your internal reasoning about the second `{reasoning_field_name}`
## {reasoning_field_name}
The second reasoning step towards producing {output_fields}
</step>
...
<step>
## internal_reasoning
Your internal reasoning about the final `{reasoning_field_name}`
## {reasoning_field_name}
The final reasoning step towards producing {output_fields}
</step>
</thinking>
<answer>
{output_field_sections}
</answer>
"""

FINAL_OUTPUT_SYNTHESIS_STRICT = """
Your final answer must include the full text from all reasoning steps, copied nearly word-for-word and in sequential order.

- Preserve the exact wording, phrasing, structure, and examples from each reasoning step.
- Maintain the original order and logical flow exactly as provided.
- You may add only:
  - A brief introduction and/or conclusion.
  - Short transitional phrases to connect steps smoothly.
- Do NOT rewrite, paraphrase, summarize, or restructure any reasoning step.
- Do NOT add new ideas, arguments, facts, or examples that did not appear in the reasoning steps.

Your goal is to produce a coherent, readable final answer that is essentially a verbatim synthesis of the reasoning steps with minimal connective tissue.
""".strip()


FINAL_OUTPUT_SYNTHESIS_FAITHFUL = """
Your final answer must remain highly faithful to the reasoning steps and their underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and structure of the original reasoning, while allowing minimal refinement.
""".strip()


FINAL_OUTPUT_SYNTHESIS_RESTRUCTURED = """
Your final answer should preserve the same core ideas and reasoning from the steps provided, while improving clarity and coherence.

- Maintain the essential arguments and logical intent.
- You may rephrase, reorganize, and restructure the content for better flow and readability.
- The overall set of ideas should remain the same, but the presentation may differ.
- Do NOT introduce new ideas or factual content beyond what appears in the reasoning steps.

Your goal is to produce a well-structured synthesis that faithfully reflects the original reasoning while optimizing expression and organization.
""".strip()


FINAL_OUTPUT_CONCLUSION = """
Your final answer must be a standalone response to the user's task and instructions.

- Focus on producing a clear, logically consistent, and high-quality final answer.
- You are not required to preserve the structure, wording, or order of the reasoning steps (between <thinking>...</thinking> tags).
- Use the reasoning steps only as internal guidance; do NOT mention them or refer to them.
- The user will *not* have access to the reasoning steps you wrote, so referencing them is confusing and unhelpful. The user will only see what you write between <answer>...</answer> tags.
- Do NOT explain what you are going to do; just produce the final deliverable. While reasoning was meant for planning, the final output should be a standalone response to the user's task and instructions.
- If the task requires strict formatting (math, formatting specifications for text, code, etc.), follow those requirements exactly in the final output.

Your goal is to output only the final answer content that satisfies the user's instructions.
""".strip()


####################
### User Prompts ###
####################

USER_DEMO_PROMPT = """
{task_instructions}

{formatted_input_fields}
"""

#########################
### Assistant Prompts ###
#########################

ASSISTANT_DEMO_PROMPT = """
<thinking>
{formatted_reasoning_steps}
</thinking>
<answer>
{formatted_answer}
</answer>
"""
