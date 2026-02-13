"""Comprehensive unit tests for GenerativeLocalVLLM class.

We include both unit tests with mocked dependencies and integration tests that require a GPU.

Response format for MockGenerativeLocalVLLM (strict):
    list[list[list[str]]] = layers[requests[completions]]

Expected usage:

```bash
pytest lm/test_generative_local_lm.py -vv
```
"""

# Standard library imports
import logging
import os
from collections.abc import Generator
from typing import Any

# Third-party imports
import pytest
import torch
from vllm import SamplingParams

# Local imports
from constants import Verbosity
from lm.generative_local_lm import (
    ChatCompletionResponse,
    Choice,
    GenerativeLocalVLLM,
    ModelExecutionError,
    Usage,
)
from lm.lm_constants import ENABLE_THINKING
from utilities_for_tests import MockGenerativeLocalVLLM

logger = logging.getLogger(__name__)

GENERATOR_ARGUMENT_SYSTEM_PROMPT = """
# Instructions

Generate an argument which takes the provided stance towards the provided topic.

Your inputs will be:
1. `topic` (str): The topic to generate an argument about
2. `stance` (Literal['PRO', 'ANTI']): The stance to take on the topic

Your goal is to produce the following output:
`argument` (str): The generated argument

When solving this problem, you must break down your solution into a series of
reasoning steps, followed by a final answer.
Each step towards the answer should be encased within <step>...</step> tags, and
contain a `claim` that advances the solution towards producing `argument`.

Your final answer must remain highly faithful to the reasoning steps and their
underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must
  remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and
structure of the original reasoning, while allowing minimal refinement.

Your reasoning process should follow the rules below:
- Each `claim` (of type `str`) entails a component of the argument that advocates
  for the given stance towards the topic.
- Your final answer should be between 1 and 3 sentences.

## Response Format

Once a user provides `topic` and `stance`, your response must follow this exact
template:

<thinking>
<step>
## claim
The first reasoning step towards producing `argument`
</step>
<step>
## claim
The second reasoning step towards producing `argument`
</step>
...
<step>
## claim
The final reasoning step towards producing `argument`
</step>
</thinking>
<answer>
## argument
Your response for `argument` here
</answer>
""".strip()

EVALUATOR_PRM_SYSTEM_PROMPT = """
Judge the quality of reasoning steps for a problem-solving process.
The task requires producing `argument` given `topic` and `stance`.
Reasoning steps towards producing `argument` are provided in `reasoning_steps`.

Since you are evaluating intermediate reasoning, don't score based on
completeness, and instead score based on the rubric items below:
- soundness: Logical validity, factual accuracy, and coherence with prior steps
  (a float between 0.0 and 10.0).
- promise: Likelihood of the reasoning to lead to a strong final answer (a float
  between 0.0 and 10.0).

Please provide your response with each output field under its own header using
the format:
## field_name
Your response for that field here
""".strip()

EVALUATOR_ORM_SYSTEM_PROMPT = """
Judge the quality of the final answer for the problem at hand.
The problem requires producing `argument` given `topic` and `stance`.
Consider correctness, completeness, clarity, and overall solution quality.
Make sure to fact check claims or aspects that don't seem immediately obvious,
may be untrue, or may contain a mistake.
Penalize any omissions, inaccuracies, or inclusion of irrelevant information in
the final solution.
Rubric:
- quality: Holistic final-answer quality: correctness, completeness, and clarity
  with respect to the prompt. Penalize wrong answers, missing required elements,
  and irrelevant content.

Assign numeric quality scores to each of the output fields: quality.

Please provide your response with each output field under its own header using
the format:
## field_name
Your response for that field here
""".strip()

CONTROLLER_SYSTEM_PROMPT_1D_NON_NATIVE = """
Generate an argument which takes the provided stance towards the provided topic.

You are given `topic` and `stance` and your goal is to finish with `argument`.
To accomplish this goal, you will need to reason about the problem step by step
rather than generating `argument` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason
about the problem before generating `argument`.
Refer to the existing reasoning steps under the `reasoning` header to inform
your next step.
Reasoning steps are ordered sequentially, and each one includes a `claim` header
above the content of the step itself.
Choose a tool to use from the following options:
(1) `intervene_on_next_reasoning_step`: choose one causal structure.
(2) `finish`: generate the final output next.

Please provide your response with each output field under its own header using
the format:
## field_name
Your response for that field here
""".strip()

CONTROLLER_SYSTEM_PROMPT_2D_NON_NATIVE = """
Generate an argument which takes the provided stance towards the provided topic.

You are given `topic` and `stance` and your goal is to finish with `argument`.
Choose a tool to use from the following options:
(1) `intervene_on_next_reasoning_step`: choose structure + subtopic.
(2) `finish`: generate the final output next.

Please provide your response with each output field under its own header using
the format:
## field_name
Your response for that field here
""".strip()

CONTROLLER_SYSTEM_PROMPT_3D_NON_NATIVE = """
Generate an argument which takes the provided stance towards the provided topic.

You are given `topic` and `stance` and your goal is to finish with `argument`.
Choose a tool to use from the following options:
(1) `intervene_on_next_reasoning_step`: choose structure + subtopic + style.
(2) `finish`: generate the final output next.

Please provide your response with each output field under its own header using
the format:
## field_name
Your response for that field here
""".strip()

CONTROLLER_SYSTEM_PROMPT_NATIVE = """
You are a controller that selects the next action via tool calling.
Given the topic, stance, and existing reasoning, call a tool to either
intervene on the next reasoning step or finish.
""".strip()


def _clip_for_log(text: str, max_chars: int = 6000) -> str:
    """Clip long strings to keep logs readable."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 40] + "\n\t...[truncated]...\n" + text[-30:]


def _log_messages_batch(case_id: str, messages: list[list[dict[str, str]]]) -> None:
    """Log the full rendered messages for a batch."""
    logger.info(f"=== CASE: {case_id} ===")
    logger.info(f"\tbatch_size={len(messages)}")
    for conv_idx, conv in enumerate(messages):
        logger.info(f"\t--- conversation[{conv_idx}] ---")
        for msg_idx, msg in enumerate(conv):
            role = msg.get("role", "")
            content = msg.get("content", "")
            logger.info(f"\t\t[{msg_idx}] role={role}")
            logger.info(f"\t\t[{msg_idx}] content:\n{_clip_for_log(content)}")


def _log_batch_outputs(case_id: str, responses: list[ChatCompletionResponse]) -> None:
    """Log all outputs for a batch."""
    logger.info(f"=== OUTPUTS: {case_id} ===")
    logger.info(f"\tresponses={len(responses)}")
    for i, resp in enumerate(responses):
        logger.info(
            f"\tresponse[{i}]:\n"
            f"\t\tprompt_tokens={resp.usage.prompt_tokens}\n"
            f"\t\tcompletion_tokens={resp.usage.completion_tokens}\n"
            f"\t\ttotal_tokens={resp.usage.total_tokens}\n"
            f"\t\tchoices={len(resp.choices)}"
        )
        for j, choice in enumerate(resp.choices):
            logger.info(f"\t\tchoice[{j}] finish_reason={choice.finish_reason}")
            logger.info(f"\t\tchoice[{j}] text:\n{_clip_for_log(choice.text or '')}")


# =============================================================================
# Unit Tests (Mocked - No GPU Required)
# =============================================================================


class TestMockGenerativeLocalVLLMBasics:
    """Test basic MockGenerativeLocalVLLM functionality."""

    def test_mock_initialization(self) -> None:
        """Test that MockGenerativeLocalVLLM initializes without errors."""
        # Format: layers[requests[completions]] - 1 layer, 1 request, 1 completion
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Hello, world!"]]])
        assert mock_lm.model_name == "mock-generative-model"
        assert mock_lm.model_path == "mock-generative-model"
        assert mock_lm.history == []

    def test_mock_with_string_response(self) -> None:
        """Test MockGenerativeLocalVLLM with a simple string response."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Test response"]]])
        response = mock_lm.forward(prompt="Hello")
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Test response"

    def test_mock_with_list_response(self) -> None:
        """Test MockGenerativeLocalVLLM with multiple choices."""
        # Format: 1 layer, 1 request, 2 completions
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Response 1", "Response 2"]]])
        response = mock_lm.forward(
            prompt="Hello",
            sampling_params=SamplingParams(n=2),
        )
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == 2
        assert response.choices[0].message.content == "Response 1"
        assert response.choices[1].message.content == "Response 2"


class TestGenerativeLocalVLLMForward:
    """Test forward method for single inputs using MockGenerativeLocalVLLM."""

    @pytest.fixture
    def mock_lm(self) -> MockGenerativeLocalVLLM:
        """Create a MockGenerativeLocalVLLM instance.

        Returns:
            A MockGenerativeLocalVLLM instance with a simple response.
        """
        # Format: layers[requests[completions]]
        return MockGenerativeLocalVLLM(responses=[[["Generated response"]]])

    @pytest.mark.parametrize(
        # Parameter names
        [
            "prompt",
            "messages",
            "expected_content",
        ],
        # Parameter values
        [
            pytest.param(
                "Hello",                            # prompt
                None,                               # messages
                "Generated response",               # expected_content
                id="forward_with_prompt",
            ),
            pytest.param(
                None,                               # prompt
                [                                   # messages
                    {"role": "user", "content": "Hello"}
                ],
                "Generated response",               # expected_content
                id="forward_with_messages",
            ),
        ],
    )
    def test_forward_success(
        self,
        mock_lm: MockGenerativeLocalVLLM,
        prompt: str | None,
        messages: list[dict[str, str]] | None,
        expected_content: str,
    ) -> None:
        """Test forward method returns ChatCompletionResponse.

        Args:
            mock_lm: MockGenerativeLocalVLLM instance.
            prompt: Prompt string to test.
            messages: Messages to test.
            expected_content: Expected content in the response.
        """
        response = mock_lm.forward(prompt=prompt, messages=messages)

        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == 1
        assert response.choices[0].message.content == expected_content

    def test_forward_with_sampling_params(self, mock_lm: MockGenerativeLocalVLLM) -> None:
        """Test forward method with custom sampling parameters.

        Args:
            mock_lm: MockGenerativeLocalVLLM instance.
        """
        sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        response = mock_lm.forward(prompt="Hello", sampling_params=sampling_params)

        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == 1

    def test_forward_raises_model_execution_error(self) -> None:
        """Test that pre-execution errors (e.g., prompt too long) raise ModelExecutionError."""
        err = ValueError(
            "The decoder prompt (length 20000) is longer than the maximum model length of 16384."
        )
        mock_lm = MockGenerativeLocalVLLM(responses=None, chat_exception=[[err]])
        with pytest.raises(ModelExecutionError, match="decoder prompt"):
            mock_lm.forward(prompt="Hello")


class TestGenerativeLocalVLLMBatch:
    """Test batch method for multiple inputs using MockGenerativeLocalVLLM."""

    @pytest.fixture
    def mock_lm(self) -> MockGenerativeLocalVLLM:
        """Create a MockGenerativeLocalVLLM instance for batch testing.

        Returns:
            A MockGenerativeLocalVLLM instance with batch responses.
        """
        # Format: 1 layer, 2 requests, 1 completion each
        return MockGenerativeLocalVLLM(responses=[
            [["Response 1"], ["Response 2"]],
        ])

    @pytest.mark.parametrize(
        # Parameter names
        [
            "prompts",
            "messages",
            "expected_count",
        ],
        # Parameter values
        [
            pytest.param(
                ["Hello", "Hi"],                    # prompts
                None,                               # messages
                2,                                  # expected_count
                id="batch_with_prompts",
            ),
            pytest.param(
                None,                               # prompts
                [                                   # messages
                    [
                        {"role": "user", "content": "Hello"}
                    ],
                    [
                        {"role": "user", "content": "Hi"}
                    ],
                ],
                2,                                  # expected_count
                id="batch_with_messages",
            ),
        ],
    )
    def test_batch_success(
        self,
        mock_lm: MockGenerativeLocalVLLM,
        prompts: list[str] | None,
        messages: list[list[dict[str, str]]] | None,
        expected_count: int,
    ) -> None:
        """Test batch method returns list of ChatCompletionResponse.

        Args:
            mock_lm: MockGenerativeLocalVLLM instance.
            prompts: Prompts to test.
            messages: Messages to test.
            expected_count: Expected number of responses.
        """
        responses = mock_lm.batch(prompts=prompts, messages=messages)

        assert isinstance(responses, list)
        assert len(responses) == expected_count
        for response in responses:
            assert isinstance(response, ChatCompletionResponse)

    def test_batch_with_sampling_params_list(self) -> None:
        """Test batch method with list of sampling parameters."""
        # Format: 1 layer, 2 requests, 1 completion each
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Response 1"], ["Response 2"]],
        ])
        sampling_params = [
            SamplingParams(temperature=0.5),
            SamplingParams(temperature=0.9),
        ]
        responses = mock_lm.batch(prompts=["Hello", "Hi"], sampling_params=sampling_params)

        assert len(responses) == 2


class TestGenerativeLocalVLLMCall:
    """Test __call__ method dispatching using MockGenerativeLocalVLLM."""

    def test_call_with_prompt(self) -> None:
        """Test __call__ with prompt dispatches to forward."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Hello response"]]])
        response = mock_lm(prompt="Hello")

        assert isinstance(response, ChatCompletionResponse)

    def test_call_with_single_messages(self) -> None:
        """Test __call__ with single message thread dispatches to forward."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Messages response"]]])
        response = mock_lm(messages=[{"role": "user", "content": "Hello"}])

        assert isinstance(response, ChatCompletionResponse)

    def test_call_with_batch_messages(self) -> None:
        """Test __call__ with batch messages dispatches to batch."""
        # Format: 1 layer, 2 requests, 1 completion each
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Response 1"], ["Response 2"]],
        ])
        responses = mock_lm(messages=[
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
        ])

        assert isinstance(responses, list)
        assert len(responses) == 2


class TestGenerativeLocalVLLMUsage:
    """Test usage statistics building using MockGenerativeLocalVLLM."""

    def test_response_has_usage(self) -> None:
        """Test that responses include usage statistics."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Test response"]]])
        response = mock_lm.forward(prompt="Hello")

        assert isinstance(response.usage, Usage)
        # MockGenerativeLocalVLLM calculates usage based on word count
        assert response.usage.completion_tokens >= 0
        assert response.usage.total_tokens >= 0

    def test_usage_defaults_when_missing(self) -> None:
        """Test that usage defaults to zeros when token fields are missing."""
        usage = Usage.model_validate({})
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestGenerativeLocalVLLMChoices:
    """Test choice building using MockGenerativeLocalVLLM."""

    def test_multiple_choices(self) -> None:
        """Test that multiple choices are correctly built."""
        # Format: 1 layer, 1 request, 3 completions
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Choice 1", "Choice 2", "Choice 3"]]])
        response = mock_lm.forward(prompt="Hello", sampling_params=SamplingParams(n=3))

        assert len(response.choices) == 3
        for i, choice in enumerate(response.choices):
            assert isinstance(choice, Choice)
            assert choice.index == i
            assert choice.message.content == f"Choice {i + 1}"


class TestGenerativeLocalVLLMModelName:
    """Test model_name property using MockGenerativeLocalVLLM."""

    def test_model_name_property(self) -> None:
        """Test model_name property returns expected value."""
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Test"]]])
        assert mock_lm.model_name == "mock-generative-model"


class TestGenerativeLocalVLLMValidation:
    """Test input validation using the real GenerativeLocalVLLM validation methods."""

    @pytest.fixture
    def mock_lm(self) -> MockGenerativeLocalVLLM:
        """Create a MockGenerativeLocalVLLM instance.

        Returns:
            A MockGenerativeLocalVLLM instance.
        """
        return MockGenerativeLocalVLLM(responses=[[["Test"]]])

    @pytest.mark.parametrize(
        # Parameter names
        [
            "prompt",
            "messages",
            "expected_exception",
            "expected_message",
        ],
        # Parameter values
        [
            pytest.param(
                "Hello",                            # prompt
                None,                               # messages
                None,                               # expected_exception
                None,                               # expected_message
                id="valid_prompt_only",
            ),
            pytest.param(
                None,                               # prompt
                [{"role": "user", "content": "Hello"}],  # messages
                None,                               # expected_exception
                None,                               # expected_message
                id="valid_messages_only",
            ),
            pytest.param(
                None,                               # prompt
                None,                               # messages
                ValueError,                         # expected_exception
                "Must provide either prompt or messages",  # expected_message
                id="invalid_neither_prompt_nor_messages",
            ),
        ],
    )
    def test_forward_validation(
        self,
        mock_lm: MockGenerativeLocalVLLM,
        prompt: str | None,
        messages: list[dict[str, str]] | None,
        expected_exception: type[Exception] | None,
        expected_message: str | None,
    ) -> None:
        """Test forward method validates inputs correctly.

        Args:
            mock_lm: MockGenerativeLocalVLLM instance.
            prompt: Prompt string to test.
            messages: Messages to test.
            expected_exception: Expected exception type.
            expected_message: Expected exception message substring.
        """
        if expected_exception is not None:
            with pytest.raises(expected_exception, match=expected_message):
                mock_lm.forward(prompt=prompt, messages=messages)
        else:
            response = mock_lm.forward(prompt=prompt, messages=messages)
            assert isinstance(response, ChatCompletionResponse)


class TestGenerativeLocalVLLMMultipleOutputs:
    """Test multiple outputs (n > 1) using MockGenerativeLocalVLLM."""

    def test_multiple_outputs_single_call(self) -> None:
        """Test generating multiple outputs in a single call."""
        # Format: 1 layer, 1 request, 3 completions
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Output 1", "Output 2", "Output 3"]],
        ])
        response = mock_lm.forward(
            prompt="Generate three outputs",
            sampling_params=SamplingParams(n=3),
        )

        assert len(response.choices) == 3
        assert response.choices[0].message.content == "Output 1"
        assert response.choices[1].message.content == "Output 2"
        assert response.choices[2].message.content == "Output 3"

    def test_batch_with_different_n_values(self) -> None:
        """Test batch processing with different n values per request."""
        # Format: 1 layer, 2 requests with different number of completions
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Single output"], ["Output A", "Output B"]],
        ])
        sampling_params = [
            SamplingParams(n=1),
            SamplingParams(n=2),
        ]
        responses = mock_lm.batch(
            prompts=["First", "Second"],
            sampling_params=sampling_params,
        )

        assert len(responses) == 2
        assert len(responses[0].choices) == 1
        assert len(responses[1].choices) == 2


class TestGenerativeLocalVLLMLayeredResponses:
    """Test layered responses for sequential calls using MockGenerativeLocalVLLM."""

    def test_layered_responses(self) -> None:
        """Test that mock returns different responses for sequential calls."""
        # Format: 2 layers, each with 1 request, 1 completion
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["First call response"]],
            [["Second call response"]],
        ])

        # First call
        response1 = mock_lm.forward(prompt="Call 1")
        assert response1.choices[0].message.content == "First call response"

        # Second call
        response2 = mock_lm.forward(prompt="Call 2")
        assert response2.choices[0].message.content == "Second call response"

    def test_layered_batch_responses(self) -> None:
        """Test layered responses for batch calls."""
        # Format: 2 layers, each with 2 requests, 1 completion each
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Batch 1 Response 1"], ["Batch 1 Response 2"]],
            [["Batch 2 Response 1"], ["Batch 2 Response 2"]],
        ])

        # First batch call
        responses1 = mock_lm.batch(prompts=["A", "B"])
        assert responses1[0].choices[0].message.content == "Batch 1 Response 1"
        assert responses1[1].choices[0].message.content == "Batch 1 Response 2"

        # Second batch call
        responses2 = mock_lm.batch(prompts=["C", "D"])
        assert responses2[0].choices[0].message.content == "Batch 2 Response 1"
        assert responses2[1].choices[0].message.content == "Batch 2 Response 2"


class TestGenerativeLocalVLLMResponseReset:
    """Test response reset functionality using MockGenerativeLocalVLLM."""

    def test_set_responses(self) -> None:
        """Test setting new responses."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Initial response"]]])
        response1 = mock_lm.forward(prompt="Test")
        assert response1.choices[0].message.content == "Initial response"

        # Set new responses
        mock_lm.set_responses([[["New response"]]])
        response2 = mock_lm.forward(prompt="Test")
        assert response2.choices[0].message.content == "New response"

    def test_reset_responses(self) -> None:
        """Test resetting responses to empty."""
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Test"]]])
        mock_lm.reset_responses()

        with pytest.raises(ModelExecutionError, match="No chat responses"):
            mock_lm.forward(prompt="Test")


# =============================================================================
# Integration Tests (GPU Required)
# =============================================================================

# GPU Skip Marker
pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU tests require GPU access",
)


@pytestmark_gpu
class TestGenerativeLocalVLLMIntegration:
    """Integration tests for GenerativeLocalVLLM using real models (requires GPU).

    These tests instantiate actual models and verify that generation works correctly
    with reasonable inputs. They verify that token counts in Usage objects are > 0.
    """

    @pytest.fixture(scope="class")
    def shared_gpu_model(self) -> Generator[GenerativeLocalVLLM, None, None]:
        """Shared GenerativeLocalVLLM fixture for all GPU integration tests.

        This fixture loads a model once and shares it across all GPU test methods
        to avoid loading multiple models and running out of GPU memory.

        Yields:
            GenerativeLocalVLLM: A real GenerativeLocalVLLM instance.
        """
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        base_path = "/projects/BSTEWART/model_storage"
        model_name = "Qwen3-4B-Instruct-2507"
        model_path = os.path.join(base_path, model_name)

        lm = None
        try:
            logger.info(f"Initializing shared GPU model from: {model_path}")
            lm = GenerativeLocalVLLM(
                model=model_path,
                tensor_parallel_size=1,
                dtype="auto",
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                enforce_eager=True,
                verbosity=Verbosity.INFO,
            )
            logger.info("Shared GPU model initialized successfully")
            yield lm
        finally:
            # Cleanup after all GPU tests complete
            if lm is not None:
                logger.info("Cleaning up shared GPU model...")
                lm.kill()

    @pytest.mark.parametrize(
        # Parameter names
        [
            "case_id",
            "prompts",
            "messages",
            "sampling_params",
            "use_tqdm",
            "chat_template",
            "chat_template_content_format",
            "continue_final_message",
            "tools",
            "chat_template_kwargs",
            "extra_kwargs",
        ],
        [
            pytest.param(
                "generator_first_step_batch",       # case_id
                None,                               # prompts
                [                                   # messages
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {"role": "assistant", "content": "<thinking>\n<step>\n## claim\nTherefore"},
                    ],
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {"role": "assistant", "content": "<thinking>\n<step>\n## claim\nHowever"},
                    ],
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {
                            "role": "assistant",
                            "content": "<thinking>\n<step>\n## claim\nEvidence shows that",
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {"role": "assistant", "content": "<thinking>\n<step>\n## claim\nFor example"},
                    ],
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {"role": "assistant", "content": "<thinking>\n<step>\n## claim\nNext"},
                    ],
                ],
                SamplingParams(  # sampling_params
                    temperature=0.7,
                    max_tokens=220,
                    stop=["</step>"],
                    include_stop_str_in_output=True,
                ),
                False,  # use_tqdm
                None,  # chat_template
                "auto",  # chat_template_content_format
                True,  # continue_final_message
                None,  # tools
                {ENABLE_THINKING: False},  # chat_template_kwargs
                {},  # extra_kwargs
                id="generator_first_step_batch",
            ),
            pytest.param(
                "generator_middle_layer_batch",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {
                            "role": "assistant",
                            "content": """
<thinking>
<step>
## internal_reasoning
I should quantify and compare costs, benefits, and real-world impacts across economic, social,
and environmental dimensions.
## claim
Therefore UBI can stabilize households facing volatile income, reducing stress and improving
decision-making.
</step>
<step>
## claim
However
""".strip(),
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {
                            "role": "assistant",
                            "content": """
<thinking>
<step>
## internal_reasoning
I should consider inalienable human rights, civil liberties, privacy protections, and the
freedom to make one's own choices.
## claim
Moreover UBI can protect autonomy by giving people the freedom to refuse exploitative work.
</step>
<step>
## claim
Evidence shows that
""".strip(),
                        },
                    ],
                ],
                SamplingParams(
                    temperature=0.7,
                    max_tokens=220,
                    stop=["</step>"],
                    include_stop_str_in_output=True,
                ),
                False,
                None,
                "auto",
                True,
                None,
                {ENABLE_THINKING: False},
                {},
                id="generator_middle_layer_batch",
            ),
            pytest.param(
                "generator_final_responses_batch",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {
                            "role": "assistant",
                            "content": """
<thinking>
<step>
## internal_reasoning
I should quantify and compare costs, benefits, and real-world impacts across economic, social,
and environmental dimensions.
## claim
Therefore UBI can reduce poverty by ensuring a stable income floor, strengthening resilience.
</step>
<step>
## internal_reasoning
I should identify risks, unintended outcomes, cascading effects, and potential for escalation.
## claim
Evidence shows that pilot programs reduce income volatility without large employment drops.
</step>
<step>
## internal_reasoning
I should analyze whether outcomes, processes, and distributions are fair to all parties involved.
## claim
However UBI must be designed to be fiscally sustainable and equitable across regions.
</step>
</thinking>
<answer>
## argument
""".strip(),
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {
                            "role": "assistant",
                            "content": """
<thinking>
<step>
## internal_reasoning
I should consider legal frameworks, accountability mechanisms, democratic principles, and proper
authority.
## claim
For example UBI can simplify welfare administration, reducing bureaucratic overhead and error.
</step>
<step>
## internal_reasoning
I should examine what behaviors are encouraged, who holds power, and how interests align or
conflict.
## claim
Moreover unconditional cash reduces coercive dependence on low-wage employers, improving
bargaining power.
</step>
<step>
## internal_reasoning
I should evaluate whether the proposal can actually be implemented and enforced effectively.
## claim
Importantly a feasible UBI requires clear funding mechanisms and integration with existing
programs.
</step>
</thinking>
<answer>
## argument
""".strip(),
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": GENERATOR_ARGUMENT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
Generate an argument which takes the provided stance towards the provided topic.

## topic
The government should provide universal basic income.

## stance
PRO
""".strip(),
                        },
                        {
                            "role": "assistant",
                            "content": """
<thinking>
<step>
## internal_reasoning
I should evaluate historical precedents, long-term vs short-term tradeoffs, and obligations to
future generations.
## claim
Next UBI can support long-term adaptation to automation by smoothing income shocks.
</step>
<step>
## internal_reasoning
I should evaluate actions based on moral rules, character virtues, relationships, and ethical
obligations.
## claim
In other words providing a basic income floor aligns with ethical duties to prevent destitution.
</step>
<step>
## internal_reasoning
I should stay composed and speak with certainty to establish credibility and appear reasonable.
## claim
In conclusion UBI can reduce poverty and volatility while requiring careful fiscal design.
</step>
</thinking>
<answer>
## argument
""".strip(),
                        },
                    ],
                ],
                SamplingParams(
                    temperature=0.3,
                    max_tokens=320,
                    stop=["</answer>"],
                    include_stop_str_in_output=True,
                ),
                False,
                None,
                "auto",
                True,
                None,
                {ENABLE_THINKING: False},
                {},
                id="generator_final_responses_batch",
            ),
            pytest.param(
                "evaluator_prm_first_layer",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": EVALUATOR_PRM_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning_steps
["Therefore UBI can reduce poverty by guaranteeing a minimum income floor."]

Respond with the corresponding output fields, starting with the field
`## soundness`
`## promise`
""".strip(),
                        },
                    ]
                ],
                SamplingParams(temperature=0.1, max_tokens=140),
                False,
                None,
                "auto",
                False,
                None,
                {ENABLE_THINKING: False},
                {},
                id="evaluator_prm_first_layer",
            ),
            pytest.param(
                "evaluator_prm_third_layer_of_four",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": EVALUATOR_PRM_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning_steps
[
  "## internal_reasoning\nI should quantify impacts.\n## claim\nEvidence shows that pilot programs reduce poverty volatility.",
  "## internal_reasoning\nI should address fairness.\n## claim\nHowever targeted welfare can miss eligible recipients.",
  "## internal_reasoning\nI should add implementation detail.\n## claim\nNext a simple UBI can reduce bureaucracy and stigma."
]

Respond with the corresponding output fields, starting with the field
`## soundness`
`## promise`
""".strip(),
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": EVALUATOR_PRM_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning_steps
[
  "## internal_reasoning\nI should frame incentives.\n## claim\nIn other words UBI can let workers refuse exploitative jobs.",
  "## internal_reasoning\nI should cite precedent.\n## claim\nFor example Alaska’s dividend shows a workable universal benefit.",
  "## internal_reasoning\nI should summarize.\n## claim\nImportantly stabilizing consumption can soften recessions."
]

Respond with the corresponding output fields, starting with the field
`## soundness`
`## promise`
""".strip(),
                        },
                    ],
                ],
                SamplingParams(temperature=0.1, max_tokens=160),
                False,
                None,
                "auto",
                False,
                None,
                {ENABLE_THINKING: False},
                {},
                id="evaluator_prm_third_layer_of_four",
            ),
            pytest.param(
                "evaluator_orm_early_stop",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": EVALUATOR_ORM_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## argument
UBI can reduce poverty and insecurity by guaranteeing a stable income floor.

## reasoning_steps
["Therefore UBI can reduce poverty by guaranteeing a minimum income floor."]

Respond with the corresponding output fields, starting with the field
`## quality`
""".strip(),
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": EVALUATOR_ORM_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## argument
UBI can reduce poverty, simplify welfare, and increase bargaining power, while requiring careful
fiscal design.

## reasoning_steps
[
  "Evidence shows that pilots can reduce income volatility and stress.",
  "However fiscal costs require tradeoffs and efficient tax design."
]

Respond with the corresponding output fields, starting with the field
`## quality`
""".strip(),
                        },
                    ],
                ],
                SamplingParams(temperature=0.1, max_tokens=160),
                False,
                None,
                "auto",
                False,
                None,
                {ENABLE_THINKING: False},
                {},
                id="evaluator_orm_early_stop",
            ),
            pytest.param(
                "controller_1d_non_native",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_1D_NON_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning

## number_of_additional_reasoning_steps
4

Respond with the corresponding output fields, starting with the field
`## considerations`
`## action`
`## action_arguments`
""".strip(),
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_1D_NON_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning
<thinking>
<step>
## internal_reasoning
I should quantify and compare costs, benefits, and real-world impacts across economic, social,
and environmental dimensions.
## claim
Evidence shows that UBI pilots reduce poverty volatility and stress for recipients.
</step>

## number_of_additional_reasoning_steps
3

Respond with the corresponding output fields, starting with the field
`## considerations`
`## action`
`## action_arguments`
""".strip(),
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_1D_NON_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning
<thinking>
<step>
## internal_reasoning
I should consider inalienable human rights, civil liberties, privacy protections, and the
freedom to make one's own choices.
## claim
However critics worry that unconditional payments reduce work incentives, so design matters.
</step>
<step>
## internal_reasoning
I should evaluate whether the proposal can actually be implemented and enforced effectively.
## claim
Next UBI could be phased in with tax offsets and benefit integration to control costs.
</step>

## number_of_additional_reasoning_steps
2

Respond with the corresponding output fields, starting with the field
`## considerations`
`## action`
`## action_arguments`
""".strip(),
                        },
                    ],
                ],
                SamplingParams(temperature=0.7, max_tokens=200),
                False,
                None,
                "auto",
                False,
                None,
                {ENABLE_THINKING: False},
                {},
                id="controller_1d_non_native",
            ),
            pytest.param(
                "controller_1d_native",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning

## number_of_additional_reasoning_steps
4
""".strip(),
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning
<thinking>
<step>
## internal_reasoning
I should quantify and compare costs, benefits, and real-world impacts across economic, social,
and environmental dimensions.
## claim
Evidence shows that UBI pilots reduce poverty volatility.
</step>

## number_of_additional_reasoning_steps
3
""".strip(),
                        },
                    ],
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning
<thinking>
<step>
## internal_reasoning
I should consider inalienable human rights, civil liberties, privacy protections, and the
freedom to make one's own choices.
## claim
However critics worry about costs and work incentives.
</step>
<step>
## internal_reasoning
I should evaluate whether the proposal can actually be implemented and enforced effectively.
## claim
Next UBI could be funded via taxes and offsets to fragmented benefits.
</step>

## number_of_additional_reasoning_steps
2
""".strip(),
                        },
                    ],
                ],
                SamplingParams(temperature=0.7, max_tokens=200),
                False,
                None,
                "auto",
                False,
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "intervene_on_next_reasoning_step",
                            "description": (
                                "Select a causal structure to guide the next reasoning step."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "causal_structures": {
                                        "type": "string",
                                        "enum": [
                                            "causal_reasoning",
                                            "conditional",
                                            "concession_and_contrast",
                                            "addition_and_elaboration",
                                            "evidence_and_authority",
                                            "exemplification",
                                            "clarification_and_specification",
                                            "emphasis_and_evaluation",
                                            "sequence_and_transition",
                                            "conclusion_and_summary",
                                        ],
                                    }
                                },
                                "required": ["causal_structures"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": (
                                "Signals that the reasoning so far is sufficient and the next step should "
                                "generate the final output."
                            ),
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ],
                {ENABLE_THINKING: False},
                {},
                id="controller_1d_native",
            ),
            pytest.param(
                "controller_2d_non_native",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_2D_NON_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning

## number_of_additional_reasoning_steps
4

Respond with the corresponding output fields, starting with the field
`## considerations`
`## action`
`## action_arguments`
""".strip(),
                        },
                    ],
                ],
                SamplingParams(temperature=0.7, max_tokens=200),
                False,
                None,
                "auto",
                False,
                None,
                {ENABLE_THINKING: False},
                {},
                id="controller_2d_non_native",
            ),
            pytest.param(
                "controller_2d_native",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning

## number_of_additional_reasoning_steps
4
""".strip(),
                        },
                    ],
                ],
                SamplingParams(temperature=0.7, max_tokens=200),
                False,
                None,
                "auto",
                False,
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "intervene_on_next_reasoning_step",
                            "description": (
                                "Select a causal structure and causal subtopic to guide the next reasoning step."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "causal_structures": {
                                        "type": "string",
                                        "enum": [
                                            "causal_reasoning",
                                            "conditional",
                                            "concession_and_contrast",
                                            "addition_and_elaboration",
                                            "evidence_and_authority",
                                            "exemplification",
                                            "clarification_and_specification",
                                            "emphasis_and_evaluation",
                                            "sequence_and_transition",
                                            "conclusion_and_summary",
                                        ],
                                    },
                                    "causal_subtopics": {
                                        "type": "string",
                                        "enum": [
                                            "cost_benefit_and_impact_analysis",
                                            "rights_and_liberties",
                                            "justice_and_fairness",
                                            "ethical_principles",
                                            "governance_and_accountability",
                                            "risk_and_unintended_consequences",
                                            "feasibility_and_implementation",
                                            "incentives_and_power_dynamics",
                                            "precedent_and_long_term_effects",
                                            "stakeholder_responsibility",
                                        ],
                                    },
                                },
                                "required": ["causal_structures", "causal_subtopics"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": (
                                "Signals that the reasoning so far is sufficient and the next step should "
                                "generate the final output."
                            ),
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ],
                {ENABLE_THINKING: False},
                {},
                id="controller_2d_native",
            ),
            pytest.param(
                "controller_3d_non_native",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_3D_NON_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning

## number_of_additional_reasoning_steps
4

Respond with the corresponding output fields, starting with the field
`## considerations`
`## action`
`## action_arguments`
""".strip(),
                        },
                    ],
                ],
                SamplingParams(temperature=0.7, max_tokens=200),
                False,
                None,
                "auto",
                False,
                None,
                {ENABLE_THINKING: False},
                {},
                id="controller_3d_non_native",
            ),
            pytest.param(
                "controller_3d_native",
                None,
                [
                    [
                        {
                            "role": "system",
                            "content": CONTROLLER_SYSTEM_PROMPT_NATIVE,
                        },
                        {
                            "role": "user",
                            "content": """
## topic
The government should provide universal basic income.

## stance
PRO

## reasoning

## number_of_additional_reasoning_steps
4
""".strip(),
                        },
                    ],
                ],
                SamplingParams(temperature=0.7, max_tokens=200),
                False,
                None,
                "auto",
                False,
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "intervene_on_next_reasoning_step",
                            "description": (
                                "Select a causal structure, causal subtopic, and causal style to guide the next "
                                "reasoning step."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "causal_structures": {
                                        "type": "string",
                                        "enum": [
                                            "causal_reasoning",
                                            "conditional",
                                            "concession_and_contrast",
                                            "addition_and_elaboration",
                                            "evidence_and_authority",
                                            "exemplification",
                                            "clarification_and_specification",
                                            "emphasis_and_evaluation",
                                            "sequence_and_transition",
                                            "conclusion_and_summary",
                                        ],
                                    },
                                    "causal_subtopics": {
                                        "type": "string",
                                        "enum": [
                                            "cost_benefit_and_impact_analysis",
                                            "rights_and_liberties",
                                            "justice_and_fairness",
                                            "ethical_principles",
                                            "governance_and_accountability",
                                            "risk_and_unintended_consequences",
                                            "feasibility_and_implementation",
                                            "incentives_and_power_dynamics",
                                            "precedent_and_long_term_effects",
                                            "stakeholder_responsibility",
                                        ],
                                    },
                                    "causal_styles": {
                                        "type": "string",
                                        "enum": [
                                            "figurative_language",
                                            "statistical_and_data_driven",
                                            "narrative_and_anecdote",
                                            "expert_and_authoritative_voice",
                                            "repetition_and_parallelism",
                                            "contrast_and_antithesis",
                                            "measured_and_authoritative_tone",
                                            "passionate_and_urgent_tone",
                                            "direct_engagement",
                                            "scope_and_framing",
                                        ],
                                    },
                                },
                                "required": [
                                    "causal_structures",
                                    "causal_subtopics",
                                    "causal_styles",
                                ],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "description": (
                                "Signals that the reasoning so far is sufficient and the next step should "
                                "generate the final output."
                            ),
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ],
                {ENABLE_THINKING: False},
                {},
                id="controller_3d_native",
            ),
        ],
    )
    def test_batch_realistic_tot_requests(
        self,
        shared_gpu_model: GenerativeLocalVLLM,
        case_id: str,
        prompts: list[str] | None,
        messages: list[list[dict[str, str]]] | None,
        sampling_params: SamplingParams,
        use_tqdm: bool,
        chat_template: str | None,
        chat_template_content_format: str,
        continue_final_message: bool,
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
        extra_kwargs: dict[str, Any],
    ) -> None:
        """Run highly realistic generator/evaluator/controller prompts through lm.batch()."""
        assert prompts is None, "These integration tests should exercise the `messages=...` path."
        assert messages is not None and len(messages) > 0, "Expected a non-empty messages batch."

        _log_messages_batch(case_id=case_id, messages=messages)
        if tools is not None:
            logger.info(f"=== TOOLS: {case_id} ===")
            logger.info(f"\ttools={len(tools)}")

        responses = shared_gpu_model.batch(
            prompts=prompts,
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            continue_final_message=continue_final_message,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            **extra_kwargs,
        )
        _log_batch_outputs(case_id=case_id, responses=responses)
        assert len(responses) == len(messages)
        for resp in responses:
            assert resp.choices, "Expected at least one choice per response."
        logger.info(f"Batch test passed with {len(responses)} responses")

    def test_multiple_completions_integration(
        self, shared_gpu_model: GenerativeLocalVLLM
    ) -> None:
        """Test generation with n > 1 produces multiple completions.

        This test verifies that:
        1. When n > 1, the model generates multiple choices
        2. Each choice has valid content
        3. Usage statistics reflect all completions

        Args:
            shared_gpu_model: Real GenerativeLocalVLLM instance.
        """
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50, n=3)

        response = shared_gpu_model.forward(
            prompt="Give me a creative name for a cat.",
            sampling_params=sampling_params,
        )

        # Verify multiple choices
        assert len(response.choices) == 3, f"Expected 3 choices, got {len(response.choices)}"

        for i, choice in enumerate(response.choices):
            assert choice.message.content is not None, f"Choice {i}: Expected non-None content"
            assert len(choice.message.content) > 0, f"Choice {i}: Expected non-empty content"

        # Verify usage statistics
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

        logger.info(
            f"Multiple completions test passed:\n"
            f"\tnum_choices={len(response.choices)}\n"
            f"\tcompletion_tokens={response.usage.completion_tokens}\n"
            f"\ttotal_tokens={response.usage.total_tokens}"
        )


if __name__ == "__main__":
    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        # Run all tests except GPU-specific ones
        pytest.main([__file__, "-vv"])
    else:
        # If GPU is available, run all tests including GPU-specific ones
        pytest.main([
            __file__,
            "-v",
            "-s",
            "--log-cli-level=INFO",
        ])
