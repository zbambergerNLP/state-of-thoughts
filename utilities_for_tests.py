"""Utilities for testing DSPy reasoning components.

This module provides mock classes for unit testing with mock vLLM LLM objects,
mock tokenizers, mock LocalPredict (dspy.LM) objects, and mock Predict objects.

The mocks use real vLLM output types (RequestOutput, ScoringRequestOutput) directly.

Response formats (no normalization - tests must provide exact format):
    - Chat responses: list[list[list[str]]]. Shape:
		[num_layers, num_input_messages, num_choices_per_input_message]
    - Score responses: list[list[float]]. Shape:
		[num_layers, num_query_document_pairs]
"""

# Standard library imports
import logging
import uuid
from typing import Any

# Third-party imports
import dspy
from vllm import RequestOutput, SamplingParams, ScoringRequestOutput
from vllm.outputs import CompletionOutput, ScoringOutput

# Local imports
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.scoring_local_lm import ScoringLocalVLLM
from predict.local_predict import LocalPredict

logger = logging.getLogger(__name__)


# =============================================================================
# Mock Tokenizer
# =============================================================================


class MockTokenizer:
    """Mock tokenizer that simulates token counting by word splitting."""

    def __call__(self, text: str) -> list[str]:
        """Tokenize text by splitting on whitespace (i.e., splitting whole words)."""
        if not text:
            return []
        return text.split()


# =============================================================================
# Mock vLLM LLM Class
# =============================================================================


class MockVLLM:
    """Mock for vllm.LLM that returns pre-programmed responses.

    This mock simulates the vLLM LLM class behavior without loading actual models.
    It supports layered responses for sequential calls and returns real vLLM output types.

    Response formats (strict - no normalization):
        - chat_responses: list[list[list[str]]] = layers[requests[completions]]
        - score_responses: list[list[float]] = layers[scores]
    """

    def __init__(
        self,
        chat_responses: list[list[list[str]]] | None = None,
        score_responses: list[list[float]] | None = None,
        chat_exception: list[list[Exception | None]] | None = None,
    ) -> None:
        """Initialize MockVLLM with responses.

        Args:
            chat_responses: Layered chat responses.
                Format: layers[requests[completions[text]]]
                Shape: [num_layers, num_threads, num_choices_per_thread]
                In a single `chat()` call, all threads are generated in parallel.
                Each subsequent call to `chat()` will increment the layer index (i.e.,
                process one layer at a time).
            score_responses: Layered score responses.
                Format: layers[scores[float]]
                Shape: [num_layers, num_query_document_pairs]
            chat_exception: Optional per-layer, per-thread exceptions in vLLM format:
                layers[threads]. Each exception applies to the entire thread/request (all
                completions for that thread). Use `None` for "no exception" at a given index.

                Note: vLLM processes a batch of threads in a single `chat()` call. In real code,
                per-thread failures typically surface as `finish_reason="error"` on a subset of
                RequestOutput objects, not as a Python exception. Therefore, per-thread exceptions
                here are modeled as error outputs rather than raising.
        """
        self._chat_responses = chat_responses or []
        self._score_responses = score_responses or []
        self._chat_response_layer_index = 0
        self._score_layer_index = 0
        self._tokenizer = MockTokenizer()
        self._chat_exception = chat_exception

        # Validate that chat_exception matches chat_responses dimensions when both are provided
        if chat_exception is not None and chat_responses is not None and len(chat_responses) > 0:
            if len(chat_exception) != len(chat_responses):
                raise ValueError(
                    f"chat_exception must have the same number of layers as chat_responses. "
                    f"Got {len(chat_exception)} exception layers but {len(chat_responses)} response layers."
                )
            for layer_idx, (response_layer, exception_layer) in enumerate(
                zip(chat_responses, chat_exception, strict=True)
            ):
                if len(exception_layer) != len(response_layer):
                    raise ValueError(
                        f"chat_exception layer {layer_idx} must have the same number of threads "
                        f"as chat_responses layer {layer_idx}. "
                        f"Got {len(exception_layer)} exceptions but {len(response_layer)} responses."
                    )

    def get_tokenizer(self) -> MockTokenizer:
        """Get the mock tokenizer."""
        return self._tokenizer

    def chat(
        self,
        messages: list[list[dict[str, str]]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        use_tqdm: bool = True,
        **kwargs: Any,
    ) -> list[RequestOutput]:
        """Mock the chat method of vLLM.

        Returns real vLLM RequestOutput objects.

        Args:
            messages: List of message threads to generate responses for.
                Shape is [num_threads, num_messages_per_thread].
            sampling_params: Sampling parameters for the language model.
            use_tqdm: Whether to use tqdm for progress bar.
            kwargs: Additional keyword arguments.

        Returns:
            List of RequestOutput objects.
        """
        _, _, _ = sampling_params, use_tqdm, kwargs  # Unused parameters to avoid type errors.

        batch_size = len(messages)
        layer_index = self._chat_response_layer_index

        num_response_layers = len(self._chat_responses)
        num_exception_layers = len(self._chat_exception) if self._chat_exception is not None else 0
        num_layers = max(num_response_layers, num_exception_layers)

        if num_layers == 0:
            raise AssertionError("No chat responses provided for MockVLLM")

        assert layer_index < num_layers, (
            "Not enough chat layers for MockVLLM. "
            f"Requested layer={layer_index}, "
            f"responses_layers={num_response_layers}, "
            f"exception_layers={num_exception_layers}."
        )

        layer: list[list[str]] | None = (
            self._chat_responses[layer_index] if layer_index < num_response_layers else None
        )

        exceptions_for_layer: list[Exception | None] | None = None
        if self._chat_exception is not None and layer_index < len(self._chat_exception):
            exceptions_for_layer = self._chat_exception[layer_index]

        # Build real RequestOutput objects:
        # - Model exceptions are per-request: each request can have an exception that affects all
        #   its completions.
        # - Adapter/parsing exceptions are per-completion, and are determined after the model's
        #   completions are returned.
        results: list[RequestOutput] = []
        for i in range(batch_size):
            # Check if this request has an exception
            request_exception: Exception | None = None
            if exceptions_for_layer is not None and i < len(exceptions_for_layer):
                request_exception = exceptions_for_layer[i]

            # Check if responses exist for this request
            request_responses: list[str] | None = (
                layer[i] if layer is not None and i < len(layer) else None
            )

            # If exception exists and no responses exist for this request, it's a pre-execution error
            if request_exception is not None and request_responses is None:
                raise request_exception

            # If exception exists but responses also exist, it's a mid-execution error.
            # In real vLLM usage, prompt-level failures (e.g., prompt too long) affect the entire
            # request, and therefore all completions for that request. We model that by returning
            # the same error for each completion slot in the request.
            if request_exception is not None:
                assert request_responses is not None
                outputs = [
                    CompletionOutput(
                        index=j,
                        text=str(request_exception),
                        token_ids=[],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="error",
                    )
                    for j in range(len(request_responses))
                ]
            else:
                # No exception: use normal response
                assert request_responses is not None, (
                    "No chat completions or exception provided for MockVLLM. "
                    f"layer={layer_index}, thread={i}, batch_size={batch_size}. "
                    "Provide a completion list in chat_responses[layer][thread] or an "
                    "Exception in chat_exception[layer][thread]."
                )
                outputs = [
                    CompletionOutput(
                        index=j,
                        text=text,
                        token_ids=[],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="stop",
                    )
                    for j, text in enumerate(request_responses)
                ]
            results.append(RequestOutput(
                request_id=f"req_{uuid.uuid4()}",
                prompt=None,
                prompt_token_ids=[],
                prompt_logprobs=None,
                outputs=outputs,
                finished=True,
            ))

        self._chat_response_layer_index += 1
        return results

    def score(
        self,
        queries: list[str],
        documents: list[str],
        use_tqdm: bool = True,
        **kwargs: Any,
    ) -> list[ScoringRequestOutput]:
        """Mock the score method of vLLM.

        Returns real vLLM ScoringRequestOutput objects.
        """
        assert self._score_responses, "No score responses provided for MockVLLM"
        assert self._score_layer_index < len(self._score_responses), (
            f"Not enough score response layers. "
            f"Requested layer {self._score_layer_index}, "
            f"but only {len(self._score_responses)} available."
        )

        layer = self._score_responses[self._score_layer_index]
        num_pairs = len(queries)

        assert num_pairs <= len(layer), (
            f"Not enough scores in layer {self._score_layer_index}. "
            f"Requested {num_pairs} scores, but only {len(layer)} available."
        )

        # Build real ScoringRequestOutput objects
        results = [
            ScoringRequestOutput(
                request_id=f"score_{uuid.uuid4()}",
                outputs=ScoringOutput(score=layer[i]),
                prompt_token_ids=[],
                num_cached_tokens=0,
                finished=True,
            )
            for i in range(num_pairs)
        ]
        self._score_layer_index += 1
        return results

    def set_chat_responses(self, responses: list[list[list[str]]]) -> None:
        """Set new chat responses and reset layer index."""
        self._chat_responses = responses
        self._chat_response_layer_index = 0

    def set_score_responses(self, responses: list[list[float]]) -> None:
        """Set new score responses and reset layer index."""
        self._score_responses = responses
        self._score_layer_index = 0


# =============================================================================
# Mock Language Model Classes
# =============================================================================


class MockGenerativeLocalVLLM(GenerativeLocalVLLM):
    """Mock of GenerativeLocalVLLM that injects MockVLLM instead of real vLLM.

    This mock tests the actual class logic while only mocking the vLLM layer.

    Response format:
        list[list[list[str]]]. Shape:
            [num_layers, num_input_messages, num_choices_per_input_message]
    """

    def __init__(
        self,
        responses: list[list[list[str]]] | None = None,
        chat_exception: list[list[Exception | None]] | None = None,
    ) -> None:
        """Initialize the mock with responses for generation tasks.

        Args:
            responses: Chat responses in vLLM format. Shape:
                [num_layers, num_input_messages, num_choices_per_input_message]
            chat_exception: Exceptions to raise for chat calls. See `MockVLLM` for semantics.
        """
        # Skip parent __init__ to avoid heavy vLLM initialization
        self.history: list[dict[str, Any]] = []
        self.model_path = "mock-generative-model"
        self._model_name = "mock-generative-model"
        self.model_type = "chat"
        self._verbosity = None
        self.kwargs = {"temperature": 0.0, "n": 1}
        self.model = MockVLLM(chat_responses=responses, chat_exception=chat_exception)
        self.tokenizer = self.model.get_tokenizer()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def responses(self) -> list[list[list[str]]]:
        """Get the current responses in vLLM format."""
        return self.model._chat_responses

    def set_responses(self, responses: list[list[list[str]]] | None = None) -> None:
        """Set responses for the mock in vLLM format. Shape:
            [num_layers, num_input_messages, num_choices_per_input_message].

        Args:
            responses: Chat responses in vLLM format. Shape:
                [num_layers, num_input_messages, num_choices_per_input_message].
        """
        self.model.set_chat_responses(responses if responses is not None else [])

    def reset_responses(self) -> None:
        """Reset responses to empty list."""
        self.model.set_chat_responses([])

    def kill(self) -> None:
        """No-op for mock."""
        pass


class MockScoringLocalVLLM(ScoringLocalVLLM):
    """Mock of ScoringLocalVLLM that injects MockVLLM instead of real vLLM.

    This mock tests the actual class logic while only mocking the vLLM layer.
    """

    def __init__(
        self,
        rerank_responses: list[list[float]] | None = None,
    ) -> None:
        """Initialize the mock with rerank responses for scoring tasks.

        Args:
            rerank_responses: Score responses in vLLM format.
                Format: list[list[float]] = layers[scores]
        """
        # Skip parent __init__ to avoid heavy vLLM initialization
        self.history: list[dict[str, Any]] = []
        self.model_path = "mock-scoring-model"
        self._model_name = "mock-scoring-model"
        self._verbosity = None
        self.kwargs: dict[str, Any] = {}
        self.task = "score"

        self.model = MockVLLM(score_responses=rerank_responses)
        self.tokenizer = self.model.get_tokenizer()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def rerank_responses(self) -> list[list[float]]:
        """Get the current rerank responses in vLLM format."""
        return self.model._score_responses

    def set_rerank_responses(self, rerank_responses: list[list[float]] | None = None) -> None:
        """Set rerank responses for the mock."""
        self.model.set_score_responses(rerank_responses or [])

    def reset_rerank_responses(self) -> None:
        """Reset rerank responses to empty list."""
        self.set_rerank_responses()

    def kill(self) -> None:
        """No-op for mock."""
        pass


class MockPredict(LocalPredict):
    """Mock LocalPredict for testing."""

    def __init__(
        self,
        responses: list[list[list[str]]],
        signature: dspy.Signature,
        chat_exception: list[list[Exception | None]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        n: int = 1,
    ) -> None:
        """Initialize MockPredict.

        Args:
            responses: Chat responses in vLLM format. Format: list[list[list[str]]] = layers[requests[completions]]
            signature: DSPy signature for the prediction.
            chat_exception: Exceptions to raise for chat calls. See `MockVLLM` for semantics.
            temperature: Temperature for the language model.
            max_tokens: Maximum number of tokens for the language model.
            n: Number of generations for the language model.
        """
        mock_lm = MockGenerativeLocalVLLM(responses=responses, chat_exception=chat_exception)
        # LocalPredict requires `temperature` to be present in config.
        # We provide deterministic defaults for tests; the underlying MockVLLM ignores them.
        super().__init__(
            signature=signature,
            lm=mock_lm,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        self.lm = mock_lm
        self._responses = responses
        self._signature = signature
