"""Tests for utilities_for_tests.py mock classes.

Tests the mock implementations that simulate vLLM behavior for unit testing.

Response formats (strict):
    - Chat responses: list[list[list[str]]] = layers[requests[completions]]
    - Score responses: list[list[float]] = layers[scores]
"""
# Third-party imports
import dspy
import pytest
from vllm import SamplingParams

# Local imports
from utilities_for_tests import (
    MockGenerativeLocalVLLM,
    MockPredict,
    MockScoringLocalVLLM,
    MockVLLM,
)

# =============================================================================
# Tests for MockVLLM
# =============================================================================


class TestMockVLLM:
    """Tests for MockVLLM exception and response behavior."""

    def test_chat_exception_layers_threads_yields_error_outputs(self) -> None:
        """Test per-layer, per-thread exceptions yield finish_reason='error' for those threads."""
        mock = MockVLLM(
            chat_responses=[[["ok_1"], ["ok_2"]],],
            chat_exception=[[RuntimeError("boom"), None]],
        )
        messages = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
        ]

        outputs = mock.chat(messages=messages, sampling_params=SamplingParams(n=1), use_tqdm=False)
        assert outputs[0].outputs[0].finish_reason == "error"
        assert "boom" in outputs[0].outputs[0].text
        assert outputs[1].outputs[0].finish_reason == "stop"
        assert outputs[1].outputs[0].text == "ok_2"

    def test_chat_exception_validates_dimensions(self) -> None:
        """Test that chat_exception must match chat_responses dimensions."""
        # Test mismatched number of layers
        with pytest.raises(ValueError, match="same number of layers"):
            MockVLLM(
                chat_responses=[[["ok_1"]], [["ok_2"]]],
                chat_exception=[[RuntimeError("boom")]],  # Only 1 layer, but 2 response layers
            )

        # Test mismatched number of threads per layer
        with pytest.raises(ValueError, match="same number of threads"):
            MockVLLM(
                chat_responses=[[["ok_1"], ["ok_2"]]],  # 2 threads
                chat_exception=[[RuntimeError("boom")]],  # Only 1 exception
            )

        # Test valid matching dimensions
        mock = MockVLLM(
            chat_responses=[[["ok_1"], ["ok_2"]]],
            chat_exception=[[RuntimeError("boom"), None]],  # 2 exceptions matching 2 threads
        )
        assert mock._chat_exception is not None
        assert len(mock._chat_exception) == 1
        assert len(mock._chat_exception[0]) == 2

    def test_chat_exception_pre_execution_errors(self) -> None:
        """Test that pre-execution errors (exceptions with no responses) raise immediately."""
        err1 = ValueError("Request 0: prompt too long")
        err2 = ValueError("Request 1: prompt too long")
        mock = MockVLLM(
            chat_responses=None,  # No responses = pre-execution error
            chat_exception=[[err1, err2]],  # Multiple exceptions in same layer
        )
        messages = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
        ]

        # Should raise the first exception encountered
        with pytest.raises(ValueError, match="Request 0: prompt too long"):
            mock.chat(messages=messages, sampling_params=SamplingParams(n=1), use_tqdm=False)

        # Test single pre-execution error still works
        mock2 = MockVLLM(
            chat_responses=None,
            chat_exception=[[err1, None]],  # Only first request fails
        )
        with pytest.raises(ValueError, match="Request 0: prompt too long"):
            mock2.chat(messages=messages, sampling_params=SamplingParams(n=1), use_tqdm=False)

# =============================================================================
# Tests for MockGenerativeLocalVLLM
# =============================================================================


class TestMockGenerativeLocalVLLM:
    """Tests for MockGenerativeLocalVLLM generation functionality."""

    def test_init(self) -> None:
        """Test MockGenerativeLocalVLLM initializes correctly."""
        # Format: layers[requests[completions]]
        responses = [[["test"]]]
        mock = MockGenerativeLocalVLLM(responses=responses)
        assert mock.responses == responses

    def test_init_empty(self) -> None:
        """Test MockGenerativeLocalVLLM initializes with no responses."""
        mock = MockGenerativeLocalVLLM()
        assert mock.responses == []

    def test_set_responses(self) -> None:
        """Test MockGenerativeLocalVLLM.set_responses updates responses."""
        mock = MockGenerativeLocalVLLM()
        new_responses = [[["new"]]]
        mock.set_responses(new_responses)
        assert mock.responses == new_responses

    def test_reset(self) -> None:
        """Test MockGenerativeLocalVLLM.reset_responses clears responses."""
        mock = MockGenerativeLocalVLLM(responses=[[["test"]]])
        mock.reset_responses()
        assert mock.responses == []

    def test_forward(self) -> None:
        """Test MockGenerativeLocalVLLM.forward uses real class logic."""
        # Format: layers[requests[completions]] - 2 layers for 2 calls
        mock = MockGenerativeLocalVLLM(responses=[
            [["response_1"]],  # Layer 0: 1 request, 1 completion
            [["response_2"]],  # Layer 1: 1 request, 1 completion
        ])

        messages = [{"role": "user", "content": "Hello"}]
        sp = SamplingParams(n=1)

        response = mock.forward(messages=messages, sampling_params=sp)
        assert response.choices[0].message.content == "response_1"

    def test_batch(self) -> None:
        """Test MockGenerativeLocalVLLM.batch uses real class logic."""
        # Format: layers[requests[completions]] - 1 layer with 2 requests
        mock = MockGenerativeLocalVLLM(responses=[
            [["r1"], ["r2"]],  # Layer 0: 2 requests, 1 completion each
        ])

        messages = [
            [{"role": "user", "content": "1"}],
            [{"role": "user", "content": "2"}],
        ]
        sp = SamplingParams(n=1)

        responses = mock.batch(messages=messages, sampling_params=sp)
        assert len(responses) == 2
        assert responses[0].choices[0].message.content == "r1"
        assert responses[1].choices[0].message.content == "r2"

    def test_multiple_completions(self) -> None:
        """Test MockGenerativeLocalVLLM with multiple completions per request."""
        # Format: 1 layer, 1 request, 3 completions
        mock = MockGenerativeLocalVLLM(responses=[
            [["comp1", "comp2", "comp3"]],
        ])

        messages = [{"role": "user", "content": "Hello"}]
        sp = SamplingParams(n=3)

        response = mock.forward(messages=messages, sampling_params=sp)
        assert len(response.choices) == 3
        assert response.choices[0].message.content == "comp1"
        assert response.choices[1].message.content == "comp2"
        assert response.choices[2].message.content == "comp3"


# =============================================================================
# Tests for MockScoringLocalVLLM
# =============================================================================


class TestMockScoringLocalVLLM:
    """Tests for MockScoringLocalVLLM scoring functionality."""

    def test_init(self) -> None:
        """Test MockScoringLocalVLLM initializes correctly."""
        # Format: layers[scores]
        responses = [[0.9, 0.1]]
        mock = MockScoringLocalVLLM(rerank_responses=responses)
        assert mock.task == "score"
        assert mock.rerank_responses == responses

    def test_init_empty(self) -> None:
        """Test MockScoringLocalVLLM initializes with no responses."""
        mock = MockScoringLocalVLLM()
        assert mock.rerank_responses == []

    def test_set_responses(self) -> None:
        """Test MockScoringLocalVLLM.set_rerank_responses updates responses."""
        mock = MockScoringLocalVLLM()
        new_responses = [[0.5]]
        mock.set_rerank_responses(new_responses)
        assert mock.rerank_responses == new_responses

    def test_reset(self) -> None:
        """Test MockScoringLocalVLLM.reset_rerank_responses clears responses."""
        mock = MockScoringLocalVLLM(rerank_responses=[[0.9]])
        mock.reset_rerank_responses()
        assert mock.rerank_responses == []

    def test_pairwise_scoring(self) -> None:
        """Test MockScoringLocalVLLM for pairwise scoring."""
        # Format: layers[scores] - 1 layer with 2 scores
        mock = MockScoringLocalVLLM(rerank_responses=[[0.9, 0.1]])

        queries = ["q1", "q2"]
        documents = ["d1", "d2"]

        responses = mock.batch(queries=queries, documents=documents, broadcast_scores=False)
        assert len(responses) == 2
        assert responses[0].results[0].relevance_score == 0.9
        assert responses[1].results[0].relevance_score == 0.1

    def test_broadcast_scoring(self) -> None:
        """Test MockScoringLocalVLLM for broadcast scoring."""
        # Format: layers[scores] - 1 layer with 2 scores for q1-d1, q1-d2
        mock = MockScoringLocalVLLM(rerank_responses=[[0.9, 0.1]])

        queries = ["q1", "q1"]
        documents = ["d1", "d2"]

        responses = mock.batch(queries=queries, documents=documents, broadcast_scores=True)
        assert len(responses) == 1
        assert len(responses[0].results) == 2

    def test_layered_responses(self) -> None:
        """Test MockScoringLocalVLLM with multiple layers for sequential calls."""
        # Format: 2 layers, each with 1 score
        mock = MockScoringLocalVLLM(rerank_responses=[
            [0.9],  # Layer 0
            [0.5],  # Layer 1
        ])

        response1 = mock.forward(query="Q1", document="D1")
        assert response1.results[0].relevance_score == 0.9

        response2 = mock.forward(query="Q2", document="D2")
        assert response2.results[0].relevance_score == 0.5


# =============================================================================
# Tests for MockPredict
# =============================================================================


class TestMockPredict:
    """Tests for MockPredict functionality."""

    def test_init(self) -> None:
        """Test MockPredict initializes with responses."""
        sig = dspy.Signature("input -> output")
        # Format: layers[requests[completions]]
        responses = [[["output"]]]
        mock = MockPredict(responses=responses, signature=sig)
        assert isinstance(mock.lm, MockGenerativeLocalVLLM)
        assert mock.lm.responses == responses

    def test_forward(self) -> None:
        """Test MockPredict.forward calls through to LocalPredict."""
        sig = dspy.Signature("input -> output")
        # Format: layers[requests[completions]]
        responses = [[["## output\nvalue"]]]
        mock = MockPredict(responses=responses, signature=sig)

        pred = mock.forward(input="test")
        assert len(pred) == 1
        assert pred[0].output == "value"

if __name__ == "__main__":
    pytest.main([__file__])
