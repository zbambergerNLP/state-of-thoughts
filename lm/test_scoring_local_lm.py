"""Comprehensive unit tests for ScoringLocalVLLM class.

We include both unit tests with mocked dependencies and integration tests that require a GPU.

Response format for MockScoringLocalVLLM (strict):
    list[list[float]] = layers[scores]

Expected usage:

```bash
pytest lm/test_scoring_local_lm.py -vv
```
"""

# Standard library imports
import logging
import os
from collections.abc import Generator

# Third-party imports
import pytest
import torch

# Local imports
from constants import Verbosity
from lm.scoring_local_lm import (
    RerankResponse,
    RerankResult,
    ScoringLocalVLLM,
)
from utilities_for_tests import MockScoringLocalVLLM

logger = logging.getLogger(__name__)


# =============================================================================
# Unit Tests (Mocked - No GPU Required)
# =============================================================================


class TestMockScoringLocalVLLMBasics:
    """Test basic MockScoringLocalVLLM functionality."""

    def test_mock_initialization(self) -> None:
        """Test that MockScoringLocalVLLM initializes without errors."""
        # Format: layers[scores] - 1 layer, 1 score
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85]])
        assert mock_lm.model_name == "mock-scoring-model"
        assert mock_lm.model_path == "mock-scoring-model"
        assert mock_lm.history == []

    def test_mock_with_float_response(self) -> None:
        """Test MockScoringLocalVLLM with a simple float score."""
        # Format: layers[scores] - 1 layer, 1 score
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85]])
        response = mock_lm.forward(query="What is AI?", document="AI is artificial intelligence.")

        assert isinstance(response, RerankResponse)
        assert len(response.results) == 1
        assert response.results[0].relevance_score == 0.85
        assert response.results[0].index == 0

    def test_mock_with_list_of_floats(self) -> None:
        """Test MockScoringLocalVLLM with multiple scores for batch calls."""
        # Format: layers[scores] - 1 layer, 3 scores
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.9, 0.7, 0.5]])
        responses = mock_lm.batch(
            queries=["Q1", "Q2", "Q3"],
            documents=["D1", "D2", "D3"],
            broadcast_scores=False,
        )

        assert len(responses) == 3
        assert responses[0].results[0].relevance_score == 0.9
        assert responses[1].results[0].relevance_score == 0.7
        assert responses[2].results[0].relevance_score == 0.5


class TestScoringLocalVLLMForward:
    """Test forward method for single inputs using MockScoringLocalVLLM."""

    @pytest.fixture
    def mock_lm(self) -> MockScoringLocalVLLM:
        """Create a MockScoringLocalVLLM instance.

        Returns:
            A MockScoringLocalVLLM instance with a simple score.
        """
        # Format: layers[scores]
        return MockScoringLocalVLLM(rerank_responses=[[0.85]])

    def test_forward_success(self, mock_lm: MockScoringLocalVLLM) -> None:
        """Test forward method returns RerankResponse.

        Args:
            mock_lm: MockScoringLocalVLLM instance.
        """
        response = mock_lm.forward(query="What is AI?", document="AI is artificial intelligence.")

        assert isinstance(response, RerankResponse)
        assert len(response.results) == 1
        assert response.results[0].relevance_score == 0.85
        assert response.results[0].index == 0

    def test_forward_with_different_scores(self) -> None:
        """Test forward method with different score values."""
        # Format: layers[scores]
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.95]])
        response = mock_lm.forward(query="Query", document="Document")

        assert response.results[0].relevance_score == 0.95


class TestScoringLocalVLLMBatch:
    """Test batch method for multiple inputs using MockScoringLocalVLLM."""

    def test_batch_pairwise_mode(self) -> None:
        """Test batch method in pairwise mode."""
        # Format: layers[scores] - 1 layer, 2 scores
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85, 0.70]])

        responses = mock_lm.batch(
            queries=["Query 1", "Query 2"],
            documents=["Doc 1", "Doc 2"],
            broadcast_scores=False,
        )

        assert isinstance(responses, list)
        assert len(responses) == 2
        for response in responses:
            assert isinstance(response, RerankResponse)
            assert len(response.results) == 1

        assert responses[0].results[0].relevance_score == 0.85
        assert responses[1].results[0].relevance_score == 0.70

    def test_batch_broadcast_mode(self) -> None:
        """Test batch method in broadcast mode."""
        # Format: layers[scores] - 1 layer, 4 scores for 4 pairs
        # Pairs: q1-d1, q1-d2, q2-d1, q2-d2
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85, 0.70, 0.90, 0.60]])

        # Broadcast mode: queries repeat, documents cycle
        responses = mock_lm.batch(
            queries=["Query 1", "Query 1", "Query 2", "Query 2"],
            documents=["Doc 1", "Doc 2", "Doc 1", "Doc 2"],
            broadcast_scores=True,
        )

        assert isinstance(responses, list)
        assert len(responses) == 2  # 2 query groups
        for response in responses:
            assert isinstance(response, RerankResponse)
            assert len(response.results) == 2  # 2 documents per query

        # First query group (Query 1): scores for Doc 1 and Doc 2
        assert responses[0].results[0].relevance_score == 0.85
        assert responses[0].results[1].relevance_score == 0.70
        # Second query group (Query 2): scores for Doc 1 and Doc 2
        assert responses[1].results[0].relevance_score == 0.90
        assert responses[1].results[1].relevance_score == 0.60


class TestScoringLocalVLLMCall:
    """Test __call__ method dispatching using MockScoringLocalVLLM."""

    def test_call_single(self) -> None:
        """Test __call__ with single query/document dispatches to forward."""
        # Format: layers[scores]
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85]])
        response = mock_lm(query="What is AI?", document="AI is artificial intelligence.")

        assert isinstance(response, RerankResponse)
        assert len(response.results) == 1

    def test_call_batch(self) -> None:
        """Test __call__ with batch queries/documents dispatches to batch."""
        # Format: layers[scores] - 1 layer, 2 scores
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85, 0.70]])
        responses = mock_lm(
            queries=["Query 1", "Query 2"],
            documents=["Doc 1", "Doc 2"],
        )

        assert isinstance(responses, list)
        assert len(responses) == 2

    def test_call_missing_inputs_raises(self) -> None:
        """Test __call__ raises when inputs are missing."""
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85]])
        with pytest.raises(ValueError, match="Must provide either"):
            mock_lm()

    def test_call_partial_single_raises(self) -> None:
        """Test __call__ raises when only query is provided."""
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85]])
        with pytest.raises(ValueError, match="both 'query' and 'document'"):
            mock_lm(query="What is AI?")

    def test_call_partial_batch_raises(self) -> None:
        """Test __call__ raises when only queries is provided."""
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85]])
        with pytest.raises(ValueError, match="both 'queries' and 'documents'"):
            mock_lm(queries=["Query 1", "Query 2"])


class TestScoringLocalVLLMRerankResults:
    """Test rerank result building using MockScoringLocalVLLM."""

    def test_rerank_result_structure(self) -> None:
        """Test that RerankResult objects are correctly built via batch."""
        # Format: layers[scores] - 1 layer, 3 scores
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85, 0.70, 0.55]])
        responses = mock_lm.batch(
            queries=["Q1", "Q2", "Q3"],
            documents=["D1", "D2", "D3"],
            broadcast_scores=False,
        )

        assert len(responses) == 3
        for response in responses:
            assert len(response.results) == 1
            result = response.results[0]
            assert isinstance(result, RerankResult)
            assert result.index == 0  # Each response has index 0

        assert responses[0].results[0].relevance_score == 0.85
        assert responses[1].results[0].relevance_score == 0.70
        assert responses[2].results[0].relevance_score == 0.55


class TestScoringLocalVLLMModelName:
    """Test model_name property using MockScoringLocalVLLM."""

    def test_model_name_property(self) -> None:
        """Test model_name property returns expected value."""
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85]])
        assert mock_lm.model_name == "mock-scoring-model"


class TestScoringLocalVLLMLayeredResponses:
    """Test layered responses for sequential calls using MockScoringLocalVLLM."""

    def test_layered_responses(self) -> None:
        """Test that mock returns different responses for sequential calls."""
        # Format: layers[scores] - 2 layers, 1 score each
        mock_lm = MockScoringLocalVLLM(rerank_responses=[
            [0.9],  # Layer 0: 1 score
            [0.5],  # Layer 1: 1 score
        ])

        # First call uses layer 0
        response1 = mock_lm.forward(query="Query 1", document="Doc 1")
        assert response1.results[0].relevance_score == 0.9

        # Second call uses layer 1
        response2 = mock_lm.forward(query="Query 2", document="Doc 2")
        assert response2.results[0].relevance_score == 0.5

    def test_layered_batch_responses(self) -> None:
        """Test layered responses for batch calls."""
        # Format: layers[scores] - 2 layers, 2 scores each
        mock_lm = MockScoringLocalVLLM(rerank_responses=[
            [0.9, 0.8],  # Layer 0: 2 scores for first batch call
            [0.7, 0.6],  # Layer 1: 2 scores for second batch call
        ])

        # First batch call uses layer 0
        responses1 = mock_lm.batch(
            queries=["Q1", "Q2"],
            documents=["D1", "D2"],
            broadcast_scores=False,
        )
        assert responses1[0].results[0].relevance_score == 0.9
        assert responses1[1].results[0].relevance_score == 0.8

        # Second batch call uses layer 1
        responses2 = mock_lm.batch(
            queries=["Q3", "Q4"],
            documents=["D3", "D4"],
            broadcast_scores=False,
        )
        assert responses2[0].results[0].relevance_score == 0.7
        assert responses2[1].results[0].relevance_score == 0.6


class TestScoringLocalVLLMResponseReset:
    """Test response reset functionality using MockScoringLocalVLLM."""

    def test_set_rerank_responses(self) -> None:
        """Test setting new rerank responses."""
        # Format: layers[scores]
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85]])
        response1 = mock_lm.forward(query="Q", document="D")
        assert response1.results[0].relevance_score == 0.85

        # Set new responses
        mock_lm.set_rerank_responses([[0.95]])
        response2 = mock_lm.forward(query="Q", document="D")
        assert response2.results[0].relevance_score == 0.95

    def test_reset_rerank_responses(self) -> None:
        """Test resetting rerank responses to empty."""
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.85]])
        mock_lm.reset_rerank_responses()

        with pytest.raises(AssertionError, match="No score responses"):
            mock_lm.forward(query="Q", document="D")


class TestScoringLocalVLLMRerankResponseObject:
    """Test that raw scores are converted to RerankResponse objects by the class."""

    def test_scores_become_rerank_responses(self) -> None:
        """Test that raw float scores are converted to RerankResponse objects."""
        # Format: layers[scores] - 1 layer, 2 scores
        mock_lm = MockScoringLocalVLLM(rerank_responses=[[0.99, 0.77]])

        # Call batch with 2 pairs to use both scores
        responses = mock_lm.batch(
            queries=["Q1", "Q2"],
            documents=["D1", "D2"],
            broadcast_scores=False,
        )

        # The class builds RerankResponse objects from raw scores
        assert len(responses) == 2
        assert isinstance(responses[0], RerankResponse)
        assert isinstance(responses[1], RerankResponse)
        assert responses[0].results[0].relevance_score == 0.99
        assert responses[1].results[0].relevance_score == 0.77


# =============================================================================
# Integration Tests (GPU Required)
# =============================================================================

# GPU Skip Marker
pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU tests require GPU access",
)


@pytestmark_gpu
class TestScoringLocalVLLMIntegration:
    """Integration tests for ScoringLocalVLLM using real models (requires GPU).

    These tests instantiate actual reranker models and verify that scoring works
    correctly with reasonable inputs. They verify that token counts in Usage objects
    are > 0.
    """

    @pytest.fixture(scope="class")
    def shared_gpu_model(self) -> Generator[ScoringLocalVLLM, None, None]:
        """Shared ScoringLocalVLLM fixture for all GPU integration tests.

        This fixture loads a reranker model once and shares it across all GPU test
        methods to avoid loading multiple models and running out of GPU memory.

        Yields:
            ScoringLocalVLLM: A real ScoringLocalVLLM instance.
        """
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        base_path = "/projects/BSTEWART/model_storage"
        model_name = "Qwen3-Reranker-4B"
        model_path = os.path.join(base_path, model_name)

        lm = None
        try:
            logger.info(f"Initializing shared GPU scoring model from: {model_path}")
            lm = ScoringLocalVLLM(
                model=model_path,
                tensor_parallel_size=1,
                dtype="auto",
                gpu_memory_utilization=0.6,
                max_model_len=4096,
                enforce_eager=True,
                verbosity=Verbosity.INFO,
            )
            logger.info("Shared GPU scoring model initialized successfully")
            yield lm
        finally:
            # Cleanup after all GPU tests complete
            if lm is not None:
                logger.info("Cleaning up shared GPU scoring model...")
                lm.kill()

    @pytest.mark.parametrize(
        # Parameter names
        [
            "test_name",
            "query",
            "document",
        ],
        # Parameter values
        [
            pytest.param(
                "relevant_document",                # test_name
                "What is machine learning?",        # query
                (                                   # document
                    "Machine learning is a subset of artificial intelligence that enables "
                    "systems to learn and improve from experience without being explicitly "
                    "programmed."
                ),
                id="relevant_document_scoring",
            ),
            pytest.param(
                "irrelevant_document",              # test_name
                "What is machine learning?",        # query
                "The weather today is sunny with a high of 75 degrees.",  # document
                id="irrelevant_document_scoring",
            ),
            pytest.param(
                "long_document",                    # test_name
                "Explain neural networks",          # query
                (                                   # document
                    "Neural networks are computing systems inspired by biological neural "
                    "networks that constitute animal brains. A neural network consists of "
                    "interconnected nodes or neurons organized in layers. The input layer "
                    "receives data, hidden layers process the data, and the output layer "
                    "produces the final result. Each connection has a weight that adjusts "
                    "as learning proceeds."
                ),
                id="long_document_scoring",
            ),
        ],
    )
    def test_forward_integration(
        self,
        shared_gpu_model: ScoringLocalVLLM,
        test_name: str,
        query: str,
        document: str,
    ) -> None:
        """Test forward method with real model produces valid scores.

        This test verifies that:
        1. The model produces a valid RerankResponse
        2. The response contains exactly one result
        3. The relevance score is a valid float
        4. The usage statistics have prompt_tokens > 0
        5. The total_tokens > 0

        Args:
            shared_gpu_model: Real ScoringLocalVLLM instance.
            test_name: Name of the test case.
            query: Query string to test.
            document: Document string to test.
        """
        response = shared_gpu_model.forward(query=query, document=document)

        # Verify response structure
        assert isinstance(response, RerankResponse), (
            f"Expected RerankResponse, got {type(response)}"
        )
        assert len(response.results) == 1, f"Expected 1 result, got {len(response.results)}"

        # Verify result structure
        result = response.results[0]
        assert isinstance(result, RerankResult), (
            f"Expected RerankResult, got {type(result)}"
        )
        assert result.index == 0, f"Expected index 0, got {result.index}"
        assert isinstance(result.relevance_score, float), (
            f"Expected float score, got {type(result.relevance_score)}"
        )

        # Verify usage statistics in meta
        assert response.meta is not None, "Expected meta to be present"
        assert response.meta["prompt_tokens"] > 0, (
            f"Expected prompt_tokens > 0, got {response.meta['prompt_tokens']}"
        )
        assert response.meta["total_tokens"] > 0, (
            f"Expected total_tokens > 0, got {response.meta['total_tokens']}"
        )

        logger.info(
            f"Test '{test_name}' passed:\n"
            f"\trelevance_score={result.relevance_score:.4f}\n"
            f"\tprompt_tokens={response.meta['prompt_tokens']}\n"
            f"\ttotal_tokens={response.meta['total_tokens']}"
        )

    def test_batch_pairwise_integration(
        self, shared_gpu_model: ScoringLocalVLLM
    ) -> None:
        """Test batch method in pairwise mode with real model.

        This test verifies that:
        1. The model produces a list of RerankResponse objects
        2. Each response contains valid usage statistics with tokens > 0

        Args:
            shared_gpu_model: Real ScoringLocalVLLM instance.
        """
        queries = [
            "What is Python?",
            "How does gravity work?",
            "What is the capital of France?",
        ]
        documents = [
            "Python is a high-level programming language known for its simplicity.",
            "Gravity is a force that attracts objects with mass toward each other.",
            "Paris is the capital and most populous city of France.",
        ]

        responses = shared_gpu_model.batch(
            queries=queries,
            documents=documents,
            broadcast_scores=False,
        )

        # Verify response structure
        assert isinstance(responses, list), f"Expected list, got {type(responses)}"
        assert len(responses) == len(queries), (
            f"Expected {len(queries)} responses, got {len(responses)}"
        )

        for i, response in enumerate(responses):
            assert isinstance(response, RerankResponse), (
                f"Response {i}: Expected RerankResponse, got {type(response)}"
            )
            assert len(response.results) == 1, (
                f"Response {i}: Expected 1 result, got {len(response.results)}"
            )

            # Verify usage statistics
            assert response.meta is not None, f"Response {i}: Expected meta to be present"
            assert response.meta["prompt_tokens"] > 0, (
                f"Response {i}: Expected prompt_tokens > 0, "
                f"got {response.meta['prompt_tokens']}"
            )
            assert response.meta["total_tokens"] > 0, (
                f"Response {i}: Expected total_tokens > 0, "
                f"got {response.meta['total_tokens']}"
            )

            logger.info(
                f"Response {i}: score={response.results[0].relevance_score:.4f}, "
                f"tokens={response.meta['total_tokens']}"
            )

        logger.info(f"Batch pairwise test passed with {len(responses)} responses")

    def test_batch_broadcast_integration(
        self, shared_gpu_model: ScoringLocalVLLM
    ) -> None:
        """Test batch method in broadcast mode with real model.

        This test verifies that:
        1. Broadcast mode correctly groups scores by query
        2. Each query group contains scores for all documents
        3. Usage statistics are properly aggregated

        Args:
            shared_gpu_model: Real ScoringLocalVLLM instance.
        """
        # 2 queries, 2 documents each = 4 total pairs
        queries = [
            "What is machine learning?",
            "What is machine learning?",
            "What is deep learning?",
            "What is deep learning?",
        ]
        documents = [
            "Machine learning is a subset of AI.",
            "The weather is nice today.",
            "Machine learning is a subset of AI.",
            "The weather is nice today.",
        ]

        responses = shared_gpu_model.batch(
            queries=queries,
            documents=documents,
            broadcast_scores=True,
        )

        # Verify response structure
        assert isinstance(responses, list), f"Expected list, got {type(responses)}"
        assert len(responses) == 2, f"Expected 2 query groups, got {len(responses)}"

        for i, response in enumerate(responses):
            assert isinstance(response, RerankResponse), (
                f"Response {i}: Expected RerankResponse, got {type(response)}"
            )
            assert len(response.results) == 2, (
                f"Response {i}: Expected 2 results (one per document), "
                f"got {len(response.results)}"
            )

            # Verify usage statistics
            assert response.meta is not None, f"Response {i}: Expected meta to be present"
            assert response.meta["prompt_tokens"] > 0, (
                f"Response {i}: Expected prompt_tokens > 0"
            )
            assert response.meta["total_tokens"] > 0, (
                f"Response {i}: Expected total_tokens > 0"
            )

            # Log scores for each document
            for j, result in enumerate(response.results):
                logger.info(
                    f"Query {i}, Doc {j}: score={result.relevance_score:.4f}"
                )

        logger.info(f"Batch broadcast test passed with {len(responses)} query groups")

    def test_score_ordering_integration(
        self, shared_gpu_model: ScoringLocalVLLM
    ) -> None:
        """Test that relevant documents score higher than irrelevant ones.

        This test verifies semantic correctness by checking that a relevant
        document scores higher than an irrelevant one for the same query.

        Args:
            shared_gpu_model: Real ScoringLocalVLLM instance.
        """
        query = "What is the capital of Japan?"
        relevant_doc = "Tokyo is the capital city of Japan, located on Honshu island."
        irrelevant_doc = "Pizza is a popular Italian dish made with dough and toppings."

        # Score both documents
        relevant_response = shared_gpu_model.forward(query=query, document=relevant_doc)
        irrelevant_response = shared_gpu_model.forward(query=query, document=irrelevant_doc)

        relevant_score = relevant_response.results[0].relevance_score
        irrelevant_score = irrelevant_response.results[0].relevance_score

        # Relevant document should score higher
        assert relevant_score > irrelevant_score, (
            f"Expected relevant document to score higher than irrelevant. "
            f"Relevant: {relevant_score:.4f}, Irrelevant: {irrelevant_score:.4f}"
        )

        logger.info(
            f"Score ordering test passed:\n"
            f"\tRelevant doc score: {relevant_score:.4f}\n"
            f"\tIrrelevant doc score: {irrelevant_score:.4f}"
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
