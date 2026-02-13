"""Scoring language model wrapper for vLLM.

This module provides the `ScoringLocalVLLM` class, a DSPy-compatible wrapper for
vLLM-based local language models specifically for scoring/reranking tasks. It returns
strictly `RerankResponse` objects and supports both single and batch processing.
"""

# Standard library imports
import gc
import logging
import uuid
from typing import Any, Literal

# Third-party imports
import dspy
import torch
from pydantic import BaseModel, Field
from vllm import LLM, ScoringRequestOutput, distributed
from vllm.model_executor.layers.quantization import QuantizationMethods

# Local imports
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.lm_constants import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_SEED,
)
from lm.lm_utils import DictAccessMixin, Usage

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RerankResult(BaseModel, DictAccessMixin):
    """Individual result in a rerank response (Cohere format)."""

    index: int = Field(..., description="Index of the document")
    relevance_score: float = Field(..., description="Relevance score for the document")


class RerankResponse(BaseModel, DictAccessMixin):
    """Rerank response matching Cohere format via LiteLLM. Supports dict and object access."""

    id: str = Field(..., description="Unique identifier for the response")
    results: list[RerankResult] = Field(..., description="List of rerank results")
    meta: dict[str, Any] | None = Field(None, description="Metadata about the request")

    def dict(self, **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(**kwargs)


class ScoringLocalVLLM(dspy.BaseLM):
    """DSPy-compatible wrapper for vLLM-based scoring/reranking models.

    This class is specialized for scoring tasks and returns strictly
    `RerankResponse` objects. It supports both single and batch processing
    of query-document pairs.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        tokenizer_mode: Literal["auto", "slow", "mistral", "custom"] = "auto",
        skip_tokenizer_init: bool = False,
        tensor_parallel_size: int = 1,
        dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"] = "auto",
        quantization: QuantizationMethods | None = None,
        seed: int = DEFAULT_SEED,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        hf_token: bool = False,
        enable_prefix_caching: bool = True,
        enforce_eager: bool = False,
        verbosity: Verbosity = Verbosity.WARNING,
        **kwargs: Any,
    ) -> None:
        """Create a new scoring language model instance.

        Args:
            model: The model to use. This should be a string of the form ``"llm_provider/llm_name"``
                offered by HuggingFace. For example, ``"Qwen/Qwen3-Reranker-0.6B"``.
            tokenizer: The tokenizer to use. If None, inferred from the model.
            tokenizer_mode: The tokenizer mode. "auto" uses fast tokenizer if available.
            skip_tokenizer_init: If true, skip initialization of tokenizer and detokenizer.
            tensor_parallel_size: The number of GPUs to use for tensor parallelism.
            dtype: The data type for model weights and activations.
            quantization: The method used to quantize model weights (e.g., "awq", "gptq", "fp8").
            seed: Random seed for reproducibility.
            gpu_memory_utilization: Ratio of GPU memory to reserve (0-1).
            swap_space: CPU memory (GiB) per GPU for swap space.
            cpu_offload_gb: CPU memory (GiB) for offloading model weights.
            max_model_len: Maximum sequence length covered by CUDA graphs.
            hf_token: HuggingFace token for remote files.
            enable_prefix_caching: Enable automatic prefix caching.
            enforce_eager: Enable eager execution mode in vLLM.
            verbosity: Verbosity level for logging. One of Verbosity.DEBUG, Verbosity.INFO,
                Verbosity.WARNING, or Verbosity.ERROR. Defaults to Verbosity.WARNING.
            **kwargs: Additional arguments passed to the vLLM constructor.
        """
        self._verbosity = verbosity
        logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])
        logger.info(
            f"model: {model}\n"
            f"tokenizer: {tokenizer}\n"
            f"tokenizer_mode: {tokenizer_mode}\n"
            f"skip_tokenizer_init: {skip_tokenizer_init}\n"
            f"tensor_parallel_size: {tensor_parallel_size}\n"
            f"dtype: {dtype}\n"
            f"quantization: {quantization}\n"
            f"seed: {seed}\n"
            f"gpu_memory_utilization: {gpu_memory_utilization}\n"
            f"max_model_len: {max_model_len}\n"
            f"enable_prefix_caching: {enable_prefix_caching}\n"
            f"enforce_eager: {enforce_eager}\n"
            f"kwargs: {kwargs}\n"
        )

        # Add hf_overrides for Qwen3 reranker configuration
        if "qwen" in model.lower():
            kwargs["hf_overrides"] = {
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            }

        self.model = LLM(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            max_model_len=max_model_len,
            hf_token=hf_token,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            **kwargs,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.history: list[dict[str, Any]] = []
        self.model_path = model
        self._model_name = model.split("/")[-1] if "/" in model else model
        self.task = "score"

        # Add default kwargs attribute for fallback
        self.kwargs: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        """Get the model name (last component of model path)."""
        return self._model_name

    @property
    def verbosity(self) -> Verbosity:
        """Get the current verbosity level."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity: Verbosity) -> None:
        """Set the verbosity level and update logger.

        Args:
            verbosity: The new verbosity level to set.
        """
        self._verbosity = verbosity
        logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

    def _validate_single_inputs(
        self,
        query: str | None = None,
        document: str | None = None,
    ) -> None:
        """Validate inputs for single query-document pair.

        Args:
            query: A single query string.
            document: A single document string.

        Raises:
            ValueError: If query or document is missing.
        """
        if query is None:
            raise ValueError("Must provide a query for scoring tasks.")
        if document is None:
            raise ValueError("Must provide a document for scoring tasks.")

    def _validate_batch_inputs(
        self,
        queries: list[str] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Validate inputs for batch processing.

        Args:
            queries: A batch of query strings.
            documents: A batch of document strings.

        Raises:
            ValueError: If queries or documents are missing.
            AssertionError: If queries and documents have different lengths.
        """
        if queries is None or len(queries) == 0:
            raise ValueError("Must provide non-empty queries for batch scoring.")
        if documents is None or len(documents) == 0:
            raise ValueError("Must provide non-empty documents for batch scoring.")
        assert len(queries) == len(documents), (
            f"Queries and documents must have the same length for pairwise scoring. "
            f"Got {len(queries)} queries and {len(documents)} documents."
        )

    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens in text."""
        return len(self.tokenizer(text))  # type: ignore

    def _build_usage(
        self,
        queries: list[str],
        documents: list[str],
    ) -> list[Usage]:
        """Construct token usage statistics for scoring tasks.

        Args:
            queries: List of query strings.
            documents: List of document strings.

        Returns:
            A list of Usage objects, one per query-document pair.
        """
        results = []
        for query, document in zip(queries, documents, strict=True):
            prompt_tokens = self._count_tokens(query)
            document_tokens = self._count_tokens(document)
            results.append(Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens + document_tokens,
            ))
        return results

    def _build_reranker_results(self, scores: list[list[float]]) -> list[list[RerankResult]]:
        """Construct RerankResult objects from reranker scores.

        Args:
            scores: List of lists of relevance scores.

        Returns:
            List of lists of RerankResult objects.
        """
        return [
            [RerankResult(index=i, relevance_score=score) for i, score in enumerate(example)]
            for example in scores
        ]

    def _build_rerank_responses(
        self,
        rerank_results: list[list[RerankResult]],
        usage: list[Usage],
    ) -> list[RerankResponse]:
        """Build RerankResponse objects from rerank results and usage statistics.

        Args:
            rerank_results: List of lists of RerankResult objects.
            usage: List of Usage objects.

        Returns:
            List of RerankResponse objects (Cohere format), one per query.
        """
        result = []
        for example_results, usage_obj in zip(rerank_results, usage, strict=True):
            result.append(RerankResponse(
                id=f"rerank_{uuid.uuid4()}",
                results=example_results,
                meta={
                    "prompt_tokens": usage_obj.prompt_tokens,
                    "completion_tokens": usage_obj.completion_tokens,
                    "total_tokens": usage_obj.total_tokens,
                },
            ))
        return result

    def forward(
        self,
        query: str,
        document: str,
        use_tqdm: bool = True,
        **kwargs: Any,
    ) -> RerankResponse:
        """Score a single query-document pair.

        Args:
            query: A single query string.
            document: A single document string.
            use_tqdm: Whether to display a progress bar.
            **kwargs: Additional keyword arguments.

        Returns:
            RerankResponse: A single rerank response containing the score.
        """
        # Validate inputs
        self._validate_single_inputs(query=query, document=document)

        # Call vLLM's score API
        outputs: list[ScoringRequestOutput] = self.model.score(
            [query], [document], use_tqdm=use_tqdm
        )

        # Extract score
        score = outputs[0].outputs.score

        # Build response
        rerank_results = self._build_reranker_results([[score]])
        usage = self._build_usage([query], [document])

        # Track usage if enabled
        if dspy.settings.usage_tracker:
            dspy.settings.usage_tracker.add_usage(self.model, usage)

        responses = self._build_rerank_responses(rerank_results, usage)
        return responses[0]

    def batch(
        self,
        queries: list[str],
        documents: list[str],
        use_tqdm: bool = True,
        broadcast_scores: bool = False,
        **kwargs: Any,
    ) -> list[RerankResponse]:
        """Score a batch of query-document pairs.

        This method performs pairwise scoring: query[i] is paired with document[i].
        It supports two output formats controlled by the broadcast_scores parameter:

        1. Broadcast mode (broadcast_scores=True): Performs cartesian product grouping.
           Assumes queries are arranged as [q_1 x m, q_2 x m, ..., q_N x m].
           Groups consecutive m scores together, returning N RerankResponse objects.

        2. Pairwise mode (broadcast_scores=False): Returns N RerankResponse objects,
           each with 1 score.

        Args:
            queries: List of query strings.
            documents: List of document strings.
            use_tqdm: Whether to display a progress bar.
            broadcast_scores: If True, performs cartesian product grouping.
            **kwargs: Additional keyword arguments.

        Returns:
            list[RerankResponse]: List of rerank responses.
        """
        # Validate inputs
        self._validate_batch_inputs(queries=queries, documents=documents)

        # Call vLLM's score API
        outputs: list[ScoringRequestOutput] = self.model.score(
            queries, documents, use_tqdm=use_tqdm
        )

        # Extract scores from outputs
        all_scores = [output.outputs.score for output in outputs]

        # Structure scores based on broadcast_scores parameter
        if broadcast_scores:
            # Broadcast mode: validate preconditions and get cycle length m
            m = self._validate_broadcast_mode(queries, documents)
            # Group scores into chunks of m (one chunk per query)
            scores = [all_scores[i: i + m] for i in range(0, len(all_scores), m)]
            # Build usage for broadcast mode
            usage = self._build_broadcast_usage(queries, documents, m)
        else:
            # Pairwise mode: one score per query-document pair
            scores = [[score] for score in all_scores]
            usage = self._build_usage(queries, documents)

        # Build rerank results from scores
        rerank_results = self._build_reranker_results(scores)

        # Track usage if enabled
        if dspy.settings.usage_tracker:
            dspy.settings.usage_tracker.add_usage(self.model, usage)

        return self._build_rerank_responses(rerank_results, usage)

    def _validate_broadcast_mode(
        self,
        queries: list[str],
        documents: list[str],
    ) -> int:
        """Validate broadcast mode preconditions and return cycle length m.

        In broadcast mode, documents repeat every m items: [d_1, ..., d_m, d_1, ..., d_m, ...]
        Queries must be arranged as [q_1×m, q_2×m, ..., q_N×m] where each q_i repeats m times.

        Args:
            queries: List of query strings arranged in cartesian product format.
            documents: List of document strings that form repeating cycles.

        Returns:
            Cycle length m (number of documents per query).

        Raises:
            AssertionError: If preconditions are violated.
        """
        assert len(documents) > 0, "Documents list cannot be empty in broadcast mode"

        # Find cycle length m by locating when the first document repeats
        if len(documents) == 1:
            m = 1
        else:
            try:
                m = documents.index(documents[0], 1)
            except ValueError:
                # All documents are unique - single cycle with m = len(documents)
                m = len(documents)

        # Verify documents form complete cycles
        assert len(documents) % m == 0, (
            f"Documents in broadcast mode must form complete cycles. "
            f"Found cycle length m={m}, but total length {len(documents)} is not divisible by {m}."
        )

        # Verify queries are arranged correctly
        num_cycles = len(documents) // m
        assert len(queries) == num_cycles * m, (
            f"Queries in broadcast mode must be arranged as [q_1×m, q_2×m, ..., q_N×m]. "
            f"Expected {num_cycles * m} queries (N={num_cycles} cycles × m={m}), "
            f"but got {len(queries)} queries."
        )

        # Verify queries repeat correctly within each cycle
        for cycle_idx in range(num_cycles):
            cycle_start = cycle_idx * m
            cycle_queries = queries[cycle_start: cycle_start + m]
            assert all(q == cycle_queries[0] for q in cycle_queries), (
                f"Queries in broadcast mode must repeat within each cycle. "
                f"Cycle {cycle_idx} (indices {cycle_start} to {cycle_start + m - 1}) "
                f"contains different queries: {cycle_queries}"
            )

        return m

    def _build_broadcast_usage(
        self,
        queries: list[str],
        documents: list[str],
        m: int,
    ) -> list[Usage]:
        """Build usage statistics for broadcast mode.

        Args:
            queries: List of query strings.
            documents: List of document strings.
            m: Cycle length (number of documents per query).

        Returns:
            List of Usage objects, one per query group.
        """
        usage = []
        num_query_groups = len(queries) // m
        for i in range(num_query_groups):
            query_idx = i * m
            query = queries[query_idx]
            documents_for_group = documents[:m]
            prompt_tokens = self._count_tokens(query)
            total_document_tokens = sum(self._count_tokens(doc) for doc in documents_for_group)
            usage.append(Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens + total_document_tokens,
            ))
        return usage

    def score(
        self,
        queries: list[str],
        documents: list[str],
        use_tqdm: bool = True,
        broadcast_scores: bool = False,
        **kwargs: Any,
    ) -> list[RerankResponse]:
        """Score query-document pairs using the reranker model.

        This is an alias for `batch()` that provides a consistent interface
        for scoring operations. The adapter layer uses this method.

        Args:
            queries: List of query strings.
            documents: List of document strings.
            use_tqdm: Whether to display a progress bar.
            broadcast_scores: If True, performs cartesian product grouping.
            **kwargs: Additional keyword arguments.

        Returns:
            list[RerankResponse]: List of rerank responses.
        """
        return self.batch(
            queries=queries,
            documents=documents,
            use_tqdm=use_tqdm,
            broadcast_scores=broadcast_scores,
            **kwargs,
        )

    def __call__(
        self,
        query: str | None = None,
        document: str | None = None,
        queries: list[str] | None = None,
        documents: list[str] | None = None,
        **kwargs: Any,
    ) -> RerankResponse | list[RerankResponse]:
        """Dispatch to forward or batch based on input structure.

        Args:
            query: A single query string (for single calls).
            document: A single document string (for single calls).
            queries: A batch of query strings (for batch calls).
            documents: A batch of document strings (for batch calls).
            **kwargs: Additional keyword arguments passed to forward/batch.

        Returns:
            For single calls: RerankResponse.
            For batch calls: list[RerankResponse].
        """
        # Check if batch inputs are provided
        if queries is not None or documents is not None:
            if queries is None or documents is None:
                raise ValueError(
                    "For batch scoring, both 'queries' and 'documents' must be provided."
                )
            return self.batch(queries=queries, documents=documents, **kwargs)

        # Check if single inputs are provided
        if query is not None or document is not None:
            if query is None or document is None:
                raise ValueError(
                    "For single scoring, both 'query' and 'document' must be provided."
                )
            return self.forward(query=query, document=document, **kwargs)

        raise ValueError(
            "Must provide either (query, document) for single scoring or "
            "(queries, documents) for batch scoring."
        )

    def update_history(self, entry: dict[str, Any]) -> None:
        """Update the history with a new entry (for compatibility).

        Args:
            entry: The history entry to add.
        """
        self.history.append(entry)

    def sleep(self, sleep_level: int) -> None:
        """Put the engine to sleep.

        Args:
            sleep_level: The sleep level (1 or 2).
        """
        self.model.sleep(level=sleep_level)

    def wake_up(self, tags: list[str] | None) -> None:
        """Wake up the engine from sleep mode.

        Args:
            tags: Optional list of tags to reallocate memory for.
        """
        self.model.wake_up(tags=tags)

    def kill(self) -> None:
        """Free all model resources and memory."""
        try:
            distributed.parallel_state.destroy_model_parallel()
        except Exception:
            pass
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

