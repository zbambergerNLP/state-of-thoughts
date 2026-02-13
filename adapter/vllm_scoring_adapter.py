"""
DSPy adapter for scoring/reranking tasks using local vLLM reranker models.
"""

# Standard library imports
import logging
from typing import Any, Literal

# Third-party imports
from dspy.utils.callback import BaseCallback

# Local imports
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.scoring_local_lm import RerankResponse, ScoringLocalVLLM

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)



class LocalVLLMScoringAdapter:
	"""
	Adapter for scoring/reranking tasks using LocalVLLM.

	This adapter is specifically configured for scoring tasks where the
	underlying model is initialized with task="score" for efficient
	query-document pair scoring. It provides methods for formatting inputs,
	executing batch scoring operations, and parsing results.
	"""

	def __init__(
		self,
		callbacks: list[BaseCallback] | None = None,
		message_start_token: str = "<|im_start|>",  # noqa: S107
		message_end_token: str = "<|im_end|>",      # noqa: S107
		assistant_prefix: str = "<think>\n\n</think>\n\n",
		verbosity: Verbosity = Verbosity.INFO,
	) -> None:
		"""
		Initialize the LocalVLLMScoringAdapter.

		Args:
			callbacks: Optional list of callbacks to execute during scoring
				operations. Callbacks should implement BaseCallback interface
				from DSPy.
			message_start_token: Token used to start a message block (e.g., "<|im_start|>").
				Defaults to Qwen3 format.
			message_end_token: Token used to end a message block (e.g., "<|im_end|>").
				Defaults to Qwen3 format.
			assistant_prefix: Prefix text for assistant messages in the suffix.
				Defaults to redacted reasoning format.
			verbosity: Verbosity level for logging. Defaults to INFO.
		"""
		self.callbacks: list[BaseCallback] = callbacks or []
		self._message_start_token: str = message_start_token
		self._message_end_token: str = message_end_token
		self._assistant_prefix: str = assistant_prefix
		self._verbosity: Verbosity = verbosity

	@property
	def assistant_prefix(self) -> str:
		return self._assistant_prefix

	@assistant_prefix.setter
	def assistant_prefix(self, value: str) -> None:
		self._assistant_prefix = value

	@property
	def message_start_token(self) -> str:
		return self._message_start_token

	@message_start_token.setter
	def message_start_token(self, value: str) -> None:
		self._message_start_token = value

	@property
	def message_end_token(self) -> str:
		return self._message_end_token

	@message_end_token.setter
	def message_end_token(self, value: str) -> None:
		self._message_end_token = value

	@property
	def verbosity(self) -> Verbosity:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Verbosity) -> None:
		"""Set the verbosity level and update logger."""
		self._verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

	def __call__(
		self,
		instructions: str | list[str],
		lm: ScoringLocalVLLM,
		inputs: dict[str, Any],
		scoring_target: Literal["reasoning", "output", "action"],
		reasoning_candidates: str | list[str] | None = None,
		output_candidates: dict[str, Any] | list[dict[str, Any]] | None = None,
		action_candidates: str | list[str] | None = None,
		include_reasoning_in_document_for_output: bool = False,
	) -> list[RerankResponse]:
		"""
		Score query-document pairs using the LocalVLLM model.

		This method normalizes inputs to lists and calls the underlying
		LocalVLLM's score method. A single query string or list of queries is
		scored against a single document string or list of documents.

		Args:
			instructions: Fully-formed instruction string(s) to place in the reranker prompt.
				- If a single string, it is broadcast to all queries in the call.
				- If a list of strings, it must be aligned 1:1 with the queries produced by this call.
			lm: The LocalVLLM instance with task="score" to use for scoring.
				Must be initialized with "score" constant.
			inputs: The inputs for the reasoning task (e.g., a topic and a stance for an argument
                generation task, or a question for a question-answering task). The inputs
                provided must be specified as input fields in the provided signature.
			scoring_target: Literal specifying what the scoring task is:
				- "reasoning": Evaluate reasoning trajectories or partial solutions
				- "output": Evaluate final output fields (complete solutions)
				- "action": Evaluate controller actions
			reasoning_candidates: Either a single reasoning candidate or a list of reasoning
                candidates. Will be normalized to list[Reasoning] for processing.
			output_candidates: Either a single output candidate or a list of output candidates.
				Will be normalized to list[Output] for processing.
			action_candidates: Either a single action candidate or list of action candidates.
				Will be normalized to list[str] for processing.
			include_reasoning_in_document_for_output: Whether to include the reasoning trajectory in the
				Document (preceding the output) for OUTPUT scoring. Defaults to False.

		Returns:
			A list of RerankResponse objects, one per query. Each
			RerankResponse contains scoring results for that query against all
			documents, including relevance scores and metadata.

		Raises:
			AssertionError: If scoring_target is invalid, or if the
				LocalVLLM instance is not configured for scoring.
		"""
		# Set logger level based on verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[self.verbosity])

		# Validate target field name
		assert scoring_target in {"reasoning", "output", "action"}, (
		    "scoring_target must be one of 'reasoning', 'output', or 'action', "
		    f"but received '{scoring_target}'."
		)

		# Normalize inputs to always be lists
		norm_reasoning_candidates = (
			[reasoning_candidates] if isinstance(reasoning_candidates, str)
			else reasoning_candidates
		)
		norm_output_candidates = (
			[output_candidates]
			if output_candidates is not None
			and isinstance(output_candidates, dict)
			and not isinstance(output_candidates, list)
			else output_candidates
		)
		norm_action_candidates = (
			[action_candidates] if isinstance(action_candidates, str)
			else action_candidates
		)

		# Validate that the LM is configured for scoring
		assert lm.task == "score", (
			f"LocalVLLM instance must be initialized with task='score', but got task='{lm.task}'."
		)

		# Call the scoring method
		return self._score_batch(
			instructions=instructions,
			lm=lm,
			inputs=inputs,
			scoring_target=scoring_target,
			reasoning_candidates=norm_reasoning_candidates,
			output_candidates=norm_output_candidates,
			action_candidates=norm_action_candidates,
			include_reasoning_in_document_for_output=include_reasoning_in_document_for_output,
		)

	@staticmethod
	def _validate_scoring_inputs(
		scoring_target: Literal["reasoning", "output", "action"],
		reasoning_candidates: list[str] | None,
		output_candidates: list[dict[str, Any]] | None,
		action_candidates: list[str] | None,
		include_reasoning_in_document_for_output: bool,
	) -> None:
		"""Validate required candidate inputs and their shapes for scoring.

		Args:
			scoring_target: What the scoring task is:
				- "reasoning": Evaluate reasoning trajectories or partial solutions
				- "output": Evaluate final output fields (complete solutions)
				- "action": Evaluate controller actions
			reasoning_candidates: Optional list of reasoning candidates to score.
			output_candidates: Optional list of output candidates to score.
			action_candidates: Optional list of action candidates to score.
			include_reasoning_in_document_for_output: Whether to include the reasoning trajectory in the
				Document (preceding the output) for OUTPUT scoring. Defaults to False.

		Raises:
			AssertionError: If the required candidates are not provided for the scoring target.
			ValueError: If the number of reasoning candidates does not match the number of output candidates
				for OUTPUT scoring with reasoning included.
		"""
		if scoring_target == "reasoning":
			assert reasoning_candidates is not None, (
				"reasoning_candidates must be provided when scoring reasoning candidates."
			)
		elif scoring_target == "output":
			assert output_candidates is not None, (
				"output_candidates must be provided when scoring output candidates."
			)
			if include_reasoning_in_document_for_output and reasoning_candidates is not None:
				assert len(reasoning_candidates) == len(output_candidates), (
					"For output scoring with reasoning included, len(reasoning_candidates) must match "
					"len(output_candidates)."
				)
		else:  # scoring_target == "action"
			assert action_candidates is not None, (
				"action_candidates must be provided when scoring action candidates."
			)

	def _score_batch(
		self,
		instructions: str | list[str],
		lm: ScoringLocalVLLM,
		inputs: dict[str, Any],
		scoring_target: Literal["reasoning", "output", "action"],
		reasoning_candidates: list[str] | None = None,
		output_candidates: list[dict[str, Any]] | None = None,
		action_candidates: list[str] | None = None,
		include_reasoning_in_document_for_output: bool = False,
	) -> list[RerankResponse]:
		"""
		Score queries against documents using the ScoringLocalVLLM instance.

		Formats queries and documents, then calls the underlying ScoringLocalVLLM's score method,
		returning one RerankResponse per query.

		Args:
			instructions: Fully-formed instruction string to place in the reranker prompt.
			lm: The LocalVLLM instance with task="score" initialized for
				scoring operations.
			inputs: The inputs for the reasoning task (e.g., a topic and a stance
				for an argument generation task, or a question for a question-
				answering task). The inputs provided must be specified as input
				fields in the provided signature.
			scoring_target: What the scoring task is:
				- "reasoning": Evaluate reasoning trajectories or partial solutions
				- "output": Evaluate final output fields (complete solutions)
				- "action": Evaluate controller actions
			reasoning_candidates: Optional list of reasoning candidates to score.
			output_candidates: Optional list of output candidates to score.
			action_candidates: Optional list of action candidates to score.

		Returns:
			A list of RerankResponse objects, one per query.
			Each RerankResponse contains the scoring results for that query against all documents.
		"""
		self._validate_scoring_inputs(
			scoring_target=scoring_target,
			reasoning_candidates=reasoning_candidates,
			output_candidates=output_candidates,
			action_candidates=action_candidates,
			include_reasoning_in_document_for_output=include_reasoning_in_document_for_output,
		)

		# Determine number of queries based on scoring target
		if scoring_target == "output":
			assert output_candidates is not None, (
				"output_candidates must be provided when scoring output candidates."
			)
			num_queries = len(output_candidates)
		elif scoring_target == "action":
			# In the first step, if there are no reasoning candidates, then we are in the root
			# node, where there is only one state (which includes strictly the input fields).
			num_queries = len(reasoning_candidates) if reasoning_candidates else 1
		else:  # scoring_target == "reasoning"
			assert reasoning_candidates is not None, (
				"reasoning_candidates must be provided when scoring reasoning candidates."
			)
			num_queries = len(reasoning_candidates)

		# Format queries and documents.
		formatted_queries = self.format_queries(
			instructions=instructions,
			inputs=inputs,
			scoring_target=scoring_target,
			num_queries=num_queries,
			reasoning_candidates=reasoning_candidates,
		)
		formatted_documents = self.format_documents(
			scoring_target=scoring_target,
			reasoning_candidates=reasoning_candidates,
			output_candidates=output_candidates,
			action_candidates=action_candidates,
			include_reasoning_in_document_for_output=include_reasoning_in_document_for_output,
		)

		# TODO[P3]: Expand verbosity to include logging at the level of prompts in/out of the model. Logging of the inputs and outputs should be done only when the module that uses the adapter is initialized, and should run with mock inputs. Otheriwise, running this over every input will be too exhaustive.

		# Validate query-document matching based on scoring target
		num_queries = len(formatted_queries)
		num_documents = len(formatted_documents)

		# Score each query against all documents using broadcasting mode.
		# This mode gives us N responses (one per query), each with M scores (one per
		# document/action)
		if scoring_target == "action":
			# For "action": Expand N queries × M documents into N×M pairs where each query
			# is repeated M times (once per document). Then use broadcast_scores=True.
			# The score method groups scores by unique query, giving us N responses with M scores each.
			expanded_queries = []
			expanded_documents = []
			for query in formatted_queries:
				# Repeat this query M times (once for each document)
				for document in formatted_documents:
					expanded_queries.append(query)
					expanded_documents.append(document)

			# Score with broadcast_scores=True
			# The score method automatically groups scores by unique query
			return lm.score(
				queries=expanded_queries,
				documents=expanded_documents,
				use_tqdm=False,
				broadcast_scores=True,
			)

		# For "reasoning" and "output": Each query must have a corresponding document (1-to-1 matching)
		assert num_queries == num_documents, (
			f"For {scoring_target} scoring, the number of queries ({num_queries}) "
			f"must match the number of documents ({num_documents}). "
			f"Each query should be scored against its corresponding document."
		)

		# For "reasoning"/"output": Use pairwise mode (one response per query-document pair)
		rerank_responses: list[RerankResponse] = lm.score(
			queries=formatted_queries,
			documents=formatted_documents,
			use_tqdm=False,
			broadcast_scores=False,
		)

		return rerank_responses

	def _get_user_message_query(
		self,
		inputs: dict[str, Any],
		scoring_target: Literal["reasoning", "output", "action"],
		reasoning_candidates: list[str] | None = None,
		candidate_index: int = 0,
	) -> str:
		"""
		Generate the query content for the user message based on the scoring target.

		* For "reasoning" target: Includes input fields only (no reasoning).
		* For "output" target: Includes input fields only (no reasoning).
		* For "action" target: Includes input fields + the entire reasoning trajectory.

		Args:
			input: The input dictionary containing task-specific input fields.
			scoring_target: What the scoring task is:
				- "reasoning": Evaluate reasoning trajectories or partial solutions
				- "output": Evaluate final output fields (complete solutions)
				- "action": Evaluate controller actions
			reasoning_candidates: Optional list of reasoning candidates.
			candidate_index: Index of the candidate to include.

		Returns:
			A formatted query string with appropriate sections.
		"""
		# Format input fields matching the document format (field: value)
		input_lines = [f"{k}: {v}" for k, v in inputs.items()]
		input_str = "\n".join(input_lines)

		message_parts = [f"# Inputs\n{input_str}"]
		if scoring_target == "action":
			# For "action", include reasoning if present.
			if reasoning_candidates is not None:
				reasoning_str = reasoning_candidates[candidate_index].strip()
				if reasoning_str:
					message_parts.append(f"# Reasoning\n{reasoning_str}")

		query_content = "\n\n".join(message_parts)
		return query_content

	def _get_user_message_document(
		self,
		scoring_target: Literal["reasoning", "output", "action"],
		reasoning_candidates: list[str] | None = None,
		output_candidates: list[dict[str, Any]] | None = None,
		action_candidates: list[str] | None = None,
		candidate_index: int = 0,
		include_reasoning_in_document_for_output: bool = False,
	) -> str:
		"""
		Generate the document content for scoring.

		For "reasoning" target: Returns the reasoning candidate.
		For "output" target: Returns the output candidate.
		For "action" target: Returns the chosen action.

		Args:
			scoring_target: What the scoring task is:
				- "reasoning": Evaluate reasoning trajectories or partial solutions
				- "output": Evaluate final output fields (complete solutions)
				- "action": Evaluate controller actions
			reasoning_candidates: Optional list of reasoning candidates.
			output_candidates: Optional list of output candidates.
			action_candidates: Optional list of action candidate strings.
			candidate_index: Index of the candidate to include.

		Returns:
			A formatted document string with the candidate content.
		"""
		if scoring_target == "reasoning":
			return f"# Reasoning\n{reasoning_candidates[candidate_index].strip()}"
		if scoring_target == "output":
			output_str = self._format_output_dict(output_candidates[candidate_index]).strip()
			if include_reasoning_in_document_for_output and reasoning_candidates is not None:
				reasoning_str = reasoning_candidates[candidate_index].strip()
				if reasoning_str:
					return f"# Reasoning\n{reasoning_str}\n\n# Output\n{output_str}"
			return f"# Output\n{output_str}"
		# scoring_target == "action"
		return f"# Action\n{action_candidates[candidate_index].strip()}"

	def format_queries(
		self,
		instructions: str | list[str],
		scoring_target: Literal["reasoning", "output", "action"],
		inputs: dict[str, Any],
		num_queries: int,
		reasoning_candidates: list[str] | None = None,
	) -> list[str]:
		"""
		Format queries from inputs and optional reasoning trajectories.

		Queries follow the template: {prefix}<Instruct>: {instruction}\n<Query>: {query}\n

		See guidance on vLLM reranker models (in this case qwen3) here:
		https://docs.vllm.ai/en/v0.9.2/examples/offline_inference/qwen3_reranker.html

		Query content varies by scoring target:
		- reasoning: Input fields only
		- output: Input fields only
		- action: Input fields + entire reasoning trajectory (if provided)

		Args:
			instructions: Fully-formed instruction string to place in the reranker prompt.
			scoring_target: What the scoring task is:
				- "reasoning": Evaluate reasoning trajectories or partial solutions
				- "output": Evaluate final output fields (complete solutions)
				- "action": Evaluate controller actions
			input: The input dictionary containing task-specific input fields.
			num_queries: Number of queries to format.
			reasoning_candidates: List of reasoning candidates.
				When provided, trajectories may be extracted for context in the query.

		Returns:
			A list of formatted query strings meant for the `score` method of reranker models.
		"""
		# Fixed prefix for the system message and user start
		prefix = (
			f"{self.message_start_token}system\n"
			"Judge whether the Document meets the requirements based on the Query and the "
			f"Instruct provided. Note that the answer can only be \"yes\" or \"no\".{self.message_end_token}\n"
			f"{self.message_start_token}user\n"
		)

		instructions_list = (
			[instructions.strip() for _ in range(num_queries)]
			if isinstance(instructions, str) else [s.strip() for s in instructions]
		)

		if len(instructions_list) != num_queries:
			raise ValueError(
				f"The number of instructions must match the number of queries ({num_queries}) "
				f"but got {len(instructions_list)} instructions."
			)

		formatted_queries = []
		for i in range(num_queries):
			query_content = self._get_user_message_query(
				inputs=inputs,
				scoring_target=scoring_target,
				reasoning_candidates=reasoning_candidates,
				candidate_index=i,
			)
			# Format: {prefix}<Instruct>: {instruction}\n<Query>: {query}\n
			formatted_query = (
				f"{prefix}<Instruct>: {instructions_list[i].strip()}\n<Query>: {query_content}\n"
			)
			formatted_queries.append(formatted_query)

		return formatted_queries

	def format_documents(
		self,
		scoring_target: Literal["reasoning", "output", "action"],
		reasoning_candidates: list[str] | None = None,
		output_candidates: list[dict[str, Any]] | None = None,
		action_candidates: list[str] | None = None,
		include_reasoning_in_document_for_output: bool = False,
	) -> list[str]:
		"""
		Format documents/candidates according to the signature.

		Determines which type of solution is being scored based on whether an
		output field is provided (complete solution) or not (working solution):
		- **Complete Solution**: output_candidates is not None. Documents include
			both the reasoning trajectory and the final output fields.
		- **Working Solution**: output_candidates is None. Documents include only
			the reasoning trajectories or actions.

		Args:
			signature: The DSPy signature for the scoring task. Used to determine
				which candidates to format and how to format them.
			scoring_target: What the scoring task is:
				- "reasoning": Evaluate reasoning trajectories or partial solutions
				- "output": Evaluate final output fields (complete solutions)
				- "action": Evaluate controller actions
			reasoning_candidates: Optional list of reasoning candidates (working
				solutions without final outputs).
			output_candidates: Optional list of output candidates (complete
				solutions with final outputs). When provided, indicates a
				complete solution scenario.
			action_candidates: Optional list of action candidates (for controller
				action scoring).

		Returns:
			A list of formatted document strings with suffix formatting applied.
		"""
		# Suffix for the document: end of user turn, start of assistant turn + think block
		suffix = (
			f"{self.message_end_token}\n"
			f"{self.message_start_token}assistant\n"
			f"{self.assistant_prefix}"
		)

		# Determine number of documents based on candidates
		if scoring_target == "reasoning":
			if output_candidates is not None:
				num_docs = len(output_candidates)
			elif reasoning_candidates is not None:
				num_docs = len(reasoning_candidates)
			else:
				num_docs = 0
		elif scoring_target == "output":
			num_docs = len(output_candidates) if output_candidates else 0
		else:  # scoring_target == "action"
			num_docs = len(action_candidates) if action_candidates else 0

		formatted_documents = []
		for i in range(num_docs):
			doc_content = self._get_user_message_document(
				scoring_target=scoring_target,
				reasoning_candidates=reasoning_candidates,
				output_candidates=output_candidates,
				action_candidates=action_candidates,
				candidate_index=i,
				include_reasoning_in_document_for_output=include_reasoning_in_document_for_output,
			)

			# Format: <Document>: {doc}{suffix}
			formatted_doc = f"<Document>: {doc_content}{suffix}"
			formatted_documents.append(formatted_doc)

		return formatted_documents

	def _format_output_dict(self, output: dict[str, Any]) -> str:
		"""
		Format an output dictionary as a string for document scoring.

		Args:
			output: Dictionary mapping output field names to their values.

		Returns:
			A formatted string representation of the output.
		"""
		lines = []
		for field_name, field_value in output.items():
			lines.append(f"{field_name}: {field_value}")
		return "\n".join(lines).strip()

