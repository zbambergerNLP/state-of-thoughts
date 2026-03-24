"""Provider-agnostic async LLM judge abstraction.

Supports OpenAI, Google (Gemini), and Anthropic as judge providers for
pairwise argument evaluation.

Usage:
	from experiments.argument_generation.llm_judge import create_judge

	judge = create_judge("openai", "gpt-5-mini-2025-08-07")
	response = await judge.complete("Which argument is better? ...")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class AsyncLLMJudge(ABC):
	"""Abstract base class for async LLM judge providers."""

	@abstractmethod
	async def complete(self, prompt: str) -> str:
		"""Send a prompt and return the model's text response.

		Args:
			prompt: The full prompt to send to the model.

		Returns:
			The model's text response.
		"""


class OpenAIJudge(AsyncLLMJudge):
	"""OpenAI judge using AsyncOpenAI client."""

	def __init__(self, model: str, max_completion_tokens: int = 1000) -> None:
		from openai import AsyncOpenAI

		self.client = AsyncOpenAI()
		self.model = model
		self.max_completion_tokens = max_completion_tokens

	async def complete(self, prompt: str) -> str:
		response = await self.client.chat.completions.create(
			model=self.model,
			messages=[{"role": "user", "content": prompt}],
			max_completion_tokens=self.max_completion_tokens,
		)
		return (response.choices[0].message.content or "").strip()


class GoogleJudge(AsyncLLMJudge):
	"""Google Gemini judge using google-genai client."""

	def __init__(self, model: str, max_completion_tokens: int = 1000) -> None:
		from google import genai

		self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
		self.model = model
		self.max_completion_tokens = max_completion_tokens

	async def complete(self, prompt: str) -> str:
		from google.genai import types

		response = await self.client.aio.models.generate_content(
			model=self.model,
			contents=prompt,
			config=types.GenerateContentConfig(
				max_output_tokens=self.max_completion_tokens,
			),
		)
		return (response.text or "").strip()


class AnthropicJudge(AsyncLLMJudge):
	"""Anthropic Claude judge using AsyncAnthropic client."""

	def __init__(self, model: str, max_completion_tokens: int = 1000) -> None:
		from anthropic import AsyncAnthropic

		self.client = AsyncAnthropic()
		self.model = model
		self.max_completion_tokens = max_completion_tokens

	async def complete(self, prompt: str) -> str:
		response = await self.client.messages.create(
			model=self.model,
			max_tokens=self.max_completion_tokens,
			messages=[{"role": "user", "content": prompt}],
		)
		return (response.content[0].text or "").strip()


def create_judge(
	provider: str,
	model: str,
	max_completion_tokens: int = 1000,
) -> AsyncLLMJudge:
	"""Factory function to create an LLM judge for the given provider.

	Args:
		provider: One of "openai", "google", "anthropic".
		model: Model identifier for the provider.
		max_completion_tokens: Maximum tokens in the response.

	Returns:
		An AsyncLLMJudge instance.

	Raises:
		ValueError: If provider is not recognized.
	"""
	if provider == "openai":
		return OpenAIJudge(model, max_completion_tokens)
	elif provider == "google":
		return GoogleJudge(model, max_completion_tokens)
	elif provider == "anthropic":
		return AnthropicJudge(model, max_completion_tokens)
	else:
		raise ValueError(
			f"Unknown provider: {provider}. Must be one of: openai, google, anthropic"
		)
