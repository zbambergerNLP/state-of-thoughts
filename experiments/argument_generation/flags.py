"""Argument-generation specific CLI flags and helpers.

This module extends the shared experiment CLI parser (`experiments/flags.py`) with
flags used only by the argument-generation experiment, plus small helpers for
retrieving credentials.
"""

# Standard library imports
import argparse
import os

# Third-party imports
from dotenv import load_dotenv

# Local imports
from experiments.flags import parser

# Add argument generation specific flags
argument_generation_group = parser.add_argument_group(
	title="Argument Generation Configuration",
	description="Settings for argument generation",
)
argument_generation_group.add_argument(
	"--openai_api_key",
	type=str,
	default=None,
	help=(
		"OpenAI API key for evaluation. If not provided, will check OPENAI_API_KEY env var or "
		".env file."
	),
)

argument_generation_group.add_argument(
	"--mode",
	type=str,
	choices=["generation", "evaluation"],
	required=False,
	help="Mode to run the experiment in: 'generation' or 'evaluation'",
)

argument_generation_group.add_argument(
	"--test_run",
	action="store_true",
	help="Run a small test version (1 topic, 1 stance) for debugging.",
)

argument_generation_group.add_argument(
	"--judge_model",
	type=str,
	default="gpt-5.1",
	help="Language model to use for judging arguments during evaluation.",
)


def get_openai_api_key(args: argparse.Namespace) -> str:
	"""
	Retrieve OpenAI API key from args, env var, or .env file.
	Priority:
	1. --openai_api_key flag
	2. OPENAI_API_KEY environment variable
	3. .env file in project root
	"""
	if args.openai_api_key:
		return args.openai_api_key

	# Check if already in env
	if os.environ.get("OPENAI_API_KEY"):
		key = os.environ.get("OPENAI_API_KEY")
		if key is not None:
			return key

		# Try loading from .env (searches current + parents by default).
		load_dotenv()

	# Check env var again after potential load_dotenv
	key = os.environ.get("OPENAI_API_KEY")
	if key is None:
		raise ValueError(
			"OPENAI_API_KEY not found in environment,.env file,or command line args"
		)
	return key
