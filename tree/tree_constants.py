"""Constants specific to tree structures and Tree of Thoughts."""
import enum


class ReasoningState(enum.StrEnum):
	"""Enum for reasoning state fields."""
	INPUT = "input"
	REASONING = "reasoning"
	FEEDBACK = "feedback"
	OUTPUT = "output"


class NodeField(enum.StrEnum):
	"""Enum for node fields."""
	STATE = "state"
	STEP = "step"
	SCORE = "score"
	SOUNDNESS = "soundness"
	PROMISE = "promise"
	QUALITY = "quality"
	INDEX = "index"
	LAYER = "layer"
	IS_PRUNED = "is_pruned"
	PARENT_ID = "parent_id"
	CHILDREN_IDS = "children_ids"


class TreeStructure(enum.StrEnum):
	"""Enum for tree structure fields."""
	ROOT = "root"
	NODES = "nodes"
	EDGES = "edges"
	LAYERS = "layers"


class ToTField(enum.StrEnum):
	"""Enum for Tree of Thoughts fields."""
	RESPONSES = "responses"
	REASONING_STEPS = "reasoning_steps"
	RESPONSE_STRINGS = "response_strings"
	TREE = "tree"
	COST = "cost"
	RUNTIME = "runtime"


class ToTParam(enum.StrEnum):
	"""Enum for Tree of Thoughts parameters."""
	DEPTH = "depth"
	N_SAMPLES_GENERATION = "n_samples_generation"
	N_SAMPLES_JUDGE = "n_samples_judge"
	JUDGE_TEMPERATURE = "judge_temperature"
	GENERATOR_TEMPERATURE = "generator_temperature"
	CONTROLLER_TEMPERATURE = "controller_temperature"
	NUM_FINAL_CANDIDATES = "num_final_candidates"
	DO_PRUNING = "do_pruning"
	USE_SELF_CONSISTENCY = "use_self_consistency"
	CONTROLLER_TOP_P = "controller_top_p"
