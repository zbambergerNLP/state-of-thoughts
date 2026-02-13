"""
Tests for experiments.shared_utils module.

This module provides unit tests for shared utility functions used across
ToT and baseline experiments. It ensures correct parameter creation,
run configuration, and naming conventions.
"""

# Standard library imports
import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

# Local imports
from experiments import shared_utils
from predict.tree_of_thoughts.tree_parameters import TreeOfThoughtsParameters
from utilities_for_tests import (
	MockGenerativeLocalVLLM,
	MockScoringLocalVLLM,
)


@pytest.fixture
def default_args() -> argparse.Namespace:
    """Create a default argparse.Namespace with common experiment flags."""
    args = argparse.Namespace()

    # Baseline defaults
    args.model = "test_model"
    args.model_directory = str(Path(tempfile.gettempdir()) / "models")
    args.output_dir = str(Path(tempfile.gettempdir()) / "output")
    args.seeds = None
    args.seed = 42
    args.baseline_temperature = 0.7
    args.baseline_signature = "InstructionFollowing"

    # ToT defaults
    args.action_space_name = None
    args.action_space_paths = None
    args.experiment_mode = "synthesis_faithful"
    args.depth = 2
    args.n_samples_generation = 1
    args.top_k = 1
    args.n_samples_judge = 1
    args.num_final_candidates = 1
    args.n_final_responses_per_trajectory = 1
    args.do_pruning = False
    args.use_self_consistency = False
    args.evaluator_type = "generator"
    args.controller_type = "generator"

    # Sampling defaults
    args.generator_temperature = 0.7
    args.generator_max_tokens = 15000
    args.generator_top_p = 0.9
    args.generator_top_k = 50
    args.generator_min_p = 0.0
    args.generator_use_beam_search = False

    args.controller_temperature = None
    args.controller_max_tokens = 15000
    args.controller_top_p = 0.9
    args.controller_top_k = 50
    args.controller_min_p = 0.0
    args.controller_use_beam_search = False

    args.judge_temperature = 0.5
    args.judge_max_tokens = 15000
    args.judge_top_p = 0.9
    args.judge_top_k = 10
    args.judge_min_p = 0.0
    args.judge_use_beam_search = False

    return args


@pytest.mark.parametrize(
    ["name", "expected"],
    [
        pytest.param(
            "InstructionFollowing",                         # name
            "instruction_following",                        # expected
            id="simple_camel"
        ),
        pytest.param(
            "InstructionFollowingCoT",                      # name
            "instruction_following_cot",                    # expected
            id="with_cot"
        ),
        pytest.param(
            "ToT",                                          # name
            "tot",                                          # expected
            id="acronym_only"
        ),
        pytest.param(
            "InstructionFollowingWithCoTAndToT",            # name
            "instruction_following_with_cot_and_tot",       # expected
            id="mixed_acronyms"
        ),
        pytest.param(
            "Simple",                                       # name
            "simple",                                       # expected
            id="single_word"
        ),
    ]
)
def test_camel_to_snake_for_run_names(name: str, expected: str) -> None:
    """Test conversion from CamelCase to snake_case handles acronyms correctly."""
    assert shared_utils._camel_to_snake_for_run_names(name) == expected


@pytest.mark.parametrize(
    ["overrides", "expected_params"],
    [
        pytest.param(
            {},                                             # overrides
            {                                               # expected_params
                "generator_temperature": 0.7,
                "generator_max_tokens": 15000,
                "controller_max_tokens": 15000,
                "judge_max_tokens": 15000,
                "controller_temperature": None,
            },
            id="defaults"
        ),
        pytest.param(
            {                                               # overrides
                "generator_max_tokens": 100,
                "controller_max_tokens": 200,
                "judge_max_tokens": 300,
                "generator_temperature": 0.5,
                "controller_temperature": 0.9,
            },
            {                                               # expected_params
                "generator_temperature": 0.5,
                "generator_max_tokens": 100,
                "controller_max_tokens": 200,
                "judge_max_tokens": 300,
                "controller_temperature": 0.9,
            },
            id="overrides"
        ),
        pytest.param(
            {                                               # overrides
                "generator_temperature": 1.0,
                "baseline_temperature": 0.1,
            },
            {                                               # expected_params
                "generator_temperature": 1.0,
            },
            id="temp_precedence"
        ),
        pytest.param(
            {"depth": 5},                                    # overrides
            {"depth": 5},                                    # expected_params
            id="depth_override"
        )
    ]
)
def test_create_tot_params_from_args(
    default_args: argparse.Namespace,
    overrides: dict[str, Any],
    expected_params: dict[str, Any]
) -> None:
    """Test creation of TreeOfThoughtsParameters from args with varied inputs."""
    vars(default_args).update(overrides)
    actual_params = shared_utils.create_tot_params_from_args(default_args)
    assert isinstance(actual_params, TreeOfThoughtsParameters)
    actual_dict = asdict(actual_params)
    actual_subset = {k: actual_dict[k] for k in expected_params.keys()}
    assert actual_subset == expected_params


@pytest.mark.parametrize(
    ["seed", "temp", "config_name", "expected"],
    [
        pytest.param(42, 0.7, "my_config", "my_config_seed_42_temp_0_7", id="normal"),
        pytest.param(123, 1.0, "baseline", "baseline_seed_123_temp_1_0", id="int_temp"),
        pytest.param(0, 0.0, "test", "test_seed_0_temp_0_0", id="zeros"),
        pytest.param(999, 0.05, "complex_name", "complex_name_seed_999_temp_0_05", id="small_temp"),
    ]
)
def test_get_run_name(seed: int, temp: float, config_name: str, expected: str):
    """Test standardized run name string generation."""
    assert shared_utils.get_run_name(seed, temp, config_name) == expected


def test_get_run_name_empty_config():
    """Test get_run_name raises AssertionError for empty config_name."""
    with pytest.raises(AssertionError, match="config_name must not be empty"):
        shared_utils.get_run_name(42, 0.7, "")


@pytest.mark.parametrize(
    ["metadata", "expected"],
    [
        pytest.param(
            {"baseline_signature": "InstructionFollowing"},     # metadata
            "baseline_InstructionFollowing",                    # expected
            id="baseline"
        ),
        pytest.param(
            {                                                   # metadata
                "tot_signature": "InstructionFollowingWithReasoning",
                "action_space_name": "controlled",
                "experiment_mode": "synthesis_faithful",
            },
            "tot_controlled_instruction_following_with_reasoning_synthesis_faithful",
            id="tot_controlled"
        ),
        pytest.param(
            {                                                   # metadata
                "tot_signature": "InstructionFollowing",
                "experiment_mode": "mode",
            },
            "tot_uncontrolled_instruction_following_mode",      # expected
            id="tot_default_action_space"
        ),
        pytest.param(
            {},                                                 # metadata
            "unknown",                                          # expected
            id="empty_metadata"
        ),
    ]
)
def test_build_preset_key_from_metadata(metadata: dict[str, Any], expected: str) -> None:
    """Test preset key derivation from run metadata."""
    assert shared_utils.build_preset_key_from_metadata(metadata) == expected


@pytest.mark.parametrize(
    ["action_space_name", "paths", "tot_sig", "exp_mode", "depth", "expected_dir_name"],
    [
        pytest.param(
            "uncontrolled",                                                 # action space name
            [],                                                             # paths
            "MySig",                                                        # tot signature
            "mode",                                                         # experiment mode
            2,                                                              # depth
            "uncontrolled_tot_d2_s1_k1_my_sig_mode_seed_42_temp_0_7",       # expected directory name
            id="uncontrolled"
        ),
        pytest.param(
            "controlled",                                                   # action space name
            ["p.json"],                                                     # paths
            "Sig",                                                          # tot signature
            "concl",                                                        # experiment mode
            3,                                                              # depth
            "controlled_tot_d3_s1_k1_controlled_sig_concl_seed_42_temp_0_7", # expected directory name
            id="controlled"
        ),
        pytest.param(
            "uncontrolled",                                                 # action space name
            [],                                                             # paths
            "S",                                                            # tot signature
            "m",                                                            # experiment mode
            2,                                                              # depth
            "uncontrolled_tot_d2_s1_k1_s_m_seed_42_temp_0_7",               # expected directory name
            id="uncontrolled_caps"
        ),
        pytest.param(
            "uncontrolled",                                                 # action space name
            [],                                                             # paths
            "CoTSig",                                                       # tot signature
            "m",                                                            # experiment mode
            2,                                                              # depth
            "uncontrolled_tot_d2_s1_k1_cot_sig_m_seed_42_temp_0_7",         # expected directory name
            id="cot_acronym"
        ),
    ]
)
def test_setup_tot_run_naming(
    default_args: argparse.Namespace,
    action_space_name: str,
    paths: list[str],
    tot_sig: str,
    exp_mode: str,
    depth: int,
    expected_dir_name: str
) -> None:
    """Test setup_tot_run generates correct directory names and configurations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_root = Path(tmp_dir)
        experiment_name = "test_exp"
        default_args.depth = depth

        gen_lm = MockGenerativeLocalVLLM(responses=[[["test"]]])
        reranker_lm = MockScoringLocalVLLM(rerank_responses=[[0.5]])

        action_space_paths = []
        if paths:
            for p_str in paths:
                p = repo_root / p_str
                p.touch()
                action_space_paths.append(p)

        default_args.experiment_mode = exp_mode
        default_args.tot_signature = tot_sig
        default_args.generator_temperature = 0.7
        default_args.action_space_paths = action_space_paths
        default_args.action_space_name = action_space_name

        _, _, output_dir = shared_utils.setup_tot_run(
            args=default_args,
            seed=42,
            experiment_name=experiment_name,
            gen_lm=gen_lm,
            reranker_lm=reranker_lm,
            repo_root=repo_root
        )

        assert output_dir.name == expected_dir_name
        assert output_dir.exists()


@pytest.mark.parametrize(
    ["signature", "temp", "seed", "expected_dir_name"],
    [
        pytest.param(
            "IF",                                       # signature
            0.7,                                        # temp
            42,                                         # seed
            "baseline_i_f_seed_42_temp_0_7",            # expected directory name
            id="simple"
        ),
        pytest.param(
            "IFCoT",                                    # signature
            0.1,                                        # temp
            123,                                        # seed
            "baseline_i_f_cot_seed_123_temp_0_1",       # expected directory name
            id="cot"
        ),
        pytest.param(
            "CustomSig",                                # signature
            1.0,                                        # temp
            0,                                          # seed
            "baseline_custom_sig_seed_0_temp_1_0",      # expected directory name
            id="custom"
        ),
        pytest.param(
            "Sig_With_Underscores",                     # signature
            0.5,                                        # temp
            42,                                         # seed
            "baseline_sig_with_underscores_seed_42_temp_0_5", # expected directory name
            id="underscores"
        ),
        pytest.param(
            "CamelCaseSig",                             # signature
            0.7,                                        # temp
            42,                                         # seed
            "baseline_camel_case_sig_seed_42_temp_0_7", # expected directory name
            id="camel"
        ),
    ]
)
def test_setup_baseline_run_naming(
    default_args: argparse.Namespace,
    signature: str,
    temp: float,
    seed: int,
    expected_dir_name: str
) -> None:
    """Test setup_baseline_run naming convention."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_root = Path(tmp_dir)
        experiment_name = "test_baseline"
        gen_lm = MockGenerativeLocalVLLM(responses=[[["test"]]])
        default_args.baseline_temperature = temp
        default_args.baseline_signature = signature

        _, _, output_dir = shared_utils.setup_baseline_run(
            args=default_args,
            seed=seed,
            experiment_name=experiment_name,
            gen_lm=gen_lm,
            repo_root=repo_root
        )

        assert output_dir.name == expected_dir_name
        assert output_dir.exists()


@pytest.mark.parametrize(
    ["files_to_titles", "action_space_paths", "expected_output"],
    [
        pytest.param(
            None,
            None,
            """
Consider the following dimensions for your response:

STYLE:
1. simple: Use simple words.
2. complex: Use complex vocabulary.

TONE:
1. happy: Sound cheerful.
2. sad: Sound melancholic.
            """,
            id="directory_auto_find"
        ),
        pytest.param(
            None,
            ["tone.json", "style.json"],
            """
Consider the following dimensions for your response:

TONE:
1. happy: Sound cheerful.
2. sad: Sound melancholic.

STYLE:
1. simple: Use simple words.
2. complex: Use complex vocabulary.
            """,
            id="explicit_paths_reverse_order"
        ),
        pytest.param(
            {"style.json": "WRITING STYLE", "tone.json": "AFFECT"},
            ["style.json", "tone.json"],
            """
Consider the following dimensions for your response:

WRITING STYLE:
1. simple: Use simple words.
2. complex: Use complex vocabulary.

AFFECT:
1. happy: Sound cheerful.
2. sad: Sound melancholic.
            """,
            id="custom_titles"
        ),
    ]
)
def test_format_tool_descriptions_from_action_space(files_to_titles, action_space_paths, expected_output):
    """Test that action space JSONs are correctly formatted into tool descriptions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        action_space_dir = Path(tmp_dir)

        style_json_content = {
            "name": "style",
            "choices": {
                "simple": {"definition": "Use simple words."},
                "complex": {"definition": "Use complex vocabulary."}
            }
        }
        tone_json_content = {
            "name": "tone",
            "choices": {
                "happy": {"definition": "Sound cheerful."},
                "sad": {"definition": "Sound melancholic."}
            }
        }

        style_path = action_space_dir / "style.json"
        tone_path = action_space_dir / "tone.json"

        with open(style_path, "w") as f:
            json.dump(style_json_content, f)
        with open(tone_path, "w") as f:
            json.dump(tone_json_content, f)

        real_paths = None
        if action_space_paths:
            real_paths = [action_space_dir / p for p in action_space_paths]

        result = shared_utils.format_tool_descriptions_from_action_space(
            action_space_dir=action_space_dir,
            files_to_titles=files_to_titles,
            action_space_paths=real_paths
        )

        assert result.strip() == expected_output.strip()


@pytest.mark.parametrize(
    ["results", "fieldnames", "fields_to_skip", "expected_csv", "expected_error"],
    [
        pytest.param(
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            None,
            None,
            """
a,b
1,2
3,4
""".strip(),
            None,
            id="simple_dicts"
        ),
        pytest.param(
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            ["a"],
            None,
            """
a
1
3
""".strip(),
            None,
            id="fieldnames_override"
        ),
        pytest.param(
            [{"a": 1, "b": 2, "results_file": "skip_me"}, {"a": 3, "b": 4, "results_file": "me_too"}],
            None,
            ["results_file"],
            """
a,b
1,2
3,4
""".strip(),
            None,
            id="fields_to_skip"
        ),
        pytest.param(
            [
                shared_utils.ExampleResult(inputs={"q": "1"}, outputs=["a1"]),
                shared_utils.ExampleResult(inputs={"q": "2"}, outputs=["a2"]),
            ],
            ["outputs", "failure_reason"],
            None,
            """
outputs,failure_reason
['a1'],
['a2'],
""".strip(),
            None,
            id="dataclass_results"
        ),
        pytest.param(
            [],
            None,
            None,
            "",
            None,
            id="empty_results"
        ),
        pytest.param(
            [1, 2, 3],
            None,
            None,
            None,
            ValueError,
            id="invalid_type_error"
        ),
    ]
)
def test_export_results_csv(results, fieldnames, fields_to_skip, expected_csv, expected_error):
    """Test exporting results to CSV, handling dicts, dataclasses, and edge cases."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test.csv"

        if expected_error:
            with pytest.raises(expected_error):
                shared_utils.export_results_csv(results, output_path, fieldnames, fields_to_skip)
            return

        shared_utils.export_results_csv(results, output_path, fieldnames, fields_to_skip)

        if not results:
            # According to the implementation, if results is empty, it returns early and does not write the file.
            assert not output_path.exists()
            return

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            content = f.read()

        # Standardize line endings for comparison
        content_clean = content.strip().replace("\r\n", "\n")
        expected_clean = expected_csv.strip().replace("\r\n", "\n")
        assert content_clean == expected_clean
