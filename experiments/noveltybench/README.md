# NoveltyBench Evaluation

Evaluation pipeline for measuring response diversity and quality using the NoveltyBench benchmark with **vLLM** support.

## Installation

Before running any experiments, install the package from the project root:

```bash
cd /path/to/state-of-thoughts
pip install -e .
```

This installs the package in editable mode, making all local modules (`adapter`, `lm`, `predict`, `signatures`, etc.) available for import.

## What NoveltyBench Tests

NoveltyBench evaluates language models' ability to generate **diverse, high-quality responses** to the same prompt. Unlike traditional benchmarks that only assess the single best response, NoveltyBench measures:

1. **Response Diversity (Distinct_k)**: How many semantically distinct responses a model can produce
2. **Response Quality (Utility_k)**: The combined measure of novelty and quality across multiple generations

Many open-ended tasks—creative writing, argument generation, exploratory Q&A—depend on models that can offer several useful angles without sacrificing quality. NoveltyBench zeroes in on that balance by scoring both diversity and utility across the full set of generations.

## Run Evaluation

The workflow is split into three components:

1. **Baseline inference** (generate `generations.json` / `results.json`)
2. **Tree-of-Thought (ToT) inference** (generate `generations.json` / `results.json`)
3. **Evaluation** (partition → score → summarize results → generate plots)

Baseline and ToT scripts run **inference only**. Run the evaluate script separately to partition, score, and plot.

### 1. Baseline inference

Runs one baseline signature variant per invocation. Use `--baseline_signature` with one of: `InstructionFollowing`, `InstructionFollowingCoT`, `InstructionFollowingWithTools`, `InstructionFollowingWithToolsCoT`.

```bash
python -m experiments.noveltybench.run_baseline_noveltybench_experiment \
  --model Qwen3-4B-Instruct-2507 \
  --model_directory /path/to/model_storage \
  --dataset_path /path/to/dataset_storage/noveltybench \
  --split train \
  --results_split debugging \
  --subset curated \
  --max_examples 5 \
  --baseline_signature InstructionFollowing \
  --score_cuda_device 1 \
  --seeds 1024 1025 1026 \
  --baseline_temperature 0.7
```

Use `--results_split` to write outputs under a different label (e.g. `debugging`) than `--split`. Use `--max_examples` to limit the dataset size for quick runs.

### 2. Tree-of-Thought (ToT) inference

ToT uses a tree search with optional action spaces. Set `--action_space_name` to `controlled` (personalities + target_audiences) or `uncontrolled` (no action space). Action space paths are resolved automatically.

```bash
python -m experiments.noveltybench.run_tot_noveltybench_experiment \
  --model Qwen3-4B-Instruct-2507 \
  --model_directory /path/to/model_storage \
  --dataset_path /path/to/dataset_storage/noveltybench \
  --reranker_model Qwen3-Reranker-8B \
  --split train \
  --results_split debugging \
  --subset curated \
  --max_examples 5 \
  --action_space_name controlled \
  --tot_signature InstructionFollowingWithReasoning \
  --depth 1 \
  --n_samples_generation 10 \
  --top_k 10 \
  --num_final_candidates 10 \
  --n_final_responses_per_trajectory 1 \
  --score_cuda_device 2 \
  --seeds 1024 1025 1026 \
  --generator_temperature 0.7
```

For `--action_space_name uncontrolled`, no reranker is required. For `controlled` with `--controller_type reranker` or `--evaluator_type reranker`, `--reranker_model` is required. Use `--generative_gpu_index` and `--reranker_gpu_index` to assign different GPUs.

### 3. Evaluation (partition + score + plot)

Scans an experiment directory for `results.json` files, runs partitioning and scoring, then generates summary tables and plots:

```bash
python -m experiments.noveltybench.evaluate_noveltybench_experiment \
  --experiment_dir experiments/results/noveltybench_curated_debugging/qwen3_4b_instruct_2507 \
  --partition_workers 2
```

Use `--plot_only` to regenerate plots and summary tables only (skip partition + score). Use `--runs_to_evaluate` to evaluate specific run directories. Use `--score_cuda_device` to pin scoring to a GPU. Use `--skip_score` to skip the scoring stage (requires 32GB+ RAM).

## Key Flags

### Model parameters

- **`--model`**: Model name, resolved against `--model_directory`.
- **`--model_directory`**: Base directory for models (required).
- **`--reranker_model`**: Required for ToT when using reranker controller/evaluator with `action_space_name=controlled`.
- **`--scorer_model`**: Scorer for evaluation (default: Skywork-Reward-Gemma-2-27B-v0.2).

### Dataset parameters

- **`--split`**: `train`, `test`, or `debugging`.
- **`--subset`**: `curated` or `wildchat`.
- **`--dataset_path`**: Path to dataset directory.
- **`--max_examples`**: Limit dataset size for quick runs.
- **`--results_split`**: Override the results directory label (e.g. `debugging` when `--split train`).

### Sweeps

- **`--seed`** / **`--seeds`**: Single seed or list of seeds for multiple runs.

### Baseline-only

- **`--baseline_signature`**: One of `InstructionFollowing`, `InstructionFollowingCoT`, `InstructionFollowingWithTools`, `InstructionFollowingWithToolsCoT`.
- **`--baseline_temperature`**: Sampling temperature (default 0.7).

### ToT-only

- **`--action_space_name`**: `controlled` or `uncontrolled`. Paths are resolved from `experiments/noveltybench/action_space/`.
- **`--tot_signature`**: `InstructionFollowingWithReasoning` or `InstructionFollowingWithReasoningAndTools`.
- **`--depth`**: Tree depth (default 1).
- **`--n_samples_generation`**, **`--top_k`**, **`--num_final_candidates`**, **`--n_final_responses_per_trajectory`**.
- **`--generator_temperature`**: Temperature for ToT generator.
- **`--controller_type`**: `generator` or `reranker` (default reranker).
- **`--evaluator_type`**: `generator`, `reranker`, or `programmatic` (default generator).

### Evaluation / scoring

- **`--partition_workers`**: Workers for partitioning (default 1).
- **`--partition_algorithm`**: `classifier` (DeBERTa), `bertscore`, or `unigram` (default classifier).
- **`--score_cuda_device`**: `auto` or device id for scoring.
- **`--plot_only`**: Regenerate plots only, skip partition + score.
- **`--runs_to_evaluate`**: List of specific run dirs to evaluate.
- **`--skip_score`**: Skip scoring stage.

## Output Metrics

- **`mean_distinct`**: Average number of distinct response categories (higher is better).
- **`mean_utility`**: Combined novelty and quality score (higher is better).

## Outputs

Runs are written under:

`experiments/results/noveltybench_{subset}_{results_split}/{model_name}/.../`

Each run directory contains:

- `generations.json`
- `results.json`

## References

- **Paper**: [NoveltyBench: Evaluating Language Models for Humanlike Diversity](https://arxiv.org/abs/2504.05228)
- **Dataset**: [yimingzhang/novelty-bench](https://huggingface.co/datasets/yimingzhang/novelty-bench)
