# Argument Quality Experiment

This experiment investigates what reasoning trajectory features predict argument persuasiveness.
We generate 15,000 arguments (5,000 per synthesis type) using Tree of Thoughts, then use
regression analysis to identify which sequential patterns produce better arguments.

## Experiment Overview

1. **Generation**: Generate arguments across 20 random seeds using a fixed topic/stance
2. **Ranking**: Pairwise LLM comparisons -> Bradley-Terry scores for argument quality
3. **Analysis**: Regress quality scores on trajectory features (M1 baseline vs M2 sequential model)
4. **Validation/Ablation**: Generate new arguments using M2-predicted best trajectories and verify they win

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_arguments.py` | Generates arguments with Tree of Thoughts |
| `pairwise_evaluation.py` | Runs pairwise comparisons and computes Bradley-Terry scores |
| `analyze_m1_vs_m2.py` | Regression analysis comparing M1 (presence) vs M2 (sequential) features |
| `ablation_trajectory_selection.py` | Generates arguments with forced trajectory selection for validation |
| `ablation_evaluation.py` | Evaluates whether M2-predicted trajectories produce better arguments |

## Running Experiments

### Environment Setup

Before running experiments, ensure:

1. The virtual environment is activated:
   ```bash
   source dspy_reasoning_env/bin/activate
   ```

2. Model weights are downloaded to a local directory. Set this path via:
   - CLI argument: `--model_directory /path/to/models`
   - Environment variable: `export MODEL_DIR=/path/to/models`

### Running Locally (Single GPU Setup)

For local development or single-machine runs:

```bash
# Basic argument generation
python experiments/argument_generation/generate_arguments.py \
    --model Qwen3-30B-A3B-Instruct-2507 \
    --model_directory /path/to/models \
    --reranker_model Qwen3-Reranker-8B \
    --generative_gpu_index 0 \
    --reranker_gpu_index 1 \
    --outputs_directory experiments/argument_generation/outputs \
    --outputs_filename test_run \
    --topic "The government should enforce a total ban on single-use plastics." \
    --stance PRO \
    --seed 42 \
    --depth 3 \
    --n_samples_generation 50 \
    --top_k 50 \
    --num_final_candidates 100 \
    --generator_temperature 0.7 \
    --experiment_mode synthesis_strict \
    --do_save_tree
```

### Running Analysis and Ablation

After generating arguments and computing Bradley-Terry scores:

```bash
# Step 1: Run M1 vs M2 regression analysis
python experiments/argument_generation/analyze_m1_vs_m2.py

# Step 2: Generate ablation arguments with forced trajectories
# This uses M2 coefficients to select optimal trajectories and generates new arguments
python experiments/argument_generation/ablation_trajectory_selection.py \
    --synthesis_type all \
    --selection_mode all \
    --model_directory /path/to/models \
    --top_n 50 \
    --samples_per_trajectory 5

# Step 3: Evaluate ablation arguments against baselines
# Run for each synthesis type and baseline combination
python experiments/argument_generation/ablation_evaluation.py \
    --synthesis_type strict \
    --baseline_type original

python experiments/argument_generation/ablation_evaluation.py \
    --synthesis_type strict \
    --baseline_type random

python experiments/argument_generation/ablation_evaluation.py \
    --synthesis_type strict \
    --baseline_type m1b

# Step 4: Generate summary figures (after all evaluations complete)
python experiments/argument_generation/ablation_evaluation.py --create_summary
```

#### Ablation Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--synthesis_type` | `strict`, `faithful`, `restructured`, or `all` | `all` |
| `--selection_mode` | `targeted` (M2), `random`, `m1b`, or `all` | `all` |
| `--top_n` | Number of trajectories to select | 50 |
| `--samples_per_trajectory` | Arguments to generate per trajectory | 5 |

## Data

Results are stored in `argument_data/synthesis_{strict,faithful,restructured}/`:

### Generation Phase
- `explainability_synthesis_*_all_results.csv` - Generated arguments with trajectory metadata
- `pairwise_comparisons.jsonl` - Raw pairwise comparison results
- `pairwise_comparisons_bt_scores.csv` - Arguments with Bradley-Terry (BT) quality scores

### Analysis Phase
- `m2_coefficients_*.csv` - Fitted M2 (sequential) model coefficients

### Ablation/Validation Phase
Arguments generated with forced trajectory selection:
- `targeted_forced_results_*.csv` - Arguments using M2-predicted optimal trajectories
- `random_forced_results_*.csv` - Arguments using randomly selected trajectories
- `m1b_forced_results_*.csv` - Arguments using M1b-predicted trajectories (baseline)

Pairwise comparisons for ablation validation and csv files with ne BT scores:
- `targeted_vs_original_*.jsonl/.csv` - Targeted (M2) vs original generation
- `targeted_vs_random_*.jsonl/.csv` - Targeted (M2) vs random trajectories
- `targeted_vs_m1b_*.jsonl/.csv` - Targeted (M2) vs M1b baseline

## Figures

`analyze_m1_vs_m2.py` and `ablation_evaluation.py --create_summary` output figures to `figures/`
