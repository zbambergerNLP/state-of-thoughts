# Argument Quality Experiment

This experiment investigates what reasoning trajectory features predict argument persuasiveness.
We generate 15,000 arguments per topic (5,000 per synthesis type) using STATe of Thoughts, then use
regression analysis to identify which sequential patterns produce better arguments.

The experiment has three subexperiments:

1. **Predictability** (all 5 topics): Can trajectory features predict argument quality?
2. **Controllability** (plastic waste only): Do controller interventions manifest in generated text?
3. **Targeted Generation** (plastic waste only): Do M2-predicted optimal trajectories actually produce better arguments?

## Topics

| Topic | Slug | Subtopics File | Pairwise Judge |
|-------|------|----------------|----------------|
| The government should enforce a total ban on single-use plastics. | `single_use_plastic_specific_subtopics` | `subtopics_specific_single_use_plastic.json` | OpenAI `gpt-5-mini-2025-08-07` |
| Standardized testing should be abolished as a primary measure of student performance. | `standardized_testing` | `subtopics_specific_standardized_testing.json` | Google `gemini-3.1-flash-lite-preview` |
| A special tax should be imposed on meat products to reduce consumption and environmental impact. | `meat_tax` | `subtopics_specific_meat_tax.json` | Anthropic `claude-haiku-4-5-20251001` |
| Social media platforms should enforce a minimum age restriction of 16. | `social_media_age_restriction` | `subtopics_specific_social_media_age_restriction.json` | OpenAI `gpt-5-mini-2025-08-07` |
| The government should implement a universal basic income program. | `universal_basic_income` | `subtopics_specific_universal_basic_income.json` | OpenAI `gpt-5-mini-2025-08-07` |

All topics use stance **PRO**, 20 seeds (2000-2019), 250 final candidates per seed = 5,000 arguments per synthesis type.

## Environment Setup

1. Activate the virtual environment:
   ```bash
   source dspy_reasoning_env/bin/activate
   ```

2. Model weights must be downloaded to a local directory. Set via:
   - CLI argument: `--model_directory /path/to/models`
   - Environment variable: `export MODEL_DIR=/path/to/models`

3. API keys required for pairwise evaluation:
   - `OPENAI_API_KEY` for OpenAI judges
   - `GOOGLE_API_KEY` for Google Gemini judges
   - `ANTHROPIC_API_KEY` for Anthropic Claude judges

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_arguments.py` | Generates arguments with Tree of Thoughts |
| `pairwise_evaluation.py` | Runs pairwise comparisons and computes Bradley-Terry scores (multi-provider) |
| `analyze_m1_vs_m2.py` | Predictability: regression analysis comparing M1 (presence) vs M2 (sequential) features |
| `controllability_evaluation.py` | Controllability: LLM judge checks that controller interventions manifest in text |
| `ablation_trajectory_selection.py` | Targeted generation: generates arguments with forced trajectory selection |
| `ablation_evaluation.py` | Targeted generation: evaluates forced trajectory arguments against baselines |
| `llm_judge.py` | Provider-agnostic async LLM judge abstraction (OpenAI, Google, Anthropic) |

## Step 1: Argument Generation

Generate 5,000 arguments per synthesis type per topic using Tree of Thoughts.

```bash
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

SBATCH scripts are in `scripts/`. Submit all jobs for a topic:

```bash
for f in scripts/{standardized_testing,meat_tax}_synthesis_{strict,faithful,restructured}.sbatch; do
  sbatch "$f"
done
```

## Step 2: Pairwise Evaluation

Run pairwise comparisons with the appropriate judge provider per topic:

```bash
# Single-use plastic (OpenAI judge, default)
python experiments/argument_generation/pairwise_evaluation.py \
    --synthesis_type strict --topic single_use_plastic_specific_subtopics

# Standardized testing (Google Gemini judge)
python experiments/argument_generation/pairwise_evaluation.py \
    --synthesis_type strict --topic standardized_testing \
    --provider google --model gemini-3.1-flash-lite-preview

# Meat tax (Anthropic Claude judge)
python experiments/argument_generation/pairwise_evaluation.py \
    --synthesis_type strict --topic meat_tax \
    --provider anthropic --model claude-haiku-4-5-20251001
```

## Step 3: Predictability Analysis

Regress quality scores on trajectory features (M0/M1/M2 model hierarchy). Runs on all 5 topics.

```bash
# Single topic
python experiments/argument_generation/analyze_m1_vs_m2.py --topic single_use_plastic_specific_subtopics

# All topics + cross-topic unified figures
python experiments/argument_generation/analyze_m1_vs_m2.py --topic all
```

Figures are saved to `figures/predictability/{topic}/` (per-topic) and `figures/predictability/` (cross-topic).

## Step 4: Controllability Evaluation

LLM judge verifies that controller interventions (structure + subtopic prescriptions) manifest in the generated text. Runs on plastic waste topic only.

```bash
# Evaluate all synthesis types
python experiments/argument_generation/controllability_evaluation.py \
    --synthesis_type all --topic single_use_plastic_specific_subtopics

# Recompute summary from existing results (no API calls)
python experiments/argument_generation/controllability_evaluation.py \
    --synthesis_type all --topic single_use_plastic_specific_subtopics --summarize_only
```

Figures are saved to `figures/controllability/`.

## Step 5: Targeted Generation

Validates whether M2-predicted optimal trajectories actually produce better arguments. Runs on plastic waste topic only.

**Step 5a: Generate arguments with forced trajectories** (`ablation_trajectory_selection.py`):

```bash
python experiments/argument_generation/ablation_trajectory_selection.py \
    --synthesis_type all \
    --selection_mode all \
    --topic single_use_plastic_specific_subtopics \
    --model_directory /path/to/models \
    --top_n 50 \
    --samples_per_trajectory 5
```

| Option | Description | Default |
|--------|-------------|---------|
| `--synthesis_type` | `strict`, `faithful`, `restructured`, or `all` | `all` |
| `--selection_mode` | `targeted` (M2), `random`, `m1b`, or `all` | `all` |
| `--top_n` | Number of trajectories to select | 50 |
| `--samples_per_trajectory` | Arguments to generate per trajectory | 5 |

**Step 5b: Evaluate forced trajectory arguments** (`ablation_evaluation.py`):

```bash
# Evaluate each baseline
python experiments/argument_generation/ablation_evaluation.py \
    --synthesis_type strict --baseline_type original

python experiments/argument_generation/ablation_evaluation.py \
    --synthesis_type strict --baseline_type random

python experiments/argument_generation/ablation_evaluation.py \
    --synthesis_type strict --baseline_type m1b

# Generate summary figures and tables (no API calls)
python experiments/argument_generation/ablation_evaluation.py --create_summary
```

Figures are saved to `figures/targeted_generation/`.

## Data Structure

Results are stored in `argument_data/{topic}/synthesis_{strict,faithful,restructured}/`:

### Generation + Ranking
- `{topic}_synthesis_*_all_results.csv`: Generated arguments with trajectory metadata
- `pairwise_comparisons.jsonl`: Raw pairwise comparison results
- `pairwise_comparisons_bt_scores.csv`: Arguments with Bradley-Terry quality scores

### Predictability Analysis
- `m2_coefficients_*.csv`: Fitted M2 LASSO coefficients
- `m2_trajectory_rankings_*.csv`: All trajectory combinations ranked by M2 predicted score (gitignored, large)

### Controllability
- `controllability/results.jsonl`: Raw LLM judge results
- `controllability/summary.csv`: Aggregated pass rates

### Targeted Generation
- `targeted_generation/targeted_forced_results_*.csv`: Arguments from M2-predicted optimal trajectories
- `targeted_generation/random_forced_results_*.csv`: Arguments from random trajectories
- `targeted_generation/m1b_forced_results_*.csv`: Arguments from M1b-predicted trajectories
- `targeted_generation/targeted_vs_{baseline}_comparisons.jsonl`: Pairwise comparison results
- `targeted_generation/targeted_vs_{baseline}_bt_scores.csv`: Combined BT scores

## Figures

```
figures/
  predictability/
    {topic}/                          # per-topic figures
    unified_cross_topic_test_r2.pdf   # cross-topic comparison
    strict_cross_topic_test_r2.pdf
    cross_topic_by_synthesis_test_r2.pdf
    unified_length_histograms.pdf
    unified_m2_alpha_selection.pdf
    unified_m2_feature_categories.pdf
  controllability/
    controllability_{synthesis_type}.pdf
    controllability_unified.pdf
  targeted_generation/
    length_histograms_unified_{synthesis_type}.pdf
```

## Replicating Results Without Model Calls

All data files are committed, so figures and tables can be regenerated without GPU or API access:

```bash
# Predictability figures (all topics)
python experiments/argument_generation/analyze_m1_vs_m2.py --topic all

# Controllability figures
python experiments/argument_generation/controllability_evaluation.py \
    --synthesis_type all --topic single_use_plastic_specific_subtopics --summarize_only

# Targeted generation figures and tables
python experiments/argument_generation/ablation_evaluation.py --create_summary
```
