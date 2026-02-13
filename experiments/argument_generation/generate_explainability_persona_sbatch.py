"""
Generate single SBATCH file for explainability persona experiment.

Creates 1 sbatch file that runs all 15 combinations (5 personas × 3 seeds) sequentially.
Models loaded once and shared across all runs.

Configuration:
- Personas: openness, conscientiousness, extraversion, agreeableness, neuroticism
- Seeds: 42, 123, 456
- Model: Qwen3-30B-A3B-Instruct-2507
- Reranker: Qwen3-Reranker-8B
- Resources: 2 GPUs, 16 CPUs, 40GB/GPU, 2.5 hours
- Output: Single shared CSV file (all runs append)

Usage:
	python experiments/argument_generation/generate_explainability_persona_sbatch.py

Output:
	Single .sbatch file: experiments/argument_generation/scripts/explainability_persona.sbatch
"""

import os
from pathlib import Path

# Experiment configuration
PERSONAS = [
	"openness",
	"conscientiousness",
	"extraversion",
	"agreeableness",
	"neuroticism",
]
SEEDS = [42, 123, 456]

# Directory configuration
SCRIPTS_DIR = "experiments/argument_generation/scripts"
LOGS_DIR = "experiments/logs/argument_generation/explainability_persona"

# Model configuration
MODEL = "Qwen3-30B-A3B-Instruct-2507"
RERANKER = "Qwen3-Reranker-8B"
MODEL_DIR = "/projects/BSTEWART/model_storage"
OUTPUT_DIR = "experiments/argument_generation/explainability_experiment"
OUTPUT_FILENAME = "explainability_persona"

# SBATCH template with bash loops for all combinations
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=explainability_persona_all
#SBATCH --output={logs_dir}/explainability_persona_all_%j.log
#SBATCH --error={logs_dir}/explainability_persona_all_%j.log
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=40G
#SBATCH --partition=ailab

# Environment Setup
source dspy_reasoning_env/bin/activate

echo "========================================="
echo "Explainability Persona Experiment (Batch)"
echo "========================================="
echo "Total combinations: 15 (5 personas × 3 seeds)"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Define parameter arrays
PERSONAS=(openness conscientiousness extraversion agreeableness neuroticism)
SEEDS=(42 123 456)

total=15
current=0

# Nested loops
for persona in "${{PERSONAS[@]}}"; do
	for seed in "${{SEEDS[@]}}"; do
		((current++))

		echo "========================================="
		echo "[$current/$total] Running: persona=$persona, seed=$seed"
		echo "Start time: $(date)"
		echo "========================================="

		# Run experiment (continue on error)
		python experiments/argument_generation/persona_explainability_experiment.py \\
			--model {model} \\
			--model_directory {model_dir} \\
			--reranker_model {reranker} \\
			--generative_gpu_index 0 \\
			--reranker_gpu_index 1 \\
			--outputs_directory {output_dir} \\
			--outputs_filename {output_filename} \\
			--persona "$persona" \\
			--seed "$seed" \\
			--depth 3 \\
			--n_samples_generation 100 \\
			--top_k 50 \\
			--n_samples_judge 1 \\
			--num_final_candidates 100 \\
			--generator_temperature 0.7 \\
			--experiment_mode synthesis_strict \\
			--do_save_tree || echo "ERROR: Failed for persona=$persona, seed=$seed (continuing...)"

		echo ""
		echo "Completed: persona=$persona, seed=$seed"
		echo "End time: $(date)"
		echo "Progress: $current/$total"
		echo "========================================="
		echo ""
	done
done

echo "========================================="
echo "All jobs completed!"
echo "End time: $(date)"
echo "========================================="
"""


def main():
	"""Generate single consolidated SBATCH file for all persona × seed combinations."""
	# Create scripts directory
	Path(SCRIPTS_DIR).mkdir(parents=True, exist_ok=True)

	# Single output file
	output_file = f"{SCRIPTS_DIR}/explainability_persona.sbatch"

	print("=" * 60)
	print("Explainability Persona SBATCH Generator")
	print("=" * 60)
	print(f"Personas: {len(PERSONAS)} ({', '.join(PERSONAS)})")
	print(f"Seeds: {len(SEEDS)} ({', '.join(map(str, SEEDS))})")
	print(f"Total combinations: {len(PERSONAS) * len(SEEDS)}")
	print(f"Output file: {output_file}")
	print("=" * 60)
	print()

	# Format template
	content = SBATCH_TEMPLATE.format(
		logs_dir=LOGS_DIR,
		model=MODEL,
		model_dir=MODEL_DIR,
		reranker=RERANKER,
		output_dir=OUTPUT_DIR,
		output_filename=OUTPUT_FILENAME,
	)

	# Write file
	with open(output_file, "w") as f:
		f.write(content)

	# Make executable
	os.chmod(output_file, 0o755)

	print(f"✓ Generated: {output_file}")
	print()
	print("=" * 60)
	print("Next Steps")
	print("=" * 60)
	print()
	print(f"1. Submit job:")
	print(f"   sbatch {output_file}")
	print()
	print(f"2. Monitor progress:")
	print(f"   squeue -u $USER")
	print(f"   tail -f {LOGS_DIR}/explainability_persona_all_*.log")
	print()
	print(f"3. Check results (shared CSV):")
	print(f"   ls -lh {OUTPUT_DIR}/{OUTPUT_FILENAME}_all_results.csv")
	print(f"   # All 15 runs append to this file - no merge needed!")
	print()


if __name__ == "__main__":
	main()
