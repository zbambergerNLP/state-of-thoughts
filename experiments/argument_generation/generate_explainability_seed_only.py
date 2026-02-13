"""
Generate SBATCH files for seed-only explainability experiment with multiple synthesis types.

Creates 3 sbatch files (one per synthesis type) that each run 20 experiments with different
random seeds. Uses a single topic (plastics ban) and single stance (PRO) to isolate seed effects.
Models loaded once and shared across all runs within each job.

Configuration:
- Topic: Single-use plastics ban (fixed)
- Stance: PRO (fixed)
- Seeds: 2000-2019 (20 different seeds)
- Synthesis types: strict, faithful, restructured (3 separate jobs)
- Tree parameters: top_k_first=50, top_k=250, num_final_candidates=250
- Model: Qwen3-30B-A3B-Instruct-2507
- Reranker: Qwen3-Reranker-8B
- Resources: 2 GPUs, 16 CPUs, 40GB/GPU, 6 hours
- Output: Separate directories per synthesis type

Usage:
	python experiments/argument_generation/generate_explainability_seed_only.py

Output:
	3 .sbatch files:
	- experiments/argument_generation/scripts/explainability_synthesis_strict.sbatch
	- experiments/argument_generation/scripts/explainability_synthesis_faithful.sbatch
	- experiments/argument_generation/scripts/explainability_synthesis_restructured.sbatch
"""

import os
from pathlib import Path

# Experiment configuration
TOPIC = "The government should enforce a total ban on single-use plastics."
STANCE = "PRO"
SEEDS = list(range(2000, 2020))  # Seeds 2000-2019 (20 seeds)

# Synthesis types to generate
SYNTHESIS_TYPES = ["synthesis_strict", "synthesis_faithful", "synthesis_restructured"]

# Tree parameters
TOP_K_FIRST = 50
TOP_K = 250
NUM_FINAL_CANDIDATES = 250
N_SAMPLES_GENERATION = 100  # Keep same
DEPTH = 3

# Directory configuration
SCRIPTS_DIR = "experiments/argument_generation/scripts"
LOGS_BASE_DIR = "experiments/logs/argument_generation/explainability"
OUTPUT_BASE_DIR = "experiments/argument_generation/explainability"

# Model configuration
MODEL = "Qwen3-30B-A3B-Instruct-2507"
RERANKER = "Qwen3-Reranker-8B"
MODEL_DIR = "/projects/BSTEWART/model_storage"

# SBATCH template with bash loop for all seeds
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=expl_{synthesis_short}
#SBATCH --output={logs_dir}/expl_{synthesis_short}_%j.log
#SBATCH --error={logs_dir}/expl_{synthesis_short}_%j.log
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=40G
#SBATCH --partition=ailab

# Environment Setup
source dspy_reasoning_env/bin/activate

echo "========================================="
echo "Seed-Only Explainability Experiment"
echo "Synthesis Type: {synthesis_type}"
echo "========================================="
echo "Topic: {topic}"
echo "Stance: {stance}"
echo "Seeds: 2000-2019"
echo "Total runs: 20"
echo "Tree params: top_k_first={top_k_first}, top_k={top_k}, num_final_candidates={num_final_candidates}"
echo "Output dir: {output_dir}"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Define seeds array
SEEDS=({seeds_str})

total=20
current=0

# Loop over seeds
for seed in "${{SEEDS[@]}}"; do
	((current++))

	echo "========================================="
	echo "[$current/$total] Seed: $seed"
	echo "Start time: $(date)"
	echo "========================================="

	# Run experiment (continue on error)
	python experiments/argument_generation/topic_stance_sweep.py \\
		--model {model} \\
		--model_directory {model_dir} \\
		--reranker_model {reranker} \\
		--generative_gpu_index 0 \\
		--reranker_gpu_index 1 \\
		--outputs_directory {output_dir} \\
		--outputs_filename {output_filename} \\
		--topic "{topic}" \\
		--stance "{stance}" \\
		--seed $seed \\
		--depth {depth} \\
		--n_samples_generation {n_samples_generation} \\
		--top_k_first {top_k_first} \\
		--top_k {top_k} \\
		--n_samples_judge 1 \\
		--num_final_candidates {num_final_candidates} \\
		--generator_temperature 0.7 \\
		--experiment_mode {synthesis_type} \\
		--do_save_tree || echo "ERROR: Failed for seed=$seed (continuing...)"

	echo ""
	echo "Completed: seed=$seed"
	echo "End time: $(date)"
	echo "Progress: $current/$total"
	echo "========================================="
	echo ""
done

echo "========================================="
echo "All jobs completed!"
echo "End time: $(date)"
echo "========================================="
"""


def main():
	"""Generate SBATCH files for all synthesis type experiments."""
	# Create scripts directory
	Path(SCRIPTS_DIR).mkdir(parents=True, exist_ok=True)

	print("=" * 60)
	print("Seed-Only Explainability SBATCH Generator")
	print("=" * 60)
	print(f"Topic: {TOPIC[:60]}...")
	print(f"Stance: {STANCE}")
	print(f"Seeds: {len(SEEDS)} ({min(SEEDS)}-{max(SEEDS)})")
	print(f"Synthesis types: {', '.join(SYNTHESIS_TYPES)}")
	print(f"Tree params: top_k_first={TOP_K_FIRST}, top_k={TOP_K}, num_final_candidates={NUM_FINAL_CANDIDATES}")
	print(f"Total runs per type: {len(SEEDS)}")
	print(f"Expected arguments per type: {NUM_FINAL_CANDIDATES} x {len(SEEDS)} = {NUM_FINAL_CANDIDATES * len(SEEDS)}")
	print("=" * 60)
	print()

	# Format seeds as space-separated string for bash array
	seeds_str = " ".join(str(s) for s in SEEDS)

	generated_files = []

	for synthesis_type in SYNTHESIS_TYPES:
		# Create short name for job name (e.g., "strict" from "synthesis_strict")
		synthesis_short = synthesis_type.replace("synthesis_", "")

		# Output paths for this synthesis type
		output_dir = f"{OUTPUT_BASE_DIR}/{synthesis_type}"
		logs_dir = f"{LOGS_BASE_DIR}/{synthesis_type}"
		output_filename = f"explainability_{synthesis_type}"
		output_file = f"{SCRIPTS_DIR}/explainability_{synthesis_type}.sbatch"

		# Create logs directory
		Path(logs_dir).mkdir(parents=True, exist_ok=True)

		# Format template
		content = SBATCH_TEMPLATE.format(
			logs_dir=logs_dir,
			model=MODEL,
			model_dir=MODEL_DIR,
			reranker=RERANKER,
			output_dir=output_dir,
			output_filename=output_filename,
			topic=TOPIC,
			stance=STANCE,
			seeds_str=seeds_str,
			synthesis_type=synthesis_type,
			synthesis_short=synthesis_short,
			top_k_first=TOP_K_FIRST,
			top_k=TOP_K,
			num_final_candidates=NUM_FINAL_CANDIDATES,
			n_samples_generation=N_SAMPLES_GENERATION,
			depth=DEPTH,
		)

		# Write file
		with open(output_file, "w") as f:
			f.write(content)

		# Make executable
		os.chmod(output_file, 0o755)

		generated_files.append(output_file)
		print(f"Generated: {output_file}")

	print()
	print("=" * 60)
	print("Next Steps")
	print("=" * 60)
	print()
	print("1. Submit all jobs:")
	for f in generated_files:
		print(f"   sbatch {f}")
	print()
	print("2. Monitor progress:")
	print("   squeue -u $USER")
	for synthesis_type in SYNTHESIS_TYPES:
		synthesis_short = synthesis_type.replace("synthesis_", "")
		print(f"   tail -f {LOGS_BASE_DIR}/{synthesis_type}/expl_{synthesis_short}_*.log")
	print()
	print("3. Check results (one CSV per synthesis type):")
	for synthesis_type in SYNTHESIS_TYPES:
		print(f"   ls -lh {OUTPUT_BASE_DIR}/{synthesis_type}/explainability_{synthesis_type}_all_results.csv")
	print()
	print(f"4. Expected output: {NUM_FINAL_CANDIDATES * len(SEEDS)} arguments per synthesis type")
	print(f"   Total across all 3 types: {NUM_FINAL_CANDIDATES * len(SEEDS) * len(SYNTHESIS_TYPES)} arguments")
	print()


if __name__ == "__main__":
	main()
