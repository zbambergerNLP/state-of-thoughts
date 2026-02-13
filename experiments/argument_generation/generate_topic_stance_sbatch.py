"""
Generate single SBATCH file for topic-stance sweep experiment.

Creates 1 sbatch file that runs all 20 combinations (10 topics × 2 stances) sequentially.
Models loaded once and shared across all runs.

Configuration:
- Topics: 10 policy topics
- Stances: PRO, ANTI
- Seed: 42 (fixed)
- Model: Qwen3-30B-A3B-Instruct-2507
- Reranker: Qwen3-Reranker-8B
- Resources: 2 GPUs, 16 CPUs, 40GB/GPU, 3 hours
- Output: Single shared CSV file (all runs append)

Usage:
	python experiments/argument_generation/generate_topic_stance_sbatch.py

Output:
	Single .sbatch file: experiments/argument_generation/scripts/topic_stance_sweep.sbatch
"""

import os
from pathlib import Path

# Experiment configuration
TOPICS = [
	"The government should implement a Universal Basic Income (UBI) for all citizens.",
	"Employees should have the legal right to work remotely if their job allows it.",
	"The government should enforce a total ban on single-use plastics.",
	"Standardized testing should be abolished as a primary measure of student performance.",
	"The government should invest heavily in nuclear energy as a primary power source.",
	"Access to social media should be restricted to individuals over the age of 16.",
	"A special tax should be imposed on meat products to reduce consumption and environmental impact.",
	"The government should phase out physical currency in favor of a fully digital payment system.",
	"Public funding for space exploration should be significantly increased.",
	"Voting in national elections should be mandatory for all eligible citizens.",
]

STANCES = ["PRO", "ANTI"]
SEED = 42

# Directory configuration
SCRIPTS_DIR = "experiments/argument_generation/scripts"
LOGS_DIR = "experiments/logs/argument_generation/topic_stance_sweep"

# Model configuration
MODEL = "Qwen3-30B-A3B-Instruct-2507"
RERANKER = "Qwen3-Reranker-8B"
MODEL_DIR = "/projects/BSTEWART/model_storage"
OUTPUT_DIR = "experiments/argument_generation/topic_stance_sweep"
OUTPUT_FILENAME = "topic_stance_sweep"

# SBATCH template with bash loops for all combinations
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=topic_stance_all
#SBATCH --output={logs_dir}/topic_stance_all_%j.log
#SBATCH --error={logs_dir}/topic_stance_all_%j.log
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
echo "Topic-Stance Sweep Experiment (Batch)"
echo "========================================="
echo "Total combinations: 20 (10 topics × 2 stances)"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Define parameter arrays
TOPICS=(
	"The government should implement a Universal Basic Income (UBI) for all citizens."
	"Employees should have the legal right to work remotely if their job allows it."
	"The government should enforce a total ban on single-use plastics."
	"Standardized testing should be abolished as a primary measure of student performance."
	"The government should invest heavily in nuclear energy as a primary power source."
	"Access to social media should be restricted to individuals over the age of 16."
	"A special tax should be imposed on meat products to reduce consumption and environmental impact."
	"The government should phase out physical currency in favor of a fully digital payment system."
	"Public funding for space exploration should be significantly increased."
	"Voting in national elections should be mandatory for all eligible citizens."
)

STANCES=(PRO ANTI)

total=20
current=0

# Nested loops
for topic_idx in "${{!TOPICS[@]}}"; do
	topic="${{TOPICS[$topic_idx]}}"
	topic_num=$((topic_idx + 1))

	for stance in "${{STANCES[@]}}"; do
		((current++))

		echo "========================================="
		echo "[$current/$total] Topic $topic_num: ${{topic:0:60}}..."
		echo "Stance: $stance"
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
			--topic "$topic" \\
			--stance "$stance" \\
			--seed {seed} \\
			--depth 3 \\
			--n_samples_generation 50 \\
			--top_k 50 \\
			--n_samples_judge 1 \\
			--num_final_candidates 100 \\
			--generator_temperature 0.7 \\
			--experiment_mode synthesis_restructured \\
			--do_save_tree || echo "ERROR: Failed for topic $topic_num, stance=$stance (continuing...)"

		echo ""
		echo "Completed: topic $topic_num, stance=$stance"
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
	"""Generate single consolidated SBATCH file for all topic × stance combinations."""
	# Create scripts directory
	Path(SCRIPTS_DIR).mkdir(parents=True, exist_ok=True)

	# Single output file
	output_file = f"{SCRIPTS_DIR}/topic_stance_sweep.sbatch"

	print("=" * 60)
	print("Topic-Stance Sweep SBATCH Generator")
	print("=" * 60)
	print(f"Topics: {len(TOPICS)}")
	print(f"Stances: {len(STANCES)} ({', '.join(STANCES)})")
	print(f"Total combinations: {len(TOPICS) * len(STANCES)}")
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
		seed=SEED,
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
	print(f"   tail -f {LOGS_DIR}/topic_stance_all_*.log")
	print()
	print(f"3. Check results (shared CSV):")
	print(f"   ls -lh {OUTPUT_DIR}/{OUTPUT_FILENAME}_all_results.csv")
	print(f"   # All 20 runs append to this file - no merge needed!")
	print()


if __name__ == "__main__":
	main()
