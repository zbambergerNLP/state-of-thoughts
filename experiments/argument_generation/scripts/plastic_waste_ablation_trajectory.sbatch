#!/bin/bash
#SBATCH --job-name=ablation_traj
#SBATCH --output=experiments/logs/argument_generation/explainability/ablation/ablation_traj_%j.log
#SBATCH --error=experiments/logs/argument_generation/explainability/ablation/ablation_traj_%j.log
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=40G
#SBATCH --partition=<your-partition>

# Environment Setup
source dspy_reasoning_env/bin/activate

# Create log directory if it doesn't exist
mkdir -p experiments/logs/argument_generation/explainability/ablation

echo "========================================="
echo "Ablation Trajectory Selection Experiment"
echo "========================================="
echo "Topic: The government should enforce a total ban on single-use plastics."
echo "Stance: PRO"
echo "Subtopics: specific (subtopics_specific_single_use_plastic.json)"
echo "Synthesis types: strict, faithful, restructured"
echo "Selection modes: targeted, random, m1b"
echo "Total experiments: 9 (3 synthesis types x 3 selection modes)"
echo "Trajectories per experiment: 50"
echo "Samples per trajectory: 5"
echo "Total arguments: 2,250"
echo "Start time: $(date)"
echo "========================================="
echo ""

python experiments/argument_generation/ablation_trajectory_selection.py \
    --synthesis_type all \
    --selection_mode all \
    --top_n 50 \
    --samples_per_trajectory 5 \
    --seed 42 \
    --depth 3 \
    --subtopics_file subtopics_specific_single_use_plastic.json \
    --model Qwen3-30B-A3B-Instruct-2507 \
    --reranker_model Qwen3-Reranker-8B \
    --model_directory /path/to/models \
    --generative_gpu_index 0 \
    --reranker_gpu_index 1 \
    --generator_temperature 0.7

echo ""
echo "========================================="
echo "Ablation experiment completed!"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "Output files:"
echo "  synthesis_strict/targeted_generation/targeted_forced_results_strict.csv"
echo "  synthesis_strict/targeted_generation/random_forced_results_strict.csv"
echo "  synthesis_strict/targeted_generation/m1b_forced_results_strict.csv"
echo "  synthesis_faithful/targeted_generation/targeted_forced_results_faithful.csv"
echo "  synthesis_faithful/targeted_generation/random_forced_results_faithful.csv"
echo "  synthesis_faithful/targeted_generation/m1b_forced_results_faithful.csv"
echo "  synthesis_restructured/targeted_generation/targeted_forced_results_restructured.csv"
echo "  synthesis_restructured/targeted_generation/random_forced_results_restructured.csv"
echo "  synthesis_restructured/targeted_generation/m1b_forced_results_restructured.csv"
echo "========================================="
