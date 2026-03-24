#!/bin/bash
#SBATCH --job-name=mt_strict
#SBATCH --output=experiments/logs/argument_generation/meat_tax/synthesis_strict/mt_strict_%j.log
#SBATCH --error=experiments/logs/argument_generation/meat_tax/synthesis_strict/mt_strict_%j.log
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=40G
#SBATCH --partition=<your-partition>

# Environment Setup
source dspy_reasoning_env/bin/activate

echo "========================================="
echo "Meat Tax Experiment"
echo "Synthesis Type: synthesis_strict"
echo "========================================="
echo "Topic: A special tax should be imposed on meat products to reduce consumption and environmental impact."
echo "Stance: PRO"
echo "Seeds: 2000-2019"
echo "Total runs: 20"
echo "Tree params: top_k_first=50, top_k=250, num_final_candidates=250"
echo "Subtopics file: subtopics_specific_meat_tax.json"
echo "Output dir: experiments/argument_generation/argument_data/meat_tax/synthesis_strict"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Define seeds array
SEEDS=(2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019)

total=20
current=0

# Loop over seeds
for seed in "${SEEDS[@]}"; do
	((current++))

	echo "========================================="
	echo "[$current/$total] Seed: $seed"
	echo "Start time: $(date)"
	echo "========================================="

	# Run experiment (continue on error)
	python experiments/argument_generation/generate_arguments.py \
		--model Qwen3-30B-A3B-Instruct-2507 \
		--model_directory /path/to/models \
		--reranker_model Qwen3-Reranker-8B \
		--generative_gpu_index 0 \
		--reranker_gpu_index 1 \
		--outputs_directory experiments/argument_generation/argument_data/meat_tax/synthesis_strict \
		--outputs_filename meat_tax_synthesis_strict \
		--subtopics_file subtopics_specific_meat_tax.json \
		--topic "A special tax should be imposed on meat products to reduce consumption and environmental impact." \
		--stance "PRO" \
		--seed $seed \
		--depth 3 \
		--n_samples_generation 100 \
		--top_k_first 50 \
		--top_k 250 \
		--n_samples_judge 1 \
		--num_final_candidates 250 \
		--generator_temperature 0.7 \
		--experiment_mode synthesis_strict \
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
