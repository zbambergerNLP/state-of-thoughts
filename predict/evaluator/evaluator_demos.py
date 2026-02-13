"""
Evaluator Demo Constants

This module contains demonstration examples for the TreeOfThoughtEvaluator
that show how to evaluate reasoning chains (PRM) and final solutions (ORM)
with both positive and negative examples across different quality levels.

These demos are structured to match the vllm_adapter expected format and
provide clear examples of the 1-7 Likert scale evaluation criteria.
"""

# Standard library imports
from typing import Any

# PRM Demos - Process Reward Model (Reasoning Chain Evaluation)
PRM_DEMOS: list[dict[str, Any]] = [
	{
		# Score 7/7: Excellent reasoning - systematic and thorough
		"input": {
			"math_problem": "If all managers attend meetings and Sarah attends all meetings, can we conclude Sarah is a manager?",
			"reasoning_steps": [
				"This is a logical reasoning problem about set relationships",
				"Let M = managers, A = people who attend meetings",
				"Given: All managers attend meetings (M ⊆ A) and Sarah attends meetings (Sarah ∈ A)",
				"We cannot conclude Sarah ∈ M because the relationship is one-way",
				"There could be non-managers who also attend meetings - the logic is invalid",
			],
		},
		"output": {
			"soundness": 7.0,
			"promise": 7.0,
		},
	},
	{
		# Score 6/5: Strong reasoning - correct with good explanation
		"input": {
			"math_problem": "Solve 3x + 7 = 22",
			"reasoning_steps": [
				"I need to isolate x by moving terms to opposite sides",
				"Subtract 7 from both sides: 3x + 7 - 7 = 22 - 7",
				"Simplify: 3x = 15",
				"Divide both sides by 3: x = 5",
			],
		},
		"output": {
			"soundness": 6.0,
			"promise": 5.0,
		},
	},
	{
		# Score 4/3: Average reasoning - adequate but could be clearer
		"input": {
			"math_problem": "Find the area of a parallelogram with base 6 and height 4",
			"reasoning_steps": [
				"parallelogram area uses base and height",
				"Area = base * height = (1/2) * 6 * 4 = 24",
			],
		},
		"output": {
			"soundness": 4.0,
			"promise": 3.0,
		},
	},
	{
		# Score 2/2: Weak reasoning - major mathematical error
		"input": {
			"math_problem": "What is 15% of 80?",
			"reasoning_steps": [
				"15% means 15 out of 100",
				"So I need to find 15 * 80 / 100",
				"15 * 80 = 1200, and 1200 / 100 = 120",
			],
		},
		"output": {
			"soundness": 2.0,
			"promise": 2.0,
		},
	},
	{
		# Score 1/1: Poor reasoning - false formulae
		"input": {
			"math_problem": "Find the volume of a cube of side length 2.",
			"reasoning_steps": [
				"The volume a cube is V = s^2",
				"V = 2^2 = 4",
			],
		},
		"output": {
			"soundness": 1.0,
			"promise": 1.0,
		},
	},
]

# ORM Demos - Outcome Reward Model (Final Solution Evaluation which includes reasoning steps)
ORM_DEMOS: list[dict[str, Any]] = [
	{
		# Score 7: Excellent solution - complete, clear, and insightful
		"input": {
			"math_problem": "A recipe calls for 2/3 cup of flour, but you want to make 1.5 times the recipe. How much flour do you need?",
			"reasoning_steps": [
				"I need to find 1.5 times the original amount of 2/3 cup",
				"Converting 1.5 to a fraction: 1.5 = 3/2",
				"Multiplying fractions: (3/2) * (2/3) = 6/6 = 1",
			],
			"answer": "To find 1.5 times 2/3 cup: (3/2) * (2/3) = 6/6 = 1 cup of flour needed",
		},
		"output": {"quality": 7.0},
	},
	{
		# Score 6: Very good solution - correct and well-presented
		"input": {
			"math_problem": "What is the slope of the line passing through (2, 5) and (8, 17)?",
			"reasoning_steps": [
				"I'll use the slope formula: m = (y_2 - y_1)/(x_2 - x_1)",
				"Identifying points: (x_1, y_1) = (2, 5) and (x_2, y_2) = (8, 17)",
				"Substituting: m = (17 - 5)/(8 - 2) = 12/6 = 2",
			],
			"answer": "Using the slope formula m = (y_2 - y_1)/(x_2 - x_1) = (17 - 5)/(8 - 2) = 12/6 = 2",
		},
		"output": {"quality": 6.0},
	},
	{
		# Score 4: Adequate solution - correct but basic presentation
		"input": {
			"math_problem": "Find the area of a circle with radius 3",
			"reasoning_steps": [
				"Using the formula for circle area",
				"Area = π * r² where r = 3",
			],
			"answer": "Area = π * 3² = 9π",
		},
		"output": {"quality": 4.0},
	},
	{
		# Score 3: Partially correct - right approach, calculation error
		"input": {
			"math_problem": "What is 25% of 120?",
			"reasoning_steps": ["25% = 25/100 = 0.25", "0.25 * 120 = 25"],
			"answer": "25% of 120 is 25",
		},
		"output": {"quality": 3.0},
	},
	{
		# Score 1: Incorrect solution - fundamental formula error
		"input": {
			"math_problem": "What is the circumference of a circle with radius 10?",
			"reasoning_steps": [
				"Using circumference formula for a circle",
				"Circumference = π * radius",
			],
			"answer": "Circumference = π * 10 = 10π",
		},
		"output": {"quality": 1.0},
	},
]

# Custom PRM Demos - For custom logical_rigor + step_clarity rubric (0-10 scale)
PRM_CUSTOM_DEMOS: list[dict[str, Any]] = [
	{
		# High quality reasoning - excellent logical rigor and step clarity
		"input": {
			"math_problem": "Solve for x: 2x + 5 = 13",
			"reasoning_steps": [
				"I need to isolate x by performing inverse operations on both sides",
				"First, subtract 5 from both sides to eliminate the constant term: 2x + 5 - 5 = 13 - 5",
				"This simplifies to: 2x = 8",
				"Next, divide both sides by 2 to isolate x: 2x / 2 = 8 / 2",
				"Therefore: x = 4",
			],
		},
		"output": {
			"logical_rigor": 9.0,
			"step_clarity": 10.0,
		},
	},
	{
		# Good quality reasoning - solid logic and clear steps
		"input": {
			"math_problem": "Find the area of a triangle with base 10 and height 6",
			"reasoning_steps": [
				"The area formula for a triangle is A = (1/2) * base * height",
				"Given: base = 10, height = 6",
				"Substituting into the formula: A = (1/2) * 10 * 6 = 30",
			],
		},
		"output": {
			"logical_rigor": 8.0,
			"step_clarity": 7.0,
		},
	},
]

# Custom ORM Demos - For custom solution_quality + presentation rubric (0-10 scale)
ORM_CUSTOM_DEMOS: list[dict[str, Any]] = [
	{
		# Excellent solution - high quality and presentation
		"input": {
			"math_problem": "If a train travels 180 miles in 3 hours, what is its average speed?",
			"reasoning_steps": [
				"Average speed is calculated using: speed = distance / time",
				"Given: distance = 180 miles, time = 3 hours",
				"Calculating: speed = 180 miles / 3 hours = 60 miles/hour",
			],
			"answer": "The train's average speed is 60 miles per hour (mph). Using the formula speed = distance/time with 180 miles traveled in 3 hours, we get 180/3 = 60 mph.",
		},
		"output": {
			"solution_quality": 10.0,
			"presentation": 9.0,
		},
	},
	{
		# Good solution - correct with solid presentation
		"input": {
			"math_problem": "What is 30% of 150?",
			"reasoning_steps": [
				"30% can be written as 0.30 or 30/100",
				"To find 30% of 150, multiply: 0.30 * 150",
				"0.30 * 150 = 45",
			],
			"answer": "30% of 150 equals 45",
		},
		"output": {
			"solution_quality": 8.0,
			"presentation": 7.0,
		},
	},
]
