"""
Default demonstration examples for the TreeOfThoughtGenerator generator.

Provides smart default demos for reasoning step generation and final answer generation.
"""

# Reasoning demos - for generating intermediate reasoning steps
REASONING_DEMOS = [
	{
		"input": {"math_problem": "What is 12 + 8?"},
		"reasoning": [
			{
				"internal_reasoning": "I need to add two numbers together",
				"math_operation": "12 + 8 equals 20",
			}
		],
		"output": {},
	},
	{
		"input": {"math_problem": "What is the capital of France?"},
		"reasoning": [
			{
				"internal_reasoning": "I should recall my knowledge of European capitals",
				"math_operation": "France is a country in Western Europe, and its capital city is Paris",
			}
		],
		"output": {},
	},
]

# Answer demos - for generating final answers after reasoning steps
ANSWER_DEMOS = [
	{
		"input": {"math_problem": "What is 5 + 3?"},
		"reasoning": [
			{
				"internal_reasoning": "I'll add these numbers step by step",
				"math_operation": "5 + 3 equals 8",
			}
		],
		"output": {"answer": "8"},
	},
	{
		"input": {"math_problem": "What is 2 * 6?"},
		"reasoning": [
			{
				"internal_reasoning": "I need to multiply these numbers",
				"math_operation": "2 multiplied by 6 equals 12",
			}
		],
		"output": {"answer": "12"},
	},
]
# The above are currently not used

MATH_DEMOS = [
	{
		"input": {"math_problem": "What is (10 - 3) * 2?"},
		"reasoning": [
			{
				"internal_reasoning": "I need to follow the order of operations",
				"math_operation": "First, I need to evaluate the expression in parentheses",
			},
			{
				"internal_reasoning": "Let me calculate the subtraction",
				"math_operation": "10 - 3 = 7",
			},
			{
				"internal_reasoning": "Now I can multiply the result",
				"math_operation": "Now multiply: 7 * 2 = 14",
			},
		],
		"output": {"answer": "14"},
	},
	{
		"input": {"math_problem": "What is 15 + 8 - 5?"},
		"reasoning": [
			{
				"internal_reasoning": "I should evaluate from left to right",
				"math_operation": "I'll work left to right",
			},
			{
				"internal_reasoning": "Let me add the first two numbers",
				"math_operation": "15 + 8 = 23",
			},
			{
				"internal_reasoning": "Now subtract the last number",
				"math_operation": "23 - 5 = 18",
			},
		],
		"output": {"answer": "18"},
	},
]

# Argument generation demos showing proper use of transition prefixes and internal reasoning
# These demos reflect actual tool usage: claims start with structure prefixes
# and internal_reasoning reflects guidance from subtopic and style choices.
ARGUMENT_DEMOS = [
	{
		"input": {"topic": "renewable energy", "stance": "PRO"},
		"reasoning": [
			{
				"internal_reasoning": "I should evaluate environmental consequences and sustainability factors. I should present logical arguments based on facts.",
				"claim": "Note that renewable energy significantly reduces carbon emissions compared to fossil fuels",
			},
			{
				"internal_reasoning": "I should analyze costs, benefits, market effects, and economic implications. I should present logical arguments based on facts.",
				"claim": "Therefore solar and wind power have become cost-competitive with traditional energy sources",
			},
			{
				"internal_reasoning": "I should analyze costs, benefits, market effects, and economic implications. I should speak with authority and unwavering confidence.",
				"claim": "Moreover energy independence through renewables improves national security",
			},
		],
		"output": {
			"argument": "Renewable energy is essential for our future. It provides a sustainable path forward that reduces emissions, costs less, and strengthens our independence."
		},
	},
	{
		"input": {"topic": "remote work", "stance": "PRO"},
		"reasoning": [
			{
				"internal_reasoning": "I should consider how this affects individuals, groups, and society as a whole. I should build credibility and establish mutual trust.",
				"claim": "First remote work eliminates commuting time and reduces stress",
			},
			{
				"internal_reasoning": "I should analyze costs, benefits, market effects, and economic implications. I should present logical arguments based on facts.",
				"claim": "On the other hand companies save significantly on office space and overhead costs",
			},
			{
				"internal_reasoning": "I should present logical arguments based on facts. I should emphasize information sharing and provide clear insights.",
				"claim": "For example employees report higher productivity when working from home",
			},
		],
		"output": {
			"argument": "Remote work benefits both employees and employers. Workers gain time and reduce stress, while companies cut costs and often see productivity gains."
		},
	},
	{
		"input": {"topic": "universal basic income", "stance": "PRO"},
		"reasoning": [
			{
				"internal_reasoning": "I should analyze costs, benefits, market effects, and economic implications. I should present logical arguments based on facts.",
				"claim": "Specifically UBI could reduce poverty by providing a guaranteed income floor for all citizens",
			},
			{
				"internal_reasoning": "I should consider how this affects individuals, groups, and society as a whole. I should build credibility and establish mutual trust.",
				"claim": "Similarly UBI would improve economic security for vulnerable populations",
			},
			{
				"internal_reasoning": "I should analyze costs, benefits, market effects, and economic implications. I should speak with authority and unwavering confidence.",
				"claim": "Essentially UBI represents a fundamental shift toward economic justice",
			},
		],
		"output": {
			"argument": "Universal Basic Income offers a transformative approach to economic security. It reduces poverty, improves security for vulnerable groups, and represents a fundamental shift toward economic justice."
		},
	},
	{
		"input": {"topic": "AI regulation", "stance": "PRO"},
		"reasoning": [
			{
				"internal_reasoning": "I should analyze legal requirements, regulations, and compliance considerations. I should express strong disagreement.",
				"claim": "If AI systems perpetuate discrimination through biased algorithms, regulatory oversight is needed to ensure fairness",
			},
			{
				"internal_reasoning": "I should examine technical requirements, limitations, and implementation challenges. I should speak with authority and unwavering confidence.",
				"claim": "For the sake of preventing harm, government must establish clear safety standards for AI development",
			},
			{
				"internal_reasoning": "I should analyze legal requirements, regulations, and compliance considerations. I should present logical arguments based on facts.",
				"claim": "However existing frameworks are insufficient to address rapidly evolving AI capabilities",
			},
		],
		"output": {
			"argument": "AI regulation is urgently needed. If AI systems perpetuate discrimination, oversight is required. For the sake of preventing harm, government must establish safety standards. However existing frameworks are insufficient for rapidly evolving AI capabilities."
		},
	},
]
