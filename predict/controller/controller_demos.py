"""
Controller Demo Constants

This module contains demonstration examples for the TreeOfThoughtsController
that show when to use FINISH vs CONTINUE_REASONING tools based on the completeness
of the reasoning process.

These demos are structured to match the controller's signature format and guide
the model to make appropriate decisions about reasoning completion.
"""

# Standard library imports
from typing import Any

# Demos showing when to "finish" reasoning (reasoning is sufficient)
CONTROLLER_FINISH_DEMOS: list[dict[str, Any]] = [
	{
		# Complete arithmetic solution
		"input": {
			"math_problem": "What is 15 * 8?",
			"reasoning": (
				"""
<thinking>
<step>
## math_operation
I need to multiply 15 by 8
</step>
<step>
## math_operation
15 * 8 = 120
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 3,
		},
		"output": {
			"considerations": "The multiplication has been completed correctly and the answer is clear. No further reasoning is needed.",
			"action": "finish",
		},
	},
	{
		# Complete linear equation solution
		"input": {
			"math_problem": "Solve 2x + 3 = 11",
			"reasoning": (
				"""
<thinking>
<step>
## math_operation
I need to isolate x by subtracting 3 from both sides.
</step>
<step>
## math_operation
2x + 3 - 3 = 11 - 3
</step>
<step>
## math_operation
2x = 8
</step>
<step>
## math_operation
Dividing both sides by 2: x = 4
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "The equation has been completely solved. I found x = 4 and verified the steps. The solution is complete.",
			"action": "finish",
		},
	},
	{
		# Complete geometry calculation
		"input": {
			"math_problem": "What is the area of a circle with radius 4?",
			"reasoning": (
				"""
<thinking>
<step>
## math_operation
The formula for the area of a circle is A = πr².
</step>
<step>
## math_operation
Given r = 4, so A = π * 4² = π * 16 = 16π.
</step>
<step>
## math_operation
The area is 16π square units.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I have correctly applied the area formula and calculated the result. The answer is complete and clearly stated.",
			"action": "finish",
		},
	},
]

# Demos showing when to "continue_reasoning" (more reasoning needed)
CONTROLLER_CONTINUE_DEMOS: list[dict[str, Any]] = [
	{
		# Problem identified but not solved
		"input": {
			"math_problem": "Solve x^3 - 10x + 5 = 0",
			"reasoning": (
				"""
<thinking>
<step>
## math_operation
This is a cubic equation in standard form.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 4,
		},
		"output": {
			"considerations": "I've identified this as a cubic equation, but I haven't chosen a solution method or begun solving it. I need to continue reasoning to find the solution.",
			"action": "continue_reasoning",
		},
	},
	{
		# Method identified but not applied
		"input": {
			"math_problem": "Find the derivative of f(x) = x³ + 2x² - x + 1",
			"reasoning": (
				"""
<thinking>
<step>
## math_operation
I need to find the derivative using the power rule.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 3,
		},
		"output": {
			"considerations": "I know I need to use the power rule, but I haven't actually applied it to each term yet. More reasoning steps are needed to compute the derivative.",
			"action": "continue_reasoning",
		},
	},
	{
		# Partial geometric proof
		"input": {
			"math_problem": "Prove that the sum of angles in any triangle is 180°",
			"reasoning": (
				"""
<thinking>
<step>
## math_operation
I need to prove this geometric theorem.
</step>
<step>
## math_operation
Let me consider a triangle ABC with angles α, β, and γ.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 4,
		},
		"output": {
			"considerations": "I've set up the proof with a triangle ABC, but I haven't provided the actual proof steps. I need to continue with the geometric reasoning.",
			"action": "continue_reasoning",
		},
	},
	{
		# Incomplete analysis
		"input": {
			"math_problem": "Is 91 a prime number?",
			"reasoning": (
				"""
<thinking>
<step>
## math_operation
A prime number has exactly two factors: 1 and itself.
</step>
<step>
## math_operation
I need to check if 91 has any factors other than 1 and 91.
</step>
<step>
## math_operation
I should test divisibility by primes up to √91 ≈ 9.5.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 3,
		},
		"output": {
			"considerations": "I've outlined the approach for checking primality, but I haven't actually tested the divisibility. I need to continue reasoning to check each potential factor.",
			"action": "continue_reasoning",
		},
	},
	{
		# Incomplete factoring
		"input": {
			"math_problem": "Factor x^2 - 9",
			"reasoning": (
				"""
<thinking>
<step>
## math_operation
This looks like a difference of squares pattern.
</step>
<step>
## math_operation
The difference of squares formula is a^2 - b^2 = (a + b)(a - b).
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've identified this as a difference of squares and recalled the formula, but I haven't applied it to actually factor x² - 9.",
			"action": "continue_reasoning",
		},
	},
	{
		# Partial FOIL expansion
		"input": {
			"math_problem": "Expand (x + 2)(x - 3)",
			"reasoning": (
				"""
<thinking>
<step>
## math_operation
I'll use the FOIL method.
</step>
<step>
## math_operation
First: x * x = x^2
</step>
<step>
## math_operation
Outer: x * (-3) = -3x
</step>
<step>
## math_operation
Inner: 2 * x = 2x
</step>
<step>
## math_operation
Last: 2 * (-3) = -6
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've computed all the FOIL terms, but I haven't combined them into the final expanded form. I need to add the terms together.",
			"action": "continue_reasoning",
		},
	},
]


# Argument-specific demos showing diverse subtopic, style, and structure choices
ARGUMENT_CONTROLLER_DEMOS: list[dict[str, Any]] = [
	{
		# Economic Impact + Knowledge + Level-of-Detail - opening with specifics
		# First step uses "Specifically" to dive into economic details
		"input": {
			"topic": "UBI should be implemented nationally",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Specifically UBI could reduce poverty by providing a guaranteed income floor for all citizens.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 3,
		},
		"output": {
			"considerations": "I've made one economic point, but I should analyze more specific economic impacts like costs, funding mechanisms, and market effects to build a compelling argument. I should present logical arguments based on facts.",
			"action": "select_subtopic_style_structure",
			"action_arguments": {
				"subtopic": "Economic Impact",
				"style": "Knowledge",
				"structure": "Cause",
			},
		},
	},
	{
		# Social Impact + Trust + Contrast - mid-reasoning example
		# Previous step established a point, now using "Cause" to show consequence, then "Contrast" for next step
		"input": {
			"topic": "Tech companies should face stricter data protection rules",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
First current privacy laws are insufficient for protecting user data.
</step>
<step>
## claim
Therefore companies currently collect vast amounts of personal information with minimal oversight.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've established the problem and its consequence, but need to explore how stricter regulations would affect people and communities. I should build credibility and establish mutual trust. A contrasting perspective would strengthen the argument.",
			"action": "select_subtopic_style_structure",
			"action_arguments": {
				"subtopic": "Social Impact",
				"style": "Trust",
				"structure": "Contrast",
			},
		},
	},
	{
		# Legal & Regulatory + Power + Instantiation - showing completion
		# Previous steps show proper prefixes and internal reasoning
		"input": {
			"topic": "Self-driving cars should require federal certification",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Generally current state-by-state approaches create regulatory fragmentation.
</step>
<step>
## claim
On the other hand federal oversight would establish uniform safety standards across all jurisdictions.
</step>
<step>
## claim
For example the National Highway Traffic Safety Administration could mandate standardized testing protocols for all autonomous vehicles.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 1,
		},
		"output": {
			"considerations": "I've covered regulatory analysis with concrete examples and built a strong authoritative argument. The reasoning is comprehensive enough to finish.",
			"action": "finish",
			"action_arguments": {},
		},
	},
	{
		# Technical Feasibility + Power + Purpose - diverse combination
		# Should show "For the sake of" prefix from Purpose structure
		"input": {
			"topic": "Government should mandate solar panels on new buildings",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Earlier renewable energy infrastructure was seen as optional, but climate change demands immediate action.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 4,
		},
		"output": {
			"considerations": "I've established the temporal context, but need to examine technical requirements and implementation challenges while maintaining an authoritative tone. I should focus on the purpose behind this policy.",
			"action": "select_subtopic_style_structure",
			"action_arguments": {
				"subtopic": "Technical Feasibility",
				"style": "Power",
				"structure": "Purpose",
			},
		},
	},
	{
		# Environmental Impact + Support + Similarity - showing empathy
		# Should show "Similarly" prefix from Similarity structure
		"input": {
			"topic": "Businesses should eliminate single-use plastics",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
At the same time plastic pollution threatens marine ecosystems and human health.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 3,
		},
		"output": {
			"considerations": "I've established the environmental threat, but should evaluate environmental consequences and sustainability factors while being emotionally supportive. I should emphasize our shared concern for the planet.",
			"action": "select_subtopic_style_structure",
			"action_arguments": {
				"subtopic": "Environmental Impact",
				"style": "Support",
				"structure": "Similarity",
			},
		},
	},
	{
		# Legal & Regulatory + Conflict + Concession - confrontational approach
		# Should show "However" prefix from Concession structure
		"input": {
			"topic": "Platforms should be held liable for user-generated content",
			"stance": "ANTI",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Essentially free speech principles are fundamental to democratic discourse.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've restated the core principle, but need to analyze legal requirements and regulations while taking a more confrontational stance. I should acknowledge counterarguments.",
			"action": "select_subtopic_style_structure",
			"action_arguments": {
				"subtopic": "Legal & Regulatory",
				"style": "Conflict",
				"structure": "Concession",
			},
		},
	},
	{
		# Economic Impact + Knowledge + Conjunction - adding information
		# Previous step established a point, now adding more information with "Moreover"
		"input": {
			"topic": "The government should regulate AI development",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
If AI systems perpetuate discrimination through biased algorithms, regulatory oversight is needed to ensure fairness.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've established a conditional relationship, but should analyze more economic impacts like costs, benefits, and market effects. I should present logical arguments based on facts and add more information to strengthen the argument.",
			"action": "select_subtopic_style_structure",
			"action_arguments": {
				"subtopic": "Economic Impact",
				"style": "Knowledge",
				"structure": "Conjunction",
			},
		},
	},
	{
		# Social Impact + Trust + Instantiation - providing examples
		# Should show "For example" prefix from Instantiation (Specific) structure
		"input": {
			"topic": "Universal healthcare should be implemented",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Rather than maintaining fragmented insurance systems, universal healthcare would improve access to medical services for underserved populations.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've presented an alternative approach, but should consider how this affects people and communities while building credibility. I should provide concrete examples to illustrate the point.",
			"action": "select_subtopic_style_structure",
			"action_arguments": {
				"subtopic": "Social Impact",
				"style": "Trust",
				"structure": "Instantiation (Specific)",
			},
		},
	},
]

# Style-only demos (for generative controllers with style action space + finish)
# Uses intervene_on_next_reasoning_step action with style parameter
STYLE_CONTROLLER_DEMOS: list[dict[str, Any]] = [
	{
		# Statistical context should encourage style variation
		"input": {
			"topic": "Climate change mitigation",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Research shows that renewable energy costs have dropped 89% since 2010.
</step>
<step>
## claim
Studies indicate solar power is now cheaper than fossil fuels in most markets.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've presented statistical evidence, but my argument could benefit from a more engaging, figurative approach to connect emotionally with readers.",
			"action": "intervene_on_next_reasoning_step",
			"action_arguments": {
				"style": "Figurative Language",
			},
		},
	},
	{
		# Figurative language should encourage statistical balance
		"input": {
			"topic": "Universal healthcare",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Healthcare is like oxygen - everyone needs it to survive and thrive.
</step>
<step>
## claim
A healthy population is the foundation upon which all other progress is built.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've used figurative language effectively, but now I need concrete data and statistics to strengthen my argument with evidence.",
			"action": "intervene_on_next_reasoning_step",
			"action_arguments": {
				"style": "Statistical & Data-Driven",
			},
		},
	},
	{
		# Complete argument should finish
		"input": {
			"topic": "Education funding",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Education is an investment in our collective future.
</step>
<step>
## claim
Studies show that every dollar invested in education returns $7-10 to the economy.
</step>
<step>
## claim
Therefore, increased education funding is both morally right and economically sound.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 1,
		},
		"output": {
			"considerations": "The argument is complete with both figurative appeal and statistical evidence, concluding with a clear thesis. No further reasoning is needed.",
			"action": "finish",
			"action_arguments": {},
		},
	},
]


# Structure-only demos (for generative controllers with structure action space + finish)
# Uses intervene_on_next_reasoning_step action with structure parameter
STRUCTURE_CONTROLLER_DEMOS: list[dict[str, Any]] = [
	{
		# Claim needs causal evidence
		"input": {
			"topic": "Electric vehicle adoption",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Electric vehicles are the future of transportation.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 3,
		},
		"output": {
			"considerations": "I've made a claim but haven't explained why EVs are the future. I need to add causal reasoning to explain the underlying factors.",
			"action": "intervene_on_next_reasoning_step",
			"action_arguments": {
				"structure": "Causal Reasoning",
			},
		},
	},
	{
		# After claim and cause, add contrast
		"input": {
			"topic": "Remote work policies",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Remote work increases employee productivity.
</step>
<step>
## claim
Because workers save commute time and have more control over their environment, they are more focused and efficient.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've established a claim with causal reasoning, but I should acknowledge the contrasting perspective to strengthen my argument.",
			"action": "intervene_on_next_reasoning_step",
			"action_arguments": {
				"structure": "Contrast",
			},
		},
	},
	{
		# Complete structural argument should finish
		"input": {
			"topic": "Public transit investment",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Cities should prioritize public transit expansion.
</step>
<step>
## claim
Because public transit reduces traffic congestion and emissions, it improves quality of life for all residents.
</step>
<step>
## claim
On the other hand, while cars offer flexibility, their societal costs outweigh individual benefits.
</step>
<step>
## claim
Therefore, investing in public transit is the responsible choice for sustainable urban development.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 1,
		},
		"output": {
			"considerations": "The argument has a clear claim, causal reasoning, contrast, and conclusion. The structure is complete and ready to finish.",
			"action": "finish",
			"action_arguments": {},
		},
	},
]


# Style + Structure combined demos (for generative controllers with both action spaces)
# Uses intervene_on_next_reasoning_step action with both style and structure parameters
STYLE_STRUCTURE_CONTROLLER_DEMOS: list[dict[str, Any]] = [
	{
		# Early stage - needs both style and structure
		"input": {
			"topic": "Renewable energy investment",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
We should increase investment in renewable energy.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 3,
		},
		"output": {
			"considerations": "I have only a basic claim. I need to add substance with evidence. Using figurative language with causal reasoning will make the argument both engaging and logical.",
			"action": "intervene_on_next_reasoning_step",
			"action_arguments": {
				"style": "Figurative Language",
				"structure": "Causal Reasoning",
			},
		},
	},
	{
		# After figurative+causal, balance with statistical+contrast
		"input": {
			"topic": "AI regulation",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
AI systems must be regulated to prevent harm.
</step>
<step>
## claim
Like a powerful river, AI needs proper channels to direct its force productively rather than destructively.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've used figurative language. Now I should balance with data and address counterarguments through statistical evidence and contrasting views.",
			"action": "intervene_on_next_reasoning_step",
			"action_arguments": {
				"style": "Statistical & Data-Driven",
				"structure": "Contrast",
			},
		},
	},
	{
		# Complete argument ready to finish
		"input": {
			"topic": "Carbon tax implementation",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
A carbon tax is essential for addressing climate change.
</step>
<step>
## claim
Just as we price harmful substances like tobacco to reduce consumption, pricing carbon emissions creates natural market incentives.
</step>
<step>
## claim
Studies show that countries with carbon taxes have reduced emissions by 15-20% compared to those without.
</step>
<step>
## claim
While critics argue costs hurt businesses, evidence from British Columbia shows economic growth remained strong after carbon tax implementation.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 1,
		},
		"output": {
			"considerations": "The argument includes figurative language, statistical evidence, causal reasoning, and contrast. It is comprehensive and complete.",
			"action": "finish",
			"action_arguments": {},
		},
	},
]


# Combined demo set for controller (continue/finish only - math demos)
CONTROLLER_DEMOS: list[dict[str, Any]] = (
	CONTROLLER_FINISH_DEMOS + CONTROLLER_CONTINUE_DEMOS
)


# Argument-based continue/finish only demos (for controllers with no action spaces)
# Uses ArgumentField but only continue_reasoning and finish actions
ARGUMENT_CONTINUE_FINISH_DEMOS: list[dict[str, Any]] = [
	{
		# Demo #1: Incomplete argument - needs more reasoning
		"input": {
			"topic": "Universal basic income",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
UBI could help reduce poverty rates.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 3,
		},
		"output": {
			"considerations": "I've only made one claim without supporting evidence or explanation. I need to continue reasoning to develop this argument further.",
			"action": "continue_reasoning",
		},
	},
	{
		# Demo #2: Still needs more content
		"input": {
			"topic": "Electric vehicle subsidies",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Government subsidies for EVs can accelerate the transition to clean transportation.
</step>
<step>
## claim
These subsidies reduce the upfront cost barrier that prevents many consumers from choosing EVs.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 2,
		},
		"output": {
			"considerations": "I've provided some reasoning, but the argument lacks a conclusion and could benefit from addressing counterarguments. More reasoning steps are available.",
			"action": "continue_reasoning",
		},
	},
	{
		# Demo #3: Complete argument - ready to finish
		"input": {
			"topic": "Renewable energy investment",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Investing in renewable energy is essential for long-term economic stability.
</step>
<step>
## claim
Fossil fuel prices are volatile and subject to geopolitical instability, while renewable sources provide predictable costs.
</step>
<step>
## claim
Countries that have invested heavily in renewables, like Denmark and Germany, have created millions of green jobs.
</step>
<step>
## claim
Therefore, transitioning to renewable energy is both economically prudent and environmentally necessary.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 1,
		},
		"output": {
			"considerations": "The argument presents a clear claim, supporting evidence with examples, and a conclusive statement. It is complete and ready to finish.",
			"action": "finish",
		},
	},
	{
		# Demo #4: Another complete argument
		"input": {
			"topic": "Public education funding",
			"stance": "PRO",
			"reasoning": (
				"""
<thinking>
<step>
## claim
Increased public education funding leads to better societal outcomes.
</step>
<step>
## claim
Research shows that well-funded schools have higher graduation rates and better test scores.
</step>
<step>
## claim
Education reduces crime rates and increases earning potential, benefiting the entire economy.
</step>
<step>
## claim
In conclusion, investing in public education is an investment in our collective future.
</step>
""".strip()
			),
			"number_of_additional_reasoning_steps": 1,
		},
		"output": {
			"considerations": "This argument has a thesis, evidence, analysis, and conclusion. The reasoning is complete and no further steps are needed.",
			"action": "finish",
		},
	},
]
