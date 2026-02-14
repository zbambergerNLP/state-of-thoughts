# Action Space JSON Specifications

This directory contains JSON representations of action spaces for controlled reasoning interventions. Each JSON file defines a **dimension** along which reasoning can be guided, with specific **choices** that can be selected within that dimension.

## Overview

Action spaces enable fine-grained control over how a language model generates reasoning steps. By selecting a choice from a dimension, you can force the next reasoning step to:

- **Structure**: Follow a specific discourse pattern (e.g., providing an example, introducing a counterpoint, citing evidence, etc.)
- **Style**: Adopt a particular rhetorical technique (e.g., metaphor, statistical presentation, formal tone, etc.)
- **Subtopic**: Analyze through a specific lens or framework (e.g., economic impact, human rights, risk assessment, etc.)

## File Structure

Each JSON file follows this schema:

```json
{
	"name": "Dimension Name",
	"definition": "High-level description of what interventions along this dimension do",
	"actions": {
		"Action Name": {
			"definition": "What this specific action does",
			"internal_reasoning": "(optional) Internal reasoning prompt for the model",
			"prefix": "(optional) Text prefix to inject at the start of generation"
		}
	}
}
```

### Required Fields

1. **`name`** (string): The human-readable name of the dimension
   - For expanded versions, append "(Expanded)" to the name
   - Example: `"Causal Structures"` or `"Causal Structures (Expanded)"`

2. **`definition`** (string): A description of what interventions along this dimension accomplish
   - Should be written from the perspective of **what the intervention forces/controls**
   - Format: "Force the next reasoning step to [do something], controlling [what aspect]. Interventions along this dimension ensure the next step [specific behavior]."
   - Example: `"Force the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example)."`

3. **`actions`** (object): A dictionary mapping action names to their specifications
   - Keys are action names (strings)
   - Values are action specification objects

### Action Specification Fields

Each action in the `actions` dictionary must include:

1. **`definition`** (string, required): What this specific action does
   - Should be concise and descriptive
   - Example: `"Acknowledge counterpoints or highlight opposing perspectives."`

2. **`internal_reasoning`** (string, optional): Internal reasoning prompt for the model
   - Provides guidance on how the model should think when using this action
   - Typically starts with "I should..."
   - Example: `"I should place contrasting concepts side by side to highlight differences and sharpen the argument. "`
   - **Note**: Include a trailing space if present in the original

3. **`prefix`** (string, optional): Text prefix to inject at the start of generation
   - Forces the model to continue from this specific text
   - Used primarily for structural interventions
   - Example: `"However"` or `"Evidence shows that"`

## Current Dimensions

### 1. Structures
- **Purpose**: Control discourse structure and how ideas connect
- **Files**: 
  - `structures.json` (10 core patterns)
  - `structures_expanded.json` (100+ fine-grained patterns)
- **Action fields**: `definition`, `prefix`
- **Example actions**: Causal Reasoning, Conditional, Concession & Contrast, Evidence & Authority

### 2. Styles
- **Purpose**: Control rhetorical style and expressive techniques
- **Files**:
  - `styles.json` (10 core styles)
  - `styles_expanded.json` (90+ fine-grained styles)
- **Action fields**: `definition`, `internal_reasoning`
- **Example actions**: Figurative Language, Statistical & Data-Driven, Narrative & Anecdote, Measured & Authoritative Tone

### 3. Subtopics
- **Purpose**: Control analytical lens and topical framework
- **Files**:
  - `subtopics.json` (10 core frameworks)
  - `subtopics_expanded.json` (100+ frameworks organized by tier)
- **Action fields**: `definition`, `internal_reasoning`
- **Example actions**: Cost-Benefit & Impact Analysis, Rights & Liberties, Justice & Fairness, Ethical Principles

## Creating New Action Space JSONs

Follow these steps to create a new action space JSON:

### Step 1: Define the Dimension

Determine what aspect of reasoning your dimension will control. Ask:
- What does an intervention along this dimension **force** or **control**?
- What specific behaviors does it ensure in the next reasoning step?

Write a definition following this template:
```
"Force the next reasoning step to [main action], controlling [what aspect]. 
Interventions along this dimension ensure the next step [specific behaviors with examples]."
```

### Step 2: Identify Actions

List all possible actions within your dimension. For each action:

1. **Name**: Choose a clear, descriptive name
   - Use title case
   - Be specific but concise
   - Group related actions with "&" or parentheticals if creating a condensed version
   - Examples: "Metaphor", "Statistical Presentation", "Cost-Benefit Analysis"

2. **Definition**: Write a concise description of what the action does
   - Focus on the observable behavior or outcome
   - Keep it to 1-2 sentences
   - Example: `"Present numerical data, statistics, or quantified evidence."`

3. **Internal Reasoning** (if applicable): Write a prompt for the model's internal reasoning
   - Start with "I should..."
   - Describe the cognitive strategy or approach
   - Example: `"I should use numbers and data to provide concrete, measurable support for claims. "`
   - Include trailing space if desired

4. **Prefix** (if applicable): Specify a text prefix to inject
   - Use for structural interventions that benefit from explicit connectives
   - Keep it short and natural
   - Example: `"Therefore"`, `"For example"`, `"On the other hand"`

### Step 3: Organize Actions

Consider creating both standard and expanded versions:

- **Standard version**: 10-15 high-level, frequently-used actions
  - Combine related fine-grained actions into broader categories
  - Example: "Causal Reasoning" instead of separate "Cause (Forward)", "Cause (Backward)", "Consequence"
  
- **Expanded version**: 50-100+ fine-grained, specialized actions
  - Provide maximum control and specificity
  - Organize into logical sections with comments (in source, not JSON)
  - Example: Separate actions for each type of causal relation, temporal relation, etc.

### Step 4: Format as JSON

Structure your JSON file following the schema:

```json
{
	"name": "Your Dimension Name",
	"definition": "Force the next reasoning step to...",
	"actions": {
		"Action 1": {
			"definition": "What this action does",
			"internal_reasoning": "I should... (optional)",
			"prefix": "Text prefix (optional)"
		},
		"Action 2": {
			"definition": "What this action does"
		}
	}
}
```

**Formatting guidelines**:
- Use tabs for indentation (not spaces)
- Include trailing newline at end of file
- Use double quotes for all strings
- Preserve trailing spaces in `internal_reasoning` if present in source

### Step 5: Create Expanded Version (Optional)

If creating an expanded version:

1. Name it with "(Expanded)" suffix: `"Your Dimension Name (Expanded)"`
2. Update the definition to mention the expanded scope
3. Add a sentence about how many actions and how they're organized
4. Save as `your_dimension_expanded.json`

Example:
```json
{
	"name": "Your Dimension Name (Expanded)",
	"definition": "Force the next reasoning step to... This expanded version includes 100+ fine-grained options spanning [categories].",
	"actions": { ... }
}
```

## Creating Actions Programmatically

You can create action space JSON files programmatically:

```python
import json

actions = {
    "action_name": {
        "definition": "Description of what this action does",
        "internal_reasoning": "Optional internal reasoning guidance",
        "prefix": "Optional prefix text"
    }
}

result = {
    "name": "Your Dimension Name",
    "definition": "Force the next reasoning step to...",
    "actions": actions
}

with open("your_dimension.json", "w") as f:
    json.dump(result, f, indent="\t")
```

## Best Practices

### Naming Conventions
- **Dimensions**: Use descriptive nouns (e.g., "Causal Structures", "Rhetorical Styles")
- **Actions**: Use clear, specific names that indicate the behavior
- **Files**: Use snake_case for filenames (e.g., `structures.json`)

### Writing Definitions

**Dimension definitions** should:
- Start with "Force the next reasoning step to..."
- Explain what aspect is being controlled
- Provide 2-3 concrete examples in parentheses
- Be 2-3 sentences maximum

**Action definitions** should:
- Be concise (1-2 sentences)
- Focus on observable behavior
- Avoid implementation details
- Use present tense

### Internal Reasoning

When to include `internal_reasoning`:
- ✅ For stylistic or tonal interventions where the model needs guidance on *how* to think
- ✅ For analytical frameworks where the model should adopt a particular perspective
- ❌ For structural interventions where the `prefix` provides sufficient guidance
- ❌ When the action definition is self-explanatory

### Prefixes

When to include `prefix`:
- ✅ For discourse connectives that signal structural relations
- ✅ For transitions between reasoning steps
- ✅ When the prefix naturally leads into the desired behavior
- ❌ For stylistic or tonal interventions (use `internal_reasoning` instead)
- ❌ When the prefix would be too constraining or unnatural

## Validation

Before finalizing your JSON:

1. **Schema validation**: Ensure all required fields are present
2. **Consistency**: Check that all actions follow the same format
3. **Completeness**: Verify that actions cover the full dimension
4. **Clarity**: Test that definitions are clear and unambiguous
5. **JSON validity**: Validate JSON syntax (use `python -m json.tool your_file.json`)

## Usage

These JSON files can be loaded and used to:

1. **Define tool parameters** for controller-based reasoning systems
2. **Generate prompts** that guide language model behavior
3. **Create user interfaces** for selecting reasoning interventions
4. **Analyze reasoning trajectories** by categorizing the actions taken

Example loading code:

```python
import json

with open('action_space/structures.json', 'r') as f:
    dimension = json.load(f)

print(f"Dimension: {dimension['name']}")
print(f"Definition: {dimension['definition']}")
print(f"Number of actions: {len(dimension['actions'])}")

for action_name, action_spec in dimension['actions'].items():
    print(f"\n{action_name}:")
    print(f"  Definition: {action_spec['definition']}")
    if 'prefix' in action_spec:
        print(f"  Prefix: {action_spec['prefix']}")
    if 'internal_reasoning' in action_spec:
        print(f"  Internal Reasoning: {action_spec['internal_reasoning']}")
```