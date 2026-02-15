# **STATe-of-Thoughts: Structured Action Templates for Tree-of-Thoughts**

<p align="center">
  <img src="figures/Figure_2_STATe_Detail.png" alt="STATe-of-Thoughts Plan-Generate-Evaluate-Select Loop" width="100%"/>
</p>

**STATe-of-Thoughts** (STATe) is an explainable Inference-Time-Compute (ITC) framework that *searches* over high-level reasoning patterns. STATe replaces stochastic temperature-based sampling with discrete, interpretable textual interventions: a **controller** selects actions encoding high-level reasoning choices, a **generator** produces reasoning steps conditioned on those choices, and an **evaluator** scores candidates to guide beam search.

Built on [DSPy](https://github.com/stanfordnlp/dspy) and [vLLM](https://github.com/vllm-project/vllm), this framework enables local LLMs to perform systematic exploration of reasoning trajectories, evaluate intermediate steps (process supervision), and select promising paths for complex tasks like argumentation, creative writing, and more.

**Key advantages:**
1. **Diversity**: Action-guided textual interventions produce greater response diversity than temperature-based sampling.
2. **Interpretability**: Explicit action sequences are auditable and interpretable. In our experiments, they proved highly predictive of output quality.
3. **Controllability**: Learned associations between actions and outcomes allow steering generation toward promising regions of the action space.

---

> **Note:** For background on DSPy primitives (Signatures, Modules, Adapters, etc.) and how they are used in this framework, see our **[Background Guide](BACKGROUND.md)**.

---

## **Setup**

### **Prerequisites**

- **Python 3.12+**
- **GPUs:** Recommended setup is 2 GPUs (e.g., GPU 0 for Generation, GPU 1 for Reranking)

### **Clone the repository**

If you are setting up on a remote server with GPU access, clone the repository first:

```bash
git clone https://github.com/zbambergerNLP/state-of-thoughts.git
cd state-of-thoughts
```

### **Environment Setup**

Choose one of the following options to create a Python 3.12+ environment.

#### **(1) Python virtualenv (venv)**

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_server.txt
```

#### **(2) Conda**

```bash
conda create -n state-of-thoughts python=3.12
conda activate state-of-thoughts

pip install -r requirements_server.txt
```

#### **(3) uv**

> **Warning:** On macOS, there are known issues installing vLLM with uv-managed environments. Use conda or venv on Mac instead.

```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

uv pip install -r requirements_server.txt
```

### **Download Models**

After activating your environment (with any of the options above), run the download script:

```bash
python scripts/download_models.py --model_directory /path/to/model_storage
```

This downloads the default generative model (Qwen3-30B-A3B-Instruct-2507) and reranker (Qwen3-Reranker-8B). To download specific models:

```bash
python scripts/download_models.py --model_directory /path/to/model_storage \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --model Qwen/Qwen3-Reranker-8B
```

---

## **Quick Start**

### **(1) Minimal example: instantiate and run STATe**

A minimal example for argument generation (consistent with `experiments/argument_generation/`). This instantiates STATe with an action space, reranker controller, and early stopping:

```python
import os

from adapter.constraints import ResponseLength
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.scoring_local_lm import ScoringLocalVLLM
from predict.tree_of_thoughts import TreeOfThoughts
from predict.tree_of_thoughts.tree_parameters import TreeOfThoughtsParameters
from signatures.example_signatures import GenerateArgumentWithReasoning

# 1. Initialize models (requires 2 GPUs)
model_dir = "/path/to/model_storage"
generative_lm = GenerativeLocalVLLM(
    model=os.path.join(model_dir, "Qwen3-30B-A3B-Instruct-2507"),
)
reranker_lm = ScoringLocalVLLM(
    model=os.path.join(model_dir, "Qwen3-Reranker-8B"),
)

# 2. Action space defines controller choices (structure + subtopic dimensions)
action_space_dir = "experiments/argument_generation/action_space"
action_space_paths = [
    os.path.join(action_space_dir, "structures.json"),
    os.path.join(action_space_dir, "subtopics.json"),
]

# 3. Instantiate STATe (action space + reranker controller + early stopping)
state_of_thoughts = TreeOfThoughts(
    generator_signature=GenerateArgumentWithReasoning,
    evaluator_signature=None,  # Reranker evaluator derives from generator
    generative_lm=generative_lm,
    reranker_lm=reranker_lm,
    controller_type="reranker",
    thought_length=ResponseLength(granularity="sentence", bounds=(1, 3)),
    response_length=ResponseLength(granularity="sentence", bounds=(5, 7)),
    max_reasoning_steps=3,
    final_output_kind="synthesis_faithful",
    action_space_paths=action_space_paths,
    early_stopping_enabled=True,
    seed=42,
)

# 4. Configure tree-search hyperparameters
tot_parameters = TreeOfThoughtsParameters(
    depth=3,                      # Max reasoning layers before final output
    n_samples_generation=5,       # Branching factor (candidates per node)
    top_k=3,                      # Beam width (candidates kept per layer)
    n_samples_judge=1,            # Evaluator samples per candidate
    generator_temperature=0.7,    # Higher = more diverse generations
    judge_temperature=0.7,        # Evaluator sampling (reranker ignores this)
    num_final_candidates=1,       # Number of best arguments to return
    do_pruning=True,              # Prune low-scoring candidates
)

# 5. Run inference (state keys must match generator signature input fields)
output = state_of_thoughts.forward(
    state={"topic": "The government should implement a Universal Basic Income.", "stance": "PRO"},
    tot_parameters=tot_parameters,
)

print(output.response_strings[0])
```

### **(2) Run the built-in argument generation script**

The main entry point for our argument generation experiment is `experiments/argument_generation/generate_arguments.py`:

```bash
python experiments/argument_generation/generate_arguments.py \
    --model_directory /path/to/model_storage \
    --topic "The government should implement a Universal Basic Income (UBI) for all citizens." \
    --stance PRO \
    --depth 3 \
    --n_samples_generation 5 \
    --top_k 3 \
    --n_samples_judge 1 \
    --generator_temperature 0.7 \
    --judge_temperature 0.7 \
    --do_pruning \
    --experiment_mode synthesis_faithful \
    --outputs_directory ./experiments/argument_generation/outputs \
    --outputs_filename arguments \
    --do_save_tree
```

> **Note:** Like the minimal example above, this script requires **two separate GPUs** by default, one for the generative model (Generator, Evaluator, and optionally the generative Controller) and one for the reranker model (Controller action scoring).

### **Key Flags**

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--model` | Generative model name | `Qwen3-30B-A3B-Instruct-2507` |
| `--reranker_model` | Reranker model for scoring | `Qwen3-Reranker-8B` |
| `--model_directory` | Directory containing downloaded models | `/path/to/model_storage` |
| `--generative_gpu_index` | GPU index for generative model | `0` |
| `--reranker_gpu_index` | GPU index for reranker model | `1` |

**Tree Search Parameters:**

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--depth` | Maximum depth of reasoning tree ($d$) | `2` |
| `--n_samples_generation` | Branching factor / candidates per node ($n$) | `3` |
| `--top_k` | Beam width ($k$) | `2` |
| `--do_pruning` | Enable pruning low-scoring nodes | `False` |
| `--use_self_consistency` | Enable self-consistency voting | `False` |
| `--num_final_candidates` | Number of final outputs to return | `1` |
| `--action_space_paths` | Paths to action space JSON files (one per dimension) | `None` |

**Generation Parameters:**

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--generator_temperature` | Temperature for generation | `1.0` |
| `--controller_temperature` | Temperature for generative controller | `1.2` |
| `--judge_temperature` | Temperature for evaluator | `0.7` |
| `--experiment_mode` | Final output method: `synthesis_strict`, `synthesis_faithful`, `synthesis_restructured`, or `conclusion` | `synthesis_faithful` |

**Output & Logging:**

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--do_save_tree` | Save full tree structure to disk | `False` |
| `--outputs_directory` | Directory for saved outputs | Current directory |
| `--outputs_filename` | Filename for outputs (auto-timestamped if not set) | `None` |
| `--verbosity` | Logging level: `debug`, `info`, `warning`, `error` | `info` |

---

## **Method**

STATe extends Tree of Thoughts (ToT) with three methodological contributions:

1. **Action-guided interventions**: Replaces stochastic temperature sampling with discrete action templates that diversify branches in tree search.
2. **Reliable evaluation**: Supports both verifiable and task-specific LLM-as-a-Judge evaluators to score and select among diverse candidates.
3. **Action attribution**: Tracks actions along trajectories, enabling systematic analysis of which reasoning patterns drive performance.

### **Tree of Thoughts Components**

STATe instantiates ToT as a modular **Plan &rarr; Generate &rarr; Evaluate &rarr; Select** loop.
At each layer $i$, STATe starts with a list of states, each of the form $s_i = [x, Z_i]$.
$x$ here represents both the task signature and specific input values, and $Z_i$ represents the reasoning steps generated so far.
The controller, $C$, selects $n$ interventions from action space $\mathcal{A}$ for each state in the frontier.
The generator, $G$, then produces completions that extend each of these interventions.
Finally, $V_{\text{PRM}}(s_i)$ scores intermediate trajectories, while $V_{\text{ORM}}(s_i)$ scores trajectories that produced final answers $y$.
The top-k intermediate states (i.e., ones that did not produce final answers) are retained for the next layer (beam selection).

```mermaid
flowchart LR
    Plan["<b>Plan</b><br/>Controller C<br/>selects actions<br/>from A"]
    Gen["<b>Generate</b><br/>Generator G<br/>produces candidates<br/>conditioned on actions"]
    Eval["<b>Evaluate</b><br/>Evaluator V<br/>scores candidates"]
    Sel["<b>Select</b><br/>Beam Search<br/>keeps top-k states"]

    Plan --> Gen --> Eval --> Sel
    Sel -->|"next layer"| Plan

    style Plan fill:#dceefb,stroke:#333,color:#000
    style Gen fill:#d5f5e3,stroke:#333,color:#000
    style Eval fill:#fdebd0,stroke:#333,color:#000
    style Sel fill:#fadbd8,stroke:#333,color:#000
```

| Step | Component | Function |
|:-----|:----------|:---------|
| **Plan** | Controller ($C$) | Selects actions $\{a_i^1, \ldots, a_i^n\} = C(s_{i-1}, \mathcal{A}, n)$ |
| **Generate** | Generator ($G$) | Produces candidates $z_i^j \sim G(s_{i-1}, \text{prefill}(Z_{i-1}, a_i^j()); \text{temp})$ |
| **Evaluate** | Evaluator ($V$) | Scores intermediate states: $V_{\text{PRM}}(s_i)$; scores final states: $V_{\text{ORM}}(s_i)$ |
| **Select** | Beam Search | Keeps top-k candidates from $L_i'$ ranked by score v |

---

### **1. Core Modules (`predict/`)**

Three modules implement the Plan &rarr; Generate &rarr; Evaluate &rarr; Select cycle. Each module wraps an LLM and uses adapters to format prompts and parse outputs.

<table>
<tr>
<th align="left">Module</th>
<th align="left">Role</th>
<th align="left">Input</th>
<th align="left">Output</th>
</tr>
<tr>
<td><b>Controller</b></td>
<td>Select actions from action space</td>
<td>State s<sub>i-1</sub>, action space A</td>
<td>Actions {a<sub>i</sub><sup>1</sup>, ..., a<sub>i</sub><sup>n</sup>}, each yielding a <code>ReasoningIntervention</code></td>
</tr>
<tr>
<td><b>Generator</b></td>
<td>Produce candidate reasoning steps or final answers</td>
<td>State + ReasoningIntervention (prefix, internal_reasoning)</td>
<td>Reasoning step z<sub>i</sub> or final answer y</td>
</tr>
<tr>
<td><b>Evaluator</b></td>
<td>Score candidate states</td>
<td>Child state (candidate)</td>
<td>Scalar score in [0, 1]</td>
</tr>
</table>

---

#### **Controller**

The Controller ($C$) observes the current state and selects actions from an action space $\mathcal{A}$.
Each action is treated as a *tool call*: selecting an action corresponds to choosing a tool name and providing values for its arguments (if any).
Executing the tool returns a `ReasoningIntervention`, a structured object containing an `internal_reasoning` string (guidance injected into context) and a `prefix` string (text pre-filled at the start of the next generation).

Two controller implementations exist:

**Generative Controller** (`TreeOfThoughtsController`): Uses a generative LLM to produce tool calls. Creates a *single combined tool* with one parameter per action-space dimension. The model generates one choice per parameter.

```mermaid
flowchart TD
    State["State<br/>(input + reasoning)"] --> Prompt["Controller Prompt"]
    ActionSpace["Action Space<br/>(tool definitions)"] --> Prompt

    Prompt -->|"call LLM"| LLM["Generative LLM"]
    LLM -->|"parse"| ToolCall["Tool Call<br/>(name + arguments)"]
    ToolCall -->|"execute"| Intervention["ReasoningIntervention<br/>internal_reasoning + prefix"]

    style State fill:#cfe2ff,stroke:#333,color:#000
    style ActionSpace fill:#e8daef,stroke:#333,color:#000
    style Prompt fill:#fff3cd,stroke:#333,color:#000
    style LLM fill:#cfe2ff,stroke:#333,color:#000
    style ToolCall fill:#fff3cd,stroke:#333,color:#000
    style Intervention fill:#d1e7dd,stroke:#333,color:#000
```

> **Note:** When sampling multiple actions, the generative controller tracks co-occurrence counts for duplicate (tool, arguments) pairs. This allows promising actions to be sampled $n$ times (where $n$ is the occurrence count), or executed once if deduplication is preferred.

**Reranker Controller** (`TreeOfThoughtsControllerReranker`): Scores all action-argument combinations using a discriminative reranker model. Creates *one tool per combination* of choices across all dimensions, then scores each tool's description against the current state.

```mermaid
flowchart TD
    State["State<br/>(input + reasoning)"] --> Query["Query"]
    ActionSpace["Action Space<br/>(all combinations, ...)"] --> Docs["Documents<br/>(one per combination)"]

    Query --> Scoring["Reranker LLM<br/>score each (query, doc)"]
    Docs --> Scoring

    Scoring -->|"top-n"| TopN["Top-n actions<br/>(sorted by score)"]
    TopN -->|"execute each"| Interventions["ReasoningInterventions<br/>internal_reasoning + prefix"]

    style State fill:#cfe2ff,stroke:#333,color:#000
    style ActionSpace fill:#e8daef,stroke:#333,color:#000
    style Query fill:#cfe2ff,stroke:#333,color:#000
    style Docs fill:#e8daef,stroke:#333,color:#000
    style Scoring fill:#fff3cd,stroke:#333,color:#000
    style TopN fill:#fdebd0,stroke:#333,color:#000
    style Interventions fill:#d1e7dd,stroke:#333,color:#000
```

**Controller Output**: Both controllers produce `ControllerPrediction` objects containing:

| Field | Description | Example |
|:------|:------------|:--------|
| `tool` | The selected `dspy.Tool` | tool for `"select_reasoning_intervention"` or `"finish"` |
| `chosen_values` | Tool arguments (if any) | `{"structures": "causal_reasoning", "subtopics": "justice_and_fairness"}` |
| `intervention` | `ReasoningIntervention` from executing the tool | `ReasoningIntervention(continue_reasoning=True, internal_reasoning="I should analyze whether...", prefix="Therefore")` |
| `considerations` | Rationale for the choice | `"The argument needs causal structure..."` |
| `intervention.continue_reasoning` | Whether to generate another reasoning step | `True` / `False` |

---

#### **Defining Action Spaces (Tools)**

Action spaces define the dimensions along which STATe's controller can intervene on reasoning. Each dimension (e.g., *structure*, *style*, *subtopic*) is specified as a **JSON file** with a name, a definition, and a dictionary of choices. Each choice maps to intervention fields (`internal_reasoning` and/or `prefix`) that are injected into the generator's next step.

**Action Space JSON Schema:**

```json
{
  "name": "<Dimension Name>",
  "definition": "<Description of what interventions along this dimension do>",
  "choices": {
    "<choice_key>": {
      "definition": "<What this choice does>",
      "internal_reasoning": "<Guidance injected into context (optional)>",
      "prefix": "<Text pre-filled at start of generation (optional)>"
    }
  }
}
```

> **Important:** Only one dimension can provide a `prefix`, since the prefix occupies a fixed position at the start of the generated text. All dimensions can contribute `internal_reasoning` (their guidance strings are concatenated).

**Example: Argument Generation Action Spaces**

STATe's argument generation experiment uses three action-space dimensions:

**1. Structures** (`experiments/argument_generation/action_space/structures.json`): Controls discourse structure via a prefix:

```json
{
  "name": "Structures",
  "definition": "Forces the next reasoning step to adhere to a specific discourse structure...",
  "choices": {
    "causal_reasoning": {
      "definition": "States causes, effects, consequences, or logical implications.",
      "prefix": "Therefore"
    },
    "conditional": {
      "definition": "Introduces conditional, hypothetical, or counterfactual scenarios.",
      "prefix": "If"
    },
    "concession_and_contrast": {
      "definition": "Acknowledges counterpoints or highlights opposing perspectives.",
      "prefix": "However"
    },
    "exemplification": {
      "definition": "Provides concrete examples, illustrations, or case studies.",
      "prefix": "For example"
    }
  }
}
```

**2. Subtopics** (`experiments/argument_generation/action_space/subtopics.json`): Controls content theme via internal reasoning:

```json
{
  "name": "Subtopics",
  "definition": "Forces the next reasoning step to analyze the issue through a specific argumentative lens...",
  "choices": {
    "cost_benefit_and_impact_analysis": {
      "definition": "Weighs economic, social, and practical consequences systematically",
      "internal_reasoning": "I should quantify and compare costs, benefits, and real-world impacts..."
    },
    "rights_and_liberties": {
      "definition": "Protects fundamental rights, freedoms, privacy, and individual autonomy",
      "internal_reasoning": "I should consider inalienable human rights, civil liberties..."
    }
  }
}
```

**3. Styles** (`experiments/argument_generation/action_space/styles.json`): Controls rhetorical style via internal reasoning:

```json
{
  "name": "Causal Styles",
  "definition": "Forces the next reasoning step to adopt a specific rhetorical style...",
  "choices": {
    "figurative_language": {
      "definition": "Uses metaphor, simile, analogy, or symbolism...",
      "internal_reasoning": "I should employ non-literal comparison to make abstract concepts vivid..."
    },
    "statistical_and_data_driven": {
      "definition": "Presents numerical data, statistics, or quantified evidence.",
      "internal_reasoning": "I should use numbers and data to provide concrete, measurable support..."
    }
  }
}
```

**How controllers use action spaces:**

- The **generative controller** creates a *single combined tool* with one parameter per dimension.
The LLM generates a choice for each parameter (e.g., `{"structures": "causal_reasoning", "subtopics": "justice_and_fairness", "styles": "statistical_and_data_driven"}`).
Executing the tool combines the `internal_reasoning` and `prefix` from all chosen values.

- The **reranker controller** creates *one tool per combination* of choices across all dimensions (e.g., [10 structures](experiments/argument_generation/action_space/structures.json) &times; [10 subtopics](experiments/argument_generation/action_space/subtopics.json) &times; [10 styles](experiments/argument_generation/action_space/styles.json) = 1,000 tools). 
Each tool has a description derived from its choices, and the reranker scores all tools against the current state to select the top-$n$.

**Creating action spaces for your own tasks:**

1. **Identify controllable dimensions**: Enumerate aspects of generation that can be meaningfully controlled at each step (content, structure, style, strategy).
2. **Decide prefix vs. internal reasoning**: Only one dimension can use a prefix; all can use internal reasoning. Structural/discourse dimensions benefit most from prefix control.
3. **Consider early stopping**: Include a `finish` tool if variable-depth reasoning is desired. The finish tool is automatically added when `early_stopping_enabled=True` (the default).
4. **Topic-specific subtopics**: You can create topic-specific action spaces (see `subtopics_specific_pollution.json` for an example tailored to single-use plastics).

See Appendix C of the paper for detailed practitioner guidance on action space design.

---

#### **Generator**

The Generator ($G$) expands the reasoning tree by producing candidate thoughts $z_i^j$ or final outputs $y$. Given a parent state $s_{i-1} = [x, Z_{i-1}]$ and an action $a_i^j$, we sample a continuation:

$$z_i^j \sim p_\theta(z \mid x, \text{prefill}(Z_{i-1}, a_i^j()); \text{temp})[\text{stop\_token}]$$

The prefill operation ensures that the model's generation begins with the intervention text, biasing reasoning along the desired dimension. Stop tokens (`</step>` for reasoning, `</answer>` for final output) control when generation halts.

```mermaid
flowchart LR
    State["State s<sub>i-1</sub>"]
    Intervention["ReasoningIntervention<br/>(internal_reasoning, prefix)"]
    Prefill["Prefill assistant<br/>message"]
    vLLM["vLLM Generation<br/>(stop at &lt;/step&gt; or &lt;/answer&gt;)"]
    Child["Child state s<sub>i</sub>"]

    State --> Prefill
    Intervention --> Prefill
    Prefill --> vLLM
    vLLM --> Child
```

**Synthesis Modes:**
Once the maximum depth $d$ is reached or the controller selects `FINISH`, STATe synthesizes a final output from the reasoning trace. Four modes are supported:

| Mode | Description |
|:-----|:------------|
| **Strict** | Concatenates reasoning steps verbatim with minimal connectives |
| **Faithful** | Permits rephrasing while preserving order and structure |
| **Restructured** | Allows free reorganization using the trace as source material |
| **Conclusion** | Treats the trace as internal guidance only; no constraints on final output |

---

#### **Evaluator**

The Evaluator ($V$) assigns scalar scores to guide beam search:

- **PRM (Process Reward Model)**: Scores intermediate reasoning states $V_{\text{PRM}}(s_i) \to [0,1]$ where $s_i = [x, Z_i]$
- **ORM (Outcome Reward Model)**: Scores final states $V_{\text{ORM}}(s_i) \to [0,1]$ where $s_i = [x, Z_{i-1}, y]$

Three evaluator implementations are supported:
1. **Generative LLM-as-a-Judge**: Scores candidates against a rubric
2. **Reranker LLM-as-a-Judge**: Assigns latent relevance scores
3. **Deterministic verifier**: Programmatic evaluation (e.g., code correctness)

```mermaid
flowchart LR
    Candidates["Candidate<br/>states"]
    PRM["V_PRM<br/>(process scoring)"]
    ORM["V_ORM<br/>(outcome scoring)"]
    Scores["Scalar scores<br/>∈ [0, 1]"]

    Candidates -->|"intermediate"| PRM --> Scores
    Candidates -->|"final"| ORM --> Scores
```

**Weighted Rubrics**: The evaluator can use `rubric_weight` from the signature to combine multiple dimensions:

$$\text{score} = \sum_i (\text{score}_i \times \text{weight}_i)$$

---

### **2. Signatures & Fields (`signatures/`)**

Signatures define task schemas. We extend DSPy with `ReasoningSignature` and `ReasoningField`.

#### **ReasoningSignature**

A signature with three field types:

| Field Type | Class | Notation | Generated When |
|:-----------|:------|:---------|:---------------|
| **Input** | `InputField` | $x$ | Provided by user |
| **Reasoning** | `ReasoningField` | $z$ | Each step (iteratively) |
| **Output** | `OutputField` | $y$ | When controller says `FINISH` or max depth reached |

```python
from signatures import ReasoningSignature, InputField, ReasoningField, OutputField

class ArgumentGeneration(ReasoningSignature):
    """Generate an argument for the given stance on the topic."""
    
    # Inputs (x) - provided by user
    topic: str = InputField(desc="The topic of the argument")
    stance: str = InputField(desc="The stance to argue for (PRO or ANTI)")
    
    # Reasoning (z) - generated iteratively, one per step
    claim: str = ReasoningField(desc="A supporting claim for the stance")
    
    # Output (y) - generated when reasoning is complete
    argument: str = OutputField(desc="The final synthesized argument")
```

#### **Extended Field Features**

**ReasoningField**: Forces structured intermediate steps. The model generates one value per reasoning step until the Controller decides to finish or the maximum number of steps is reached.

**rubric_weight**: Enables weighted multi-dimensional evaluation:

```python
class EvaluateArgument(ReasoningSignature):
    """Evaluate an argument on multiple dimensions."""
    argument: str = InputField(desc="The argument to evaluate")
    
    # Weighted scoring: 30% + 30% + 40% = 100%
    persuasiveness: int = OutputField(
        desc="How convincing (1-7)", rubric_weight=0.3, ge=1, le=7
    )
    coherence: int = OutputField(
        desc="How well-structured (1-7)", rubric_weight=0.3, ge=1, le=7
    )
    relevance: int = OutputField(
        desc="How on-topic (1-7)", rubric_weight=0.4, ge=1, le=7
    )
```

**Pydantic Constraints**: Automatically translated to prompt instructions:

```python
# Generates: "Each claim should be between 2 and 5 sentences."
claim: str = ReasoningField(min_length=2, max_length=5, granularity="sentence")
```

---

### **3. Adapters (`adapter/`)**

Adapters translate abstract signatures into concrete LLM prompts and parse outputs back into structured data.

```mermaid
flowchart LR
    Sig["Signature<br/>(task schema)"]
    Adapter["Adapter"]
    Prompt["Formatted<br/>LLM Prompt"]
    Response["Raw LLM<br/>Response"]
    Parsed["Parsed<br/>Prediction"]

    Sig --> Adapter
    Adapter --> Prompt
    Prompt -->|"LLM call"| Response
    Response --> Adapter
    Adapter --> Parsed
```

#### **VLLMGeneratorAdapter**

The core adapter for multi-step reasoning with four key capabilities:

**1. XML-based Reasoning Template**

Structures LLM responses using XML tags:

```text
<thinking>
<step>
## internal_reasoning
I should introduce my primary claim
## claim
Studies show that renewable energy reduces costs...
</step>
<step>
## internal_reasoning
I should acknowledge counterarguments
## claim
While opponents argue that...
</step>
...
</thinking>
<answer>
## argument
Renewable energy is economically beneficial because...
</answer>
```

We recognize natural "stopping points" in the model's response through XML tags like `</step>` and `</answer>`. We introduce interventions by injecting internal reasoning and the first few tokens of the reasoning step (prefix) before the model continues generating.

**2. Stop Token Control**

| Controller Decision | Stop Token | Result |
|:--------------------|:-----------|:-------|
| `continue_reasoning=True` | `</step>` | One reasoning step |
| `continue_reasoning=False` | `</answer>` | Final output |

**3. Assistant Pre-filling**

Injects controller interventions using vLLM's `continue_final_message`. The adapter builds an assistant prefill that concatenates internal reasoning (context guidance) and a prefix (text that starts the model's generation). The model then continues from the prefix.

```mermaid
sequenceDiagram
    participant Controller
    participant Adapter as VLLMGeneratorAdapter
    participant vLLM

    Controller->>Adapter: ReasoningIntervention<br/>(internal_reasoning, prefix)
    Adapter->>Adapter: Build assistant prefill:<br/>"## internal_reasoning\n{guidance}\n## claim\n{prefix}"
    Adapter->>vLLM: Messages + continue_final_message=True<br/>stop_token="</step>"
    vLLM->>Adapter: Generated continuation
    Adapter->>Adapter: Parse reasoning step
```

*Illustrative interventions for argument generation (in favor of a single-use plastics ban).* Templates in black, <span style="color:teal">internal reasoning</span> in teal, <span style="color:blue">prefixes</span> in blue, <span style="color:orange">model continuation</span> in orange, <span style="color:#E6A800">final answer</span> in amber. Each column shows the generation state at different stages: first step (single claim), intermediate (multiple claims), and final (complete reasoning with synthesized answer). The answer synthesizes the reasoning steps, often rephrasing the first claim to frame the broader argument.

| **First step** | **Intermediate step** | **Final step** |
|:---------------|:----------------------|:---------------|
| <tt>&lt;thinking&gt;</tt><br><tt>&lt;step&gt;</tt><br><tt>## internal_reasoning</tt><br><span style="color:teal">I should identify risks, unintended outcomes, cascading effects, and potential for escalation.</span><br><tt>## claim</tt><br><span style="color:blue">If</span> <span style="color:orange">current levels of plastic waste continue, they will cause permanent harm to marine ecosystems...</span> | <tt>&lt;thinking&gt;</tt><br><tt>&lt;step&gt;</tt><br><tt>## internal_reasoning</tt><br><span style="color:teal">I should identify risks...</span><br><tt>## claim</tt><br><span style="color:blue">If</span> <span style="color:orange">current levels of plastic waste continue...</span><br><tt>&lt;/step&gt;</tt><br>...<br><tt>&lt;step&gt;</tt><br><tt>## internal_reasoning</tt><br><span style="color:teal">I should evaluate historical precedents, long-term vs short-term tradeoffs, and obligations to future generations.</span><br><tt>## claim</tt><br><span style="color:blue">For example,</span> <span style="color:orange">Canada's existing single-use plastic bans are expected to reduce total waste by 5%...</span> | <tt>&lt;thinking&gt;</tt><br><tt>&lt;step&gt;</tt><br><tt>## internal_reasoning</tt><br><span style="color:teal">I should identify risks...</span><br><tt>## claim</tt><br><span style="color:blue">If</span> <span style="color:orange">current levels of plastic waste continue...</span><br><tt>&lt;/step&gt;</tt><br>...<br><tt>&lt;step&gt;</tt><br><tt>## internal_reasoning</tt><br><span style="color:teal">I should evaluate...</span><br><tt>## claim</tt><br><span style="color:blue">For example,</span> <span style="color:orange">Canada's existing bans...</span><br><tt>&lt;/step&gt;</tt><br>...<br><tt>&lt;/thinking&gt;</tt><br><tt>&lt;answer&gt;</tt><br><tt>## argument</tt><br><span style="color:#E6A800">Given that plastic waste at current levels threatens permanent harm to marine ecosystems, a ban is both necessary and justified. Evidence from Canada shows that...</span> |

**4. Heterogeneous Batching**

Process mixed batches where each item independently continues reasoning or generates output:

```python
outputs = adapter(
    signature=ArgumentGeneration,
    inputs={"topic": "AI", "stance": "PRO"},
    continue_reasoning=[
        [True],   # Trajectory 1: Generate another step
        [False],  # Trajectory 2: Generate final answer
    ],
    previous_content=[traj1_history, traj2_history],
    lm_kwargs={"temperature": 0.7, "n": 2},
)
# Returns: [[step1a, step1b], [answer2a, answer2b]]
```

---

## **Local Development: Syncing to a GPU Server**

Develop locally and run unit tests on your machine. 
For integration tests and experiments (which require GPUs), sync the project to a GPU server and run them there:

```bash
REMOTE_HOST=user@host.example.edu REMOTE_PATH=/remote/path/to/state-of-thoughts/ ./scripts/sync_to_remote.sh
```

Sync only a subdirectory (e.g. `experiments/`). `REMOTE_PATH` must be the matching directory on the server:

```bash
REMOTE_HOST=user@host.example.edu REMOTE_PATH=/remote/path/to/state-of-thoughts/experiments SOURCE_PATH=experiments ./scripts/sync_to_remote.sh
```

With password authentication (requires `sshpass`):

```bash
REMOTE_HOST=user@host.example.edu REMOTE_PATH=/home/user/state-of-thoughts/ REMOTE_PASSWORD=secret ./scripts/sync_to_remote.sh
```

Run the sync script from the project's (local) root to sync the project to the server.
Once synced, SSH into the server, activate a virtual environment (see **Environment Setup** above).
You can then run either integration tests (via `pytest` as described below) or experiments (e.g., via `python experiments/argument_generation/generate_arguments.py`).

---

## **Testing**

The test suite includes both **mock-based unit tests** (fast, no GPU required) and **integration tests** (require GPU access). Run unit tests locally; run integration tests on a GPU server after syncing (see **Local Development** above).

### **Mock-Based Unit Tests**

Unit tests use `MockLocalVLLM` from `utilities_for_tests.py` to simulate model responses without requiring actual GPU resources:

```bash
# Individual components
pytest lm/test_generative_local_lm.py                       # Generative LLM (vLLM)
pytest lm/test_scoring_local_lm.py                          # Scoring/reranker LLM (vLLM)
pytest signatures/test_field.py                             # Fields
pytest signatures/test_signature.py                         # Signatures
pytest adapter/test_vllm_adapter.py                         # Generative adapter (direct generation)
pytest adapter/test_vllm_scoring_adapter.py                 # Scoring/reranker adapter
pytest adapter/test_vllm_generator_adapter.py               # Generator adapter (multi-step reasoning)
pytest adapter/test_constraints.py                          # Response length constraints
pytest adapter/test_tool_schema.py                          # Tool schema formatting
pytest adapter/test_utils.py                                # Adapter utilities
pytest predict/controller/test_controller.py                # Generative controller
pytest predict/controller/test_controller_reranker.py       # Reranker controller
pytest predict/controller/test_controller_utils.py          # Controller utilities
pytest predict/generator/test_generator.py                  # Generator
pytest predict/evaluator/test_evaluator.py                  # Generative evaluator
pytest predict/evaluator/test_evaluator_reranker.py         # Reranker evaluator
pytest predict/test_local_predict.py                        # Local predict module
pytest predict/tree_of_thoughts/test_tree_of_thoughts.py    # Tree of Thoughts (end-to-end)
pytest tree/test_tree.py                                    # Tree data structures
pytest test_misc_utils.py                                   # Miscellaneous utilities
pytest test_utilities_for_tests.py                          # Test utilities (MockLocalVLLM)
```

To run all unit tests (from the root directory):
```bash
pytest .
```

### **Integration Tests**

Integration tests require access to GPUs and run against real models. They are skipped automatically on systems without GPUs. To run them, sync the project to a GPU server (see **Local Development** above), SSH in, activate your environment, then run the same `pytest` commands. The same tests that run as unit tests locally will run as integration tests on the server when CUDA is available.
