# Synthetic Graduate-Level Physics QA System

An end-to-end system that automatically generates high-quality synthetic graduate-level physics QA pairs with rubrics, validates them using multiple frontier LLMs, and provides both CLI and web interfaces.

## What This System Does

This system generates physics questions and answers suitable for PhD qualifying exams. Each data point includes:

- **query**: A challenging graduate-level physics question
- **response_answer**: The concise final answer
- **response_reasoning**: Complete step-by-step derivation/solution
- **rubric**: A structured grading rubric with criteria and point allocations
- **response_images**: Always empty (hard constraint - no images)

The system ensures quality through a multi-stage validation pipeline that tests generated questions against multiple frontier LLMs.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web UI (FastAPI + htmx)                 │
│  - Dashboard showing generation progress                        │
│  - View generated QA pairs                                      │
│  - Start/stop generation                                        │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      CLI Interface (typer)                       │
│  - `python -m src.main generate --count 50`                     │
│  - `python -m src.main serve` (starts web UI)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                         │
│  - Single-threaded async (concurrent I/O, not parallel CPU)     │
│  - Coordinates generation → validation → output                  │
│  - Backfill loop ensures target count is reached                │
│  - Checkpointing for crash recovery                             │
│  - Statistics tracking                                          │
└─────────────────────────────────────────────────────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌────────────────┐
│  Generation   │     │   Validation    │     │   Persistence  │
│   Pipeline    │     │    Pipeline     │     │     Layer      │
├───────────────┤     ├─────────────────┤     ├────────────────┤
│ TopicSampler  │     │ SchemaValidator │     │ JSONL Writer   │
│ QAGenerator   │     │ QwenConstraint  │     │ Checkpoint Mgr │
│ (Claude Opus) │     │ CrossCheckValid │     │ Statistics     │
│ Two-Step Gen  │     │ DerivationAudit │     │                │
│ (Problem+Solve)│    │ FinalAnswerValid│     └────────────────┘
└───────────────┘     └─────────────────┘
                                │
                      ┌─────────────────┐
                      │ OpenRouter API  │
                      │    Client       │
                      └─────────────────┘
```

### Pipeline Flow (with Backfill Loop)

```
┌─────────────────────────────────────────────────────────────────┐
│  OUTER LOOP: while fully_validated < target_count               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: Generate needed_count QAs                       │   │
│  │   - Spawn async workers (one per QA needed)              │   │
│  │   - Two-step generation (problem → independent solve)    │   │
│  │   - Schema validation                                    │   │
│  │   - Completeness check (well-posed problem)              │   │
│  │   - Symbolic math verification (SymPy)                   │   │
│  │   - Sanity check (physics consistency)                   │   │
│  │   - Qwen constraint check (not too easy)                 │   │
│  │   - Cross-check with 4 frontier models                   │   │
│  │   - Write to dataset.jsonl immediately                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ PHASE 2: GPT-4 Final Validation                          │   │
│  │   - Derivation Audit (5 consecutive passes)              │   │
│  │   - Final Answer Validation (5 consecutive blind solves) │   │
│  │   - Add passed QAs to fully_validated_pairs              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  Check: len(fully_validated_pairs) >= target_count?             │
│    - YES → Exit loop, write final.jsonl                         │
│    - NO  → Loop back (BACKFILL ROUND 2, 3, ...)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Behavior**: The pipeline keeps running until `dataset.final.jsonl` contains exactly the number of QAs the user requested. If Phase 2 rejects some QAs, the pipeline automatically generates more to backfill.

### Components

1. **Generation Pipeline** (`src/generation/`)
   - `TopicSampler`: Samples from 12 physics topics with 100+ subtopics
   - `QAGenerator`: Two-step generation process:
     1. Generate problem statement only (no solution)
     2. Independently solve the problem (prevents answer bias)
     3. Generate rubric based on the solution
   - Self-reported confidence check (must be ≥0.95 to proceed)

2. **Validation Pipeline** (`src/validation/`)
   - `SchemaValidator`: Validates structure and format
   - `CompletenessValidator`: Ensures problems are well-posed (all variables defined, boundary conditions specified)
   - `SymbolicMathValidator`: Programmatic verification using SymPy (dimensional analysis, numerical sanity)
   - `SanityCheckValidator`: AI-based physics sanity checks (dimensional analysis, limiting cases, conservation laws)
   - `MultiModelSanityValidator`: Optional multi-model consensus for sanity checks (Claude + GPT-4 + Gemini)
   - `QwenConstraintValidator`: Ensures questions aren't too easy (Qwen <4/5 passes)
   - `CrossCheckValidator`: Tests with 4 frontier models (need ≥5/20 correct)
   - `DerivationAuditor`: GPT-4 checks reasoning quality (5 consecutive passes)
   - `FinalAnswerValidator`: GPT-4 blind solve + compare (5 consecutive passes)

3. **Orchestration** (`src/orchestration/`)
   - `ParallelPipelineRunner`: Coordinates the entire workflow
   - Single-threaded async architecture (concurrent I/O via asyncio)
   - Backfill loop ensures target count is reached in `final.jsonl`
   - Handles checkpointing, statistics, and error recovery

4. **Interfaces**
   - CLI: `src/main.py` (typer-based)
   - Web UI: `src/web/` (FastAPI + htmx)

## Validation Logic

The system implements the **four required validation rules** from the spec, plus additional quality checks:

### 1. No-Images Rule
- `response_images` must be empty for every data point
- No images in questions or anywhere in content

### 2. Qwen Constraint (Anti-"Too Easy" Filter) - DATASET-LEVEL
Per spec: *"The proportion of cases where Qwen3-max-thinking passes at least 4 out of 5 attempts must not exceed 5%."*

This is a **dataset-level** constraint:
- For each question, sample `qwen/qwen3-max` 5 times
- Grade each response using rubric-based evaluation
- If Qwen passes 4+ out of 5 attempts, the question is marked as "easy"
- **Dataset constraint**: No more than 5% of questions can be "easy"

### 3. Frontier Model Cross-Check - "Council of Judges" (per-question)
Per spec, test each question with 4 models × 5 samples = 20 total attempts:
- `deepseek/deepseek-v3.2`
- `openai/o4-mini-2025-04-16`
- `google/gemini-3-pro-preview`
- `x-ai/grok-4`

This acts as a **council of judges** - four independent frontier models each attempt to solve the problem. This catches model-specific blind spots and ensures the question is genuinely solvable.

**Pass criteria (per-question)**:
- At least 2 models must have accuracy > 0 (i.e., at least one correct answer among their 5 attempts)
- Total correct across 20 attempts must be ≥5 (i.e., overall accuracy ≥ 25%)

### 4. Dataset-Level Correctness Target
Per spec: *"Across the accepted dataset, the accuracy should be higher than 90%"*

Ensured through multi-stage validation:
- The query is well-posed and internally consistent
- The response_answer is correct
- The response_reasoning is correct and supports the answer

### 5. Phase 2: GPT-4 Final Validation (Post-Generation)

After QA pairs pass Phase 1 (Schema → Qwen → Cross-check), they go through Phase 2:

**Step 1: Derivation Audit** (GPT-4)
- Checks reasoning quality: logical consistency, no contradictions, complete steps
- Requires **5 consecutive passes**

**Step 2: Final Answer Validation** (GPT-4 Blind Solve)
- GPT-4 solves the problem **without seeing the provided answer**
- Compares its independently derived answer to the provided answer
- Requires **5 consecutive passes**

**Why Blind?**: Eliminates confirmation bias. GPT-4 derives its answer before ever seeing the provided answer.

**On Failure - Full Pipeline Re-validation**:
If Phase 2 validation fails:
1. Extract feedback from the failed validation
2. Regenerate the QA pair with feedback
3. **Re-run Phase 1 validation** (Qwen check → Cross-check)
4. Re-run Phase 2 validation
5. Repeat until pass or max retries reached

### How "Correct" Is Scored

The system uses a **10-point rubric-based grading system**:

**Point Allocation**:
- **Final Answer**: 3 points (correct answer is required but not sufficient)
- **Key Steps**: 7 points (3-7 key steps, each worth 1-4 points)

**Pass Criteria**:
- **Qwen (STRICT)**: Must score ≥7/10 AND ≥2/3 on final answer
- **Cross-check (GENEROUS)**: Must score ≥6/10 AND ≥2/3 on final answer

## How to Run End-to-End

### Prerequisites

- Python 3.11+
- OpenRouter API key (for Claude, DeepSeek, Gemini, Grok, and Qwen models)
- OpenAI API key (for o4-mini model - routed directly to OpenAI)

### Installation

```bash
# Clone the repository
git clone https://github.com/Sbcoit/takeHome-AQ.git
cd takeHome-AQ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env and set:
#   OPENROUTER_API_KEY=sk-or-v1-...
#   OPENAI_API_KEY=sk-...
```

### Running via Web UI

```bash
# Start the web server
python -m src.main serve

# Or with custom host/port
python -m src.main serve --host 0.0.0.0 --port 8080

# Open browser to http://127.0.0.1:8000
```

The web UI provides:
- Real-time progress monitoring
- Generation controls (start/stop)
- Log viewer
- Browse generated QA pairs with category filtering (Validated, Pending Audit, Rejected)
- Export to Excel (.xlsx) for AI/ML workflows
- Detailed view for each question

### Running via CLI

```bash
# Generate 50 QA pairs (default)
python -m src.main generate

# Generate custom count
python -m src.main generate --count 100 --output output/my_dataset.jsonl

# Generate 5 QAs for a specific topic
python -m src.main generate --count 5 --topic quantum_mechanics

# Generate with mixed topics (each worker gets a different topic)
python -m src.main generate --count 5 --mix-topics

# Customize individual retry limits (defaults: schema=3, qwen=20, crosscheck=10, final=10)
python -m src.main generate --count 5 --schema-retries 5 --qwen-retries 8 --crosscheck-retries 8 --final-retries 15

# Set unlimited retries for specific stages (0 = unlimited)
python -m src.main generate --count 5 --qwen-retries 0 --final-retries 0

# With verbose logging
python -m src.main generate --count 10 --verbose

# Show available topics
python -m src.main topics

# Show system info
python -m src.main info

# Validate an existing dataset
python -m src.main validate output/dataset.jsonl

# Export JSONL to Excel (.xlsx) for AI/ML workflows (saves to exports/)
python -m src.main export output/dataset.jsonl

# Export with custom output path
python -m src.main export output/dataset.jsonl -o custom_path/my_export.xlsx
```

**Execution Model**: Single-threaded async (concurrent I/O). Workers run concurrently on a single thread via asyncio, with concurrency limited by semaphore and rate limiter.

**Output Files**:
- `output/dataset.jsonl`: QAs that passed Phase 1 (Qwen + Cross-check)
- `output/dataset.final.jsonl`: QAs that passed **both** Phase 1 and Phase 2 (fully validated)
- `output/dataset.failed.jsonl`: QAs that passed Phase 1 but failed Phase 2 (for debugging)

**Note**: The `output/` directory is **automatically created** when the pipeline runs. When you clone this repo and run the pipeline, it will create the output folder in your project directory - no manual setup required.

**Backfill Behavior**: If you request 10 QAs and Phase 2 rejects 3, the pipeline automatically generates 3 more and validates them until `final.jsonl` has exactly 10.

### Output Format

The output is a JSONL file with one JSON object per line:

```json
{
  "query": "Consider a two-dimensional harmonic oscillator with potential V(x,y) = (1/2)mω²(x² + y²). A particle of mass m is in this potential. (a) Write the Hamiltonian and show that the problem is separable. (b) Find the energy eigenvalues. (c) Determine the degeneracy of the nth energy level.",
  "response_answer": "E_n = ℏω(n + 1), where n = n_x + n_y (n_x, n_y = 0,1,2,...). The degeneracy of level n is (n + 1).",
  "response_reasoning": "Starting with the Hamiltonian H = p_x²/2m + p_y²/2m + (1/2)mω²x² + (1/2)mω²y². This separates as H = H_x + H_y where each is a 1D harmonic oscillator. The energy eigenvalues are E = ℏω(n_x + 1/2) + ℏω(n_y + 1/2) = ℏω(n_x + n_y + 1). For the nth level where n = n_x + n_y, the pairs (n_x, n_y) can be (0,n), (1,n-1), ..., (n,0), giving (n+1) states.",
  "rubric": {
    "total_points": 10,
    "final_answer": {
      "value": "E_n = ℏω(n + 1), degeneracy = (n + 1)",
      "points": 3,
      "tolerance": "equivalent symbolic forms accepted",
      "common_errors": ["Missing ground state energy", "Wrong degeneracy formula"]
    },
    "key_steps": [
      {"step": "Write the Hamiltonian in separable form H = H_x + H_y", "points": 2},
      {"step": "Identify each component as 1D harmonic oscillator", "points": 2},
      {"step": "Sum energy eigenvalues and derive degeneracy", "points": 3}
    ],
    "partial_credit_rules": ["Correct method but arithmetic error: deduct 1-2 points"],
    "automatic_zero": ["Uses completely wrong method for the problem type"]
  },
  "response_images": []
}
```

### Excel Export Format

The system can export JSONL datasets to Excel (.xlsx) format for AI/ML workflows. The Excel format is flattened for easy use with pandas:

| Column | Description |
|--------|-------------|
| `row_id` | Unique row identifier (for data lineage) |
| `topic` | Physics topic category |
| `query` | The physics question |
| `response_answer` | Concise answer |
| `response_reasoning` | Step-by-step derivation |
| `rubric_total_points` | Total points available |
| `rubric_final_answer_value` | Expected final answer |
| `rubric_final_answer_points` | Points for final answer |
| `rubric_final_answer_tolerance` | Acceptable answer variations |
| `rubric_common_errors` | Common errors (JSON array) |
| `rubric_key_steps_count` | Number of key steps |
| `rubric_key_steps` | Grading steps (JSON array) |
| `rubric_partial_credit_rules` | Partial credit rules (JSON array) |
| `rubric_automatic_zero` | Auto-zero conditions (JSON array) |

**Usage with pandas:**
```python
import pandas as pd
import json

df = pd.read_excel("dataset.xlsx")

# Parse nested JSON columns
df['key_steps'] = df['rubric_key_steps'].apply(json.loads)
```

## Design Decisions & Tradeoffs

### 1. Two-Step Generation Process
**Decision**: Generate problem first, then solve independently
**Rationale**: Prevents the model from creating problems where the derivation doesn't support the stated answer. The solver sees only the problem, not how it was created.
**Tradeoff**: 2x generation API calls, but much higher answer accuracy.

### 2. Self-Reported Confidence Check
**Decision**: Require solver confidence ≥0.95 to proceed
**Rationale**: Low confidence indicates ambiguous or ill-posed problems that will likely fail validation anyway.
**Tradeoff**: Relies on model's self-assessment (not externally validated), but works as a quick filter.

### 3. Single-Threaded Async Architecture
**Decision**: Use asyncio with concurrent I/O (not multi-threading)
**Rationale**: Pipeline is I/O-bound (waiting on API calls). Async is efficient for this and avoids threading complexity.
**Tradeoff**: No CPU parallelism, but CPU work (JSON parsing) is minimal.

### 4. Backfill Loop for Target Count
**Decision**: Keep generating until `final.jsonl` reaches target count
**Rationale**: Users expect exactly N QAs when they request N. Phase 2 rejections shouldn't result in fewer outputs.
**Tradeoff**: Variable runtime, but guaranteed output count.

### 5. Phase 2 GPT-4 Validation (Beyond Spec)
**Decision**: Add derivation audit + blind answer validation after cross-check
**Rationale**: Cross-check validates solvability, but not reasoning quality. GPT-4 audit catches flawed derivations.
**Tradeoff**: Significant additional API cost and time, but higher quality outputs. This is optional per spec.

### 6. JSONL Output Format
**Decision**: JSONL (one JSON object per line)
**Rationale**: Streaming-friendly, handles partial failures gracefully, standard ML dataset format.
**Tradeoff**: Slightly larger than compressed formats, but human-readable.

### 7. LLM-as-Judge for Correctness
**Decision**: Use Claude Sonnet as judge for rubric-based grading
**Rationale**: Physics correctness requires domain expertise; string matching won't work for equivalent symbolic expressions.
**Tradeoff**: Additional API cost per question.

### 8. Dataset-Level Qwen Constraint
**Decision**: Track "easy" questions at the dataset level, allow up to 5%
**Rationale**: Per the spec, this is a dataset-level constraint.
**Tradeoff**: Slightly more complex bookkeeping, but matches spec exactly.

### 9. htmx for Web UI
**Decision**: htmx instead of React/Vue
**Rationale**: Lightweight, no build step, server-rendered HTML with interactivity.
**Tradeoff**: Less rich interactivity, but much simpler to maintain.

### 10. Checkpoint System
**Decision**: Write QAs to disk immediately after Phase 1 validation
**Rationale**: Long-running generation shouldn't lose progress on failure.
**Tradeoff**: Small disk I/O overhead.

## What I'd Do With More Time

1. **Few-Shot Examples**: Include 1-2 examples of questions that passed all validation in the generation prompt to anchor output quality

2. **Human-in-the-Loop**: Add optional human review stage for borderline cases

3. **HuggingFace Export**: Direct export to HuggingFace datasets format for easy sharing

4. **Difficulty Gradation**: Tag questions with difficulty levels (easy/medium/hard graduate-level)

5. **Cost Optimization**: Implement caching for repeated model calls, smarter sampling strategies

6. **Better Error Analysis**: Detailed analysis of why questions fail each validation stage

7. **Prometheus Metrics**: Add observability for production deployment

## Already Implemented

- **Two-Step Generation**: Problem generation separate from solution to prevent answer bias

- **Backfill Loop**: Pipeline automatically generates more QAs if Phase 2 rejects some, ensuring target count is reached

- **Phase 2 GPT-4 Validation**: Derivation audit (5 passes) + blind answer validation (5 passes) for high-quality outputs

- **Full Pipeline Re-validation**: Regenerated QAs go through the entire pipeline again (Qwen → Cross-check → Phase 2)

- **Granular Retry Limits**: Each validation stage has its own configurable retry limit (`--schema-retries`, `--qwen-retries`, `--crosscheck-retries`, `--final-retries`). Set to 0 for unlimited retries.

- **Mixed Topic Mode**: `--mix-topics` flag distributes different physics topics across workers for diverse datasets

- **Adaptive Difficulty Feedback Loop**: Validation failure reasons are fed back into `regenerate_with_feedback()` to improve subsequent attempts

- **Topic Diversity Tracking**: `TopicSampler` tracks used subtopics and prefers unused ones

- **Symbolic Math Verification**: SymPy-based programmatic verification for dimensional analysis, numerical sanity, and expression equivalence

- **Problem Completeness Checking**: AI-based validation that problems are well-posed (all variables defined, boundary conditions specified, unambiguous physical context)

- **Multi-Model Sanity Check (Optional Council)**: Consensus-based sanity checking using multiple AI models (Claude + GPT-4 + Gemini) with configurable consensus modes (`all`, `majority`, `any`) to catch model-specific blind spots

- **10-Point Rubric-Based Grading**: Structured grading with 3 points for final answer and 7 points for key steps

- **Robust Rate Limit Handling**: Retries with exponential backoff ensures reliable generation despite rate limits

## Project Structure

```
takeHome-AQ/
├── src/
│   ├── __init__.py
│   ├── main.py                    # CLI entry point
│   ├── config.py                  # Settings & environment
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   ├── api/
│   │   └── client.py              # OpenRouter async client
│   ├── generation/
│   │   ├── topics.py              # Physics topic sampling
│   │   └── generator.py           # Two-step QA generation
│   ├── validation/
│   │   ├── __init__.py            # Validation exports
│   │   ├── schema.py              # Schema validation
│   │   ├── completeness_check.py  # Problem well-posedness
│   │   ├── symbolic_math.py       # SymPy verification
│   │   ├── sanity_check.py        # Physics sanity checks
│   │   ├── qwen_check.py          # Qwen constraint
│   │   ├── crosscheck.py          # Multi-model check
│   │   ├── correctness.py         # LLM-as-judge
│   │   ├── derivation_audit.py    # GPT-4 reasoning audit
│   │   └── final_answer.py        # GPT-4 blind validation
│   ├── orchestration/
│   │   └── pipeline.py            # Pipeline runner with backfill
│   └── web/
│       ├── app.py                 # FastAPI app
│       ├── export.py              # Excel export utilities
│       ├── templates/             # Jinja2 templates
│       └── static/                # CSS
├── tests/
│   ├── test_schemas.py
│   ├── test_validators.py
│   └── test_generation.py
├── output/                        # Generated datasets
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_schemas.py -v
```

## License

MIT
