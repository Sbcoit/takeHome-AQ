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
│  - Parallel or Sequential execution modes                       │
│  - Coordinates generation → validation → output                  │
│  - Checkpointing for crash recovery                             │
│  - Statistics tracking                                          │
│  - Retries indefinitely until target count is reached           │
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
└───────────────┘     │ FinalAnswerValid│     └────────────────┘
                      └─────────────────┘
                                │
                      ┌─────────────────┐
                      │ OpenRouter API  │
                      │    Client       │
                      └─────────────────┘
```

### Components

1. **Generation Pipeline** (`src/generation/`)
   - `TopicSampler`: Samples from 12 physics topics with 100+ subtopics
   - `QAGenerator`: Uses Claude to generate questions with carefully crafted prompts

2. **Validation Pipeline** (`src/validation/`)
   - `SchemaValidator`: Validates structure and format
   - `QwenConstraintValidator`: Ensures questions aren't too easy
   - `CrossCheckValidator`: Tests with multiple frontier models
   - `CorrectnessJudge`: Verifies physics correctness
   - `FinalAnswerValidator`: Two-step blind validation (5x consecutive passes required)

3. **Orchestration** (`src/orchestration/`)
   - `ParallelPipelineRunner`: Coordinates the entire workflow with parallel workers
   - Spawns N workers (where N = target QA count) that run concurrently
   - Handles checkpointing, statistics, and error recovery
   - Automatically spawns replacement workers when failures occur

4. **Interfaces**
   - CLI: `src/main.py` (typer-based)
   - Web UI: `src/web/` (FastAPI + htmx)

## Validation Logic

The system implements the **four required validation rules** from the spec:

### 1. No-Images Rule
- `response_images` must be empty for every data point
- No images in questions or anywhere in content

### 2. Qwen Constraint (Anti-"Too Easy" Filter) - DATASET-LEVEL
Per spec: *"The proportion of cases where Qwen3-max-thinking passes at least 4 out of 5 attempts must not exceed 5%."*

This is a **dataset-level** constraint:
- For each question, sample `qwen/qwen3-max` (with thinking mode) 5 times
- Grade each response using rubric-based evaluation
- If Qwen passes 4+ out of 5 attempts, the question is marked as "easy"
- **Dataset constraint**: No more than 5% of questions can be "easy"

**Post-generation filtering**: In parallel mode, all workers complete their validation independently. After all workers finish, the 5% easy question filter is applied:
- If ≤5% of questions are "easy", all are kept
- If >5% are "easy", excess easy questions are randomly dropped to meet the threshold

For example, in a 50-question dataset:
- Up to 2 questions can be "easy" (Qwen 4+/5 passes)
- If 5 easy questions were generated, 3 would be randomly dropped

### 3. Frontier Model Cross-Check (per-question)
Per spec, test each question with 4 models × 5 samples = 20 total attempts:
- `deepseek/deepseek-v3-0324` (deepseek-v3.2)
- `openai/o4-mini-2025-04-16`
- `google/gemini-2.5-pro-preview-05-06` (gemini-3-pro)
- `x-ai/grok-4-0709`

**Pass criteria (per-question)**:
- At least 2 models must have accuracy > 0 (i.e., at least one correct answer among their 5 attempts)
- Total correct across 20 attempts must be ≥5 (i.e., overall accuracy ≥ 25%)

### 4. Dataset-Level Correctness Target
Per spec: *"Across the accepted dataset, the accuracy should be higher than 90%"*

Uses LLM-as-judge to verify each question:
- The query is well-posed and internally consistent
- The response_answer is correct
- The response_reasoning is correct and supports the answer

**Pass criteria**: Dataset-wide accuracy must exceed 90%.

### 5. Final Answer Validation (Post-Generation Two-Step Blind Check)

After all QA pairs pass the initial validation (Schema → Qwen → Cross-check), they go through a rigorous **Final Answer Validation** phase:

**Two-Step Blind Judgment Process** (repeated 5 times per QA):
1. **Step 1 - Blind Solve**: Claude Opus sees **ONLY the question** (no answer, no solution) and independently solves the problem from first principles
2. **Step 2 - Compare**: Claude then compares its blind-derived answer to the provided answer to check equivalence

**Why Blind?**: This eliminates confirmation bias. Claude derives its answer before ever seeing the provided answer, ensuring truly independent verification.

**Pass Criteria**: A QA pair must pass **all 5 consecutive** blind judgments to be accepted.

**On Failure - Full Pipeline Re-validation**:
If final validation fails:
1. Extract feedback from the failed judgments (what answer Claude derived, why they don't match)
2. Regenerate the QA pair with the corrected answer
3. **Re-run the FULL validation pipeline** (Qwen check → Cross-check → Final 5x validation)
4. Repeat up to 15 times until the QA passes or is discarded

This ensures that any corrected answer doesn't accidentally become "too easy" or fail other validation stages.

### How "Correct" Is Scored

The system uses a **10-point rubric-based grading system**:

**Point Allocation**:
- **Final Answer**: 3 points (correct answer is required but not sufficient)
- **Key Steps**: 7 points (3-7 key steps, each worth 1-4 points)

**Pass Criteria**:
- **Qwen (STRICT)**: Must score ≥7/10 AND ≥2/3 on final answer
- **Cross-check (GENEROUS)**: Must score ≥6/10 AND ≥2/3 on final answer

**Grading philosophy for cross-check (generous)**:
- Accept equivalent answers (different but mathematically equal forms, reasonable rounding)
- Accept valid alternative approaches that reach the same answer
- Focus on whether the physics is RIGHT, not exact string matching
- Minor calculation errors are acceptable if the method is correct

**Grading philosophy for Qwen check (strict)**:
- Do NOT give benefit of the doubt
- Missing steps = missing points
- Wrong reasoning with right answer = max 3 points (answer only)
- Right reasoning with wrong answer = max 7 points

The judge uses low temperature (0.1) for consistent evaluation and outputs structured JSON with detailed explanations.

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

### Running via CLI

```bash
# Generate 50 QA pairs in parallel (default)
python -m src.main generate

# Generate custom count with parallel workers
python -m src.main generate --count 100 --output output/my_dataset.jsonl

# Generate 5 QAs for a specific topic
python -m src.main generate --count 5 --topic quantum_mechanics

# Generate with mixed topics (each worker gets a different topic)
python -m src.main generate --count 5 --mix-topics

# Customize individual retry limits (defaults: schema=3, qwen=5, crosscheck=5, final=10)
python -m src.main generate --count 5 --schema-retries 5 --qwen-retries 8 --crosscheck-retries 8 --final-retries 15

# Set unlimited retries for specific stages (0 = unlimited)
python -m src.main generate --count 5 --qwen-retries 0 --final-retries 0

# With verbose logging
python -m src.main generate --count 2 --verbose

# Show available topics
python -m src.main topics

# Show system info
python -m src.main info

# Validate an existing dataset
python -m src.main validate output/dataset.jsonl
```

**Parallel Execution**: The pipeline spawns N workers (where N = requested QA count) that run concurrently. Each worker independently generates and validates one QA pair. Failed workers are automatically replaced until the target count is reached.

**Retry Behavior** (granular per-stage limits):
- **Schema validation**: Up to 3 retries. Configure with `--schema-retries`.
- **Qwen difficulty check**: Up to 5 retries. Configure with `--qwen-retries`.
- **Cross-check validation**: Up to 5 retries. Configure with `--crosscheck-retries`.
- **Final 5x blind validation**: Up to 10 retries. Configure with `--final-retries`.
- Set any limit to 0 for unlimited retries. All flags can be used together.

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
- Browse generated QA pairs
- Detailed view for each question

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

## Design Decisions & Tradeoffs

### 1. Dataset-Level Qwen Constraint
**Decision**: Track "easy" questions at the dataset level, allow up to 5%
**Rationale**: Per the spec, this is a dataset-level constraint. Early questions can be easy; we only reject once we'd exceed 5%.
**Tradeoff**: Slightly more complex bookkeeping, but matches spec exactly.

### 2. JSONL Output Format
**Decision**: JSONL (one JSON object per line)
**Rationale**: Streaming-friendly, handles partial failures gracefully, standard ML dataset format.
**Tradeoff**: Slightly larger than compressed formats, but human-readable.

### 3. LLM-as-Judge for Correctness
**Decision**: Use a strong model (Claude Sonnet) as judge
**Rationale**: Physics correctness requires domain expertise; string matching won't work for equivalent symbolic expressions.
**Tradeoff**: Additional API cost per question.

### 4. Parallel Worker Architecture
**Decision**: Full async with parallel worker pool using asyncio
**Rationale**: Spawning N workers for N requested QAs maximizes throughput by running all pipelines concurrently.
**Tradeoff**: Higher peak API usage, but dramatically faster overall execution. Rate limiting is automatically scaled.

### 5. Post-Generation Filtering for 5% Easy Threshold
**Decision**: Accept all valid questions during parallel generation, filter at the end
**Rationale**: In parallel mode, workers can't coordinate in real-time. Post-filtering ensures we meet the 5% threshold.
**Tradeoff**: May generate slightly more questions than needed if many are "easy".

### 6. htmx for Web UI
**Decision**: htmx instead of React/Vue
**Rationale**: Lightweight, no build step, server-rendered HTML with interactivity.
**Tradeoff**: Less rich interactivity, but much simpler to maintain.

### 7. Checkpoint System
**Decision**: Periodic checkpointing with recovery
**Rationale**: Long-running generation shouldn't lose all progress on failure.
**Tradeoff**: Small disk I/O overhead.

### 8. High Retry Count for Rate Limits
**Decision**: Default 30 retries with exponential backoff (capped at 30s)
**Rationale**: OpenRouter rate limits can cause transient failures, especially with Qwen. Higher retries ensure 30+ QA batches complete reliably.
**Tradeoff**: Slower recovery from permanent failures, but much better reliability for large batches.

### 9. Granular Per-Stage Retry Limits
**Decision**: Separate configurable retry limits for each validation stage (schema=3, qwen=5, crosscheck=5, final=10)
**Rationale**: Different stages have different failure modes. Schema failures are rare but should fail fast. Qwen difficulty and cross-check answer validation may need more attempts. Final validation may require the most retries since it's the strictest.
**Tradeoff**: More configuration options, but allows fine-tuned control over generation behavior. Set any limit to 0 for unlimited retries.

### 10. Two-Step Blind Final Validation
**Decision**: Claude solves the problem blind (without seeing the answer), then compares its solution
**Rationale**: Eliminates confirmation bias. If Claude sees the answer first, it may unconsciously validate it even if wrong.
**Tradeoff**: 2x API calls per judgment (solve + compare), but much more rigorous verification.

### 11. Full Pipeline Re-validation After Regeneration
**Decision**: Regenerated QAs go through the full pipeline again (Qwen → Cross-check → Final 5x)
**Rationale**: A corrected answer might accidentally become "too easy" or fail other validation stages.
**Tradeoff**: More API calls, but ensures regenerated QAs meet all quality criteria.

### 12. Configurable Retry Limits with Optional Unlimited Mode
**Decision**: Each validation stage has its own configurable retry limit, with 0 meaning unlimited
**Rationale**: Users can balance between guaranteed results (unlimited) and fail-fast behavior (limited retries) per stage.
**Tradeoff**: Unlimited mode may take longer, but guarantees results. Limited mode is faster but may skip difficult questions.

## What I'd Do With More Time

1. **Few-Shot Examples**: Include 1-2 examples of questions that passed all validation in the generation prompt to anchor output quality

2. **Human-in-the-Loop**: Add optional human review stage for borderline cases

3. **HuggingFace Export**: Direct export to HuggingFace datasets format for easy sharing

4. **Difficulty Gradation**: Tag questions with difficulty levels (easy/medium/hard graduate-level)

5. **Cost Optimization**: Implement caching for repeated model calls, smarter sampling strategies

6. **Better Error Analysis**: Detailed analysis of why questions fail each validation stage

7. **Prometheus Metrics**: Add observability for production deployment

## Already Implemented

- **Parallel Worker Execution**: Spawns N concurrent workers for N requested QAs, dramatically improving throughput. Each worker runs the full pipeline independently, with automatic replacement of failed workers.

- **Granular Retry Limits**: Each validation stage has its own configurable retry limit (`--schema-retries`, `--qwen-retries`, `--crosscheck-retries`, `--final-retries`). Set to 0 for unlimited retries. Data is written to disk immediately so nothing is lost on failure.

- **Post-Generation 5% Easy Filter**: Easy questions are tracked during parallel generation and filtered at the end to meet the 5% dataset-level threshold.

- **Mixed Topic Mode**: `--mix-topics` flag distributes different physics topics across parallel workers for diverse datasets.

- **Adaptive Difficulty Feedback Loop**: Validation failure reasons are fed back into `regenerate_with_feedback()` to improve subsequent attempts (e.g., "Question was too easy - Qwen solved it 4/5 times")

- **Topic Diversity Tracking**: `TopicSampler` tracks used subtopics and prefers unused ones via `prefer_diverse=True`, with coverage statistics available via `get_coverage_stats()`

- **10-Point Rubric-Based Grading**: Structured grading with 3 points for final answer and 7 points for key steps (3-7 steps). Pass criteria differ for Qwen (strict: ≥7/10) vs cross-check (generous: ≥6/10).

- **Robust Rate Limit Handling**: 30 retries with exponential backoff (capped at 30s) ensures reliable generation of 30+ QA batches despite OpenRouter rate limits.

- **Escalating Difficulty Hints**: When questions are too easy, regeneration includes progressively harder requirements like "require perturbation theory" or "add topological considerations".

- **Two-Step Blind Final Validation**: After cross-check validation, Claude Opus performs a rigorous two-step blind judgment 5 times:
  1. Blind solve (sees only the question, derives answer independently)
  2. Compare (checks if blind-derived answer matches provided answer)

  This eliminates confirmation bias and ensures truly independent verification.

- **Full Pipeline Re-validation on Failure**: When final validation fails, the regenerated QA goes through the entire pipeline again (Qwen → Cross-check → Final 5x) to ensure correctness, difficulty, and validity are all maintained.

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
│   │   └── generator.py           # QA generation
│   ├── validation/
│   │   ├── schema.py              # Schema validation
│   │   ├── qwen_check.py          # Qwen constraint
│   │   ├── crosscheck.py          # Multi-model check
│   │   ├── correctness.py         # LLM-as-judge
│   │   └── final_answer.py        # Two-step blind validation
│   ├── orchestration/
│   │   └── pipeline.py            # Pipeline runner
│   └── web/
│       ├── app.py                 # FastAPI app
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
