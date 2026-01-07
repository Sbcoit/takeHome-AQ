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
│  - Coordinates generation → validation → output                  │
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
│ (Claude)      │     │ CrossCheckValid │     │ Statistics     │
└───────────────┘     │ CorrectnessJudge│     └────────────────┘
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

3. **Orchestration** (`src/orchestration/`)
   - `PipelineRunner`: Coordinates the entire workflow
   - Handles checkpointing, statistics, and error recovery

4. **Interfaces**
   - CLI: `src/main.py` (typer-based)
   - Web UI: `src/web/` (FastAPI + htmx)

## Validation Logic

### 1. Schema Validation
- All required fields present
- `response_images` is empty array
- Query has minimum 15 words
- Reasoning has minimum 50 words
- Rubric has 3-10 criteria that sum to total_points

### 2. Qwen Constraint (Anti-"Too Easy" Filter)
- Samples `qwen/qwen3-max` 5 times per question
- Grades each response against the rubric
- **Pass criteria**: High-pass rate (4+/5 criteria at 80%+ points) must be ≤ 5%
- This ensures questions aren't trivially solvable

### 3. Frontier Model Cross-Check
Tests with 4 models, 5 samples each (20 total attempts):
- `deepseek/deepseek-v3.2`
- `openai/o4-mini-2025-04-16`
- `google/gemini-3-pro`
- `x-ai/grok-4-0709`

**Pass criteria**:
- At least 2 models must get ≥1 correct answer
- Total correct across 20 attempts must be ≥5 (25% accuracy)

### 4. Correctness Judge
Uses LLM-as-judge to verify:
- Query is well-posed and unambiguous
- Answer is physically correct
- Reasoning is mathematically valid
- Rubric appropriately assesses the problem

**Pass criteria**: Overall correctness score must be acceptable.

### How "Correct" Is Scored

Correctness is determined by the judge model evaluating:

1. **Answer matching**: Final answer matches reference within tolerance (numerical) or is mathematically equivalent (symbolic)
2. **Physics validity**: Core physics principles correctly applied
3. **Mathematical correctness**: All derivation steps are valid

The judge uses low temperature (0.1) for consistent evaluation and outputs structured JSON with scores and explanations.

## How to Run End-to-End

### Prerequisites

- Python 3.11+
- OpenRouter API key (for most models)
- OpenAI API key (for o4-mini model, which is not supported by OpenRouter)

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
#   OPENAI_API_KEY=sk-...  (for o4-mini model)
```

### Running via CLI

```bash
# Generate 50 QA pairs (default)
python -m src.main generate

# Generate custom count
python -m src.main generate --count 100 --output output/my_dataset.jsonl

# With verbose logging
python -m src.main generate --count 5 --verbose

# Show available topics
python -m src.main topics

# Show system info
python -m src.main info

# Validate an existing dataset
python -m src.main validate output/dataset.jsonl
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
- Browse generated QA pairs
- Detailed view for each question

### Output Format

The output is a JSONL file with one JSON object per line:

```json
{
  "query": "Consider a two-dimensional harmonic oscillator...",
  "response_answer": "E = hbar * omega * (n_x + n_y + 1)",
  "response_reasoning": "Starting with the Hamiltonian H = ...",
  "rubric": {
    "total_points": 100,
    "criteria": [
      {"criterion": "Identifies separability", "max_points": 20, "description": "..."},
      {"criterion": "Correct Hamiltonian", "max_points": 25, "description": "..."},
      {"criterion": "Solves eigenvalue problem", "max_points": 30, "description": "..."},
      {"criterion": "Final energy expression", "max_points": 15, "description": "..."},
      {"criterion": "Degeneracy analysis", "max_points": 10, "description": "..."}
    ],
    "passing_threshold": 60
  },
  "response_images": []
}
```

## Design Decisions & Tradeoffs

### 1. Per-Question Qwen Filtering vs Dataset-Level
**Decision**: Per-question filtering
**Rationale**: More aggressive filtering ensures every question meets the difficulty threshold, not just the average.
**Tradeoff**: Higher rejection rate, but higher quality floor.

### 2. JSONL Output Format
**Decision**: JSONL (one JSON object per line)
**Rationale**: Streaming-friendly, handles partial failures gracefully, standard ML dataset format.
**Tradeoff**: Slightly larger than compressed formats, but human-readable.

### 3. LLM-as-Judge for Correctness
**Decision**: Use a strong model (Claude Sonnet) as judge
**Rationale**: Physics correctness requires domain expertise; string matching won't work for equivalent symbolic expressions.
**Tradeoff**: Additional API cost per question.

### 4. Async Architecture
**Decision**: Full async with aiohttp
**Rationale**: Enables parallel API calls for efficiency.
**Tradeoff**: More complex code, but significantly faster.

### 5. htmx for Web UI
**Decision**: htmx instead of React/Vue
**Rationale**: Lightweight, no build step, server-rendered HTML with interactivity.
**Tradeoff**: Less rich interactivity, but much simpler to maintain.

### 6. Checkpoint System
**Decision**: Periodic checkpointing with recovery
**Rationale**: Long-running generation shouldn't lose all progress on failure.
**Tradeoff**: Small disk I/O overhead.

## What I'd Do With More Time

1. **Topic Diversity Tracking**: Ensure balanced coverage across all physics areas within a dataset

2. **Adaptive Difficulty**: Adjust generation prompts based on validation rejection rates

3. **Human-in-the-Loop**: Add optional human review stage for borderline cases

4. **Batch Processing**: Generate multiple questions in parallel, then validate in batch for better throughput

5. **HuggingFace Export**: Direct export to HuggingFace datasets format for easy sharing

6. **Multi-language Support**: Generate questions in multiple languages

7. **Difficulty Gradation**: Tag questions with difficulty levels (easy/medium/hard graduate-level)

8. **Cost Optimization**: Implement caching for repeated model calls, smarter sampling strategies

9. **Better Error Analysis**: Detailed analysis of why questions fail each validation stage

10. **Prometheus Metrics**: Add observability for production deployment

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
│   │   └── correctness.py         # LLM-as-judge
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
