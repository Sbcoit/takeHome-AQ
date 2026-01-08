"""FastAPI web application for the Physics QA Generator."""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Get the directory containing this file
BASE_DIR = Path(__file__).parent

app = FastAPI(
    title="Physics QA Generator",
    description="Synthetic Graduate-Level Physics QA System",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Templates - create and immediately register custom filters
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _texify_filter(text):
    """Add $ delimiters around LaTeX expressions for MathJax rendering.

    Strategy: Find math regions and wrap them completely. A math region is
    initiated by a LaTeX command, variable with subscript/superscript, or
    begin/end environment, and extends through related math content.
    """
    if not text:
        return ""

    text = str(text)

    # If already has $ delimiters, return as-is (already formatted for MathJax)
    if '$' in text:
        return text

    # Check if text contains any LaTeX patterns
    if not re.search(r'\\[a-zA-Z]|[a-zA-Z]_[{a-zA-Z0-9\\]|[a-zA-Z]\^[{a-zA-Z0-9\\]', text):
        return text

    def find_matching_end(s, start, env_name):
        """Find matching \\end{env_name} for a \\begin{env_name}."""
        open_pattern = f'\\begin{{{env_name}}}'
        close_pattern = f'\\end{{{env_name}}}'
        depth = 1
        i = start
        while i < len(s) and depth > 0:
            if s[i:i+len(open_pattern)] == open_pattern:
                depth += 1
                i += len(open_pattern)
            elif s[i:i+len(close_pattern)] == close_pattern:
                depth -= 1
                if depth == 0:
                    return i + len(close_pattern)
                i += len(close_pattern)
            else:
                i += 1
        return i

    def find_brace_content(s, pos):
        """Find content within braces starting at pos (which should be '{')."""
        if pos >= len(s) or s[pos] != '{':
            return pos
        depth = 1
        i = pos + 1
        while i < len(s) and depth > 0:
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
            i += 1
        return i

    def read_latex_command(s, pos):
        """Read a LaTeX command and all its arguments. Returns end position."""
        if pos >= len(s) or s[pos] != '\\':
            return pos
        i = pos + 1
        # Read command name
        while i < len(s) and s[i].isalpha():
            i += 1
        # Read all brace arguments
        while i < len(s) and s[i] == '{':
            i = find_brace_content(s, i)
        return i

    def read_math_token(s, pos):
        """Read a single math token (command, variable, number, operator).
        Returns (end_pos, is_math_token)."""
        if pos >= len(s):
            return pos, False

        c = s[pos]

        # LaTeX command
        if c == '\\' and pos + 1 < len(s) and s[pos + 1].isalpha():
            return read_latex_command(s, pos), True

        # Variable or number, possibly with sub/superscript
        if c.isalnum():
            i = pos
            while i < len(s) and s[i].isalnum():
                i += 1
            # Check for subscript/superscript
            while i < len(s) and s[i] in '_^':
                i += 1
                if i < len(s):
                    if s[i] == '{':
                        i = find_brace_content(s, i)
                    elif s[i] == '\\':
                        i = read_latex_command(s, i)
                    elif s[i].isalnum() or s[i] in '+-':
                        i += 1
            return i, True

        # Math operators and delimiters
        if c in '=+-*/^_<>|&()[]{}.,;:!':
            return pos + 1, True

        return pos, False

    def extend_math_region(s, start):
        """Extend a math region as far as it goes. Returns end position."""
        i = start
        n = len(s)

        while i < n:
            # Skip whitespace within math
            ws_start = i
            while i < n and s[i] == ' ':
                i += 1

            if i >= n:
                # End with trailing whitespace removed
                return ws_start

            # Check if next thing continues math
            c = s[i]

            # \begin{...} environment - include the whole thing
            if s[i:].startswith('\\begin{'):
                begin_match = re.match(r'\\begin\{(\w+)\}', s[i:])
                if begin_match:
                    env_name = begin_match.group(1)
                    open_cmd = f'\\begin{{{env_name}}}'
                    i = find_matching_end(s, i + len(open_cmd), env_name)
                    continue

            # \left...\right pairs
            if s[i:].startswith('\\left'):
                depth = 1
                i += 5
                while i < n and depth > 0:
                    if s[i:].startswith('\\left'):
                        depth += 1
                        i += 5
                    elif s[i:].startswith('\\right'):
                        depth -= 1
                        i += 6
                        if depth == 0:
                            # Include delimiter after \right
                            if i < n and s[i] in '|)]}>.':
                                i += 1
                            elif i < n and s[i] == '\\':
                                j = i + 1
                                while j < n and s[j].isalpha():
                                    j += 1
                                i = j
                    else:
                        i += 1
                continue

            # LaTeX command
            if c == '\\' and i + 1 < n and s[i + 1].isalpha():
                i = read_latex_command(s, i)
                # Include subscripts/superscripts after command
                while i < n and s[i] in '_^':
                    i += 1
                    if i < n:
                        if s[i] == '{':
                            i = find_brace_content(s, i)
                        elif s[i] == '\\':
                            i = read_latex_command(s, i)
                        elif s[i].isalnum() or s[i] in '+-':
                            i += 1
                continue

            # Math operators - continue
            if c in '=+-*/<>':
                i += 1
                continue

            # Variables/numbers that are part of equation
            if c.isalnum():
                # Check if this looks like start of natural language (multiple words)
                word_end = i
                while word_end < n and s[word_end].isalpha():
                    word_end += 1
                word = s[i:word_end]
                # Common words that signal end of math
                stop_words = {'and', 'or', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                              'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                              'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                              'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                              'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                              'through', 'during', 'before', 'after', 'above', 'below',
                              'between', 'under', 'again', 'further', 'then', 'once',
                              'here', 'there', 'when', 'where', 'why', 'how', 'all',
                              'each', 'few', 'more', 'most', 'other', 'some', 'such',
                              'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                              'too', 'very', 'just', 'also', 'now', 'derive', 'using',
                              'use', 'find', 'calculate', 'compute', 'show', 'prove',
                              'determine', 'evaluate', 'consider', 'given', 'assume',
                              'let', 'suppose', 'if', 'then', 'thus', 'hence', 'therefore',
                              'since', 'because', 'although', 'while', 'whereas'}
                if word.lower() in stop_words:
                    return ws_start if ws_start < i else i
                end, is_math = read_math_token(s, i)
                i = end
                continue

            # Punctuation that might end expression - check context
            if c in '.,;:':
                # Check if more math follows after punctuation + space
                j = i + 1
                while j < n and s[j] == ' ':
                    j += 1
                # If next non-space starts math, include the punctuation
                if j < n and (s[j] == '\\' or s[j].isalnum() or s[j] in '=+-*/(<'):
                    # Check if this really looks like continuing math
                    if j < n and s[j] == '\\':
                        i = j
                        continue
                    elif j < n and s[j].isalnum():
                        # Only continue if followed by math operators or subscripts
                        k = j
                        while k < n and s[k].isalnum():
                            k += 1
                        if k < n and s[k] in '_^=+-*/<>\\':
                            i = j
                            continue
                # End the expression here (don't include the punctuation)
                return i

            # Anything else ends the math region
            return ws_start if ws_start < i else i

        return i

    # Process text
    result = []
    i = 0
    n = len(text)

    while i < n:
        # Check for \begin{...} environments at top level
        begin_match = re.match(r'\\begin\{(\w+)\}', text[i:])
        if begin_match:
            env_name = begin_match.group(1)
            open_cmd = f'\\begin{{{env_name}}}'
            end_pos = find_matching_end(text, i + len(open_cmd), env_name)
            expr = text[i:end_pos]
            result.append('$' + expr + '$')
            i = end_pos
            continue

        # Check for \left at top level
        if text[i:].startswith('\\left'):
            start = i
            end_pos = extend_math_region(text, i)
            expr = text[start:end_pos]
            result.append('$' + expr + '$')
            i = end_pos
            continue

        # LaTeX command starts a math region
        if text[i] == '\\' and i + 1 < n and text[i + 1].isalpha():
            start = i
            end_pos = extend_math_region(text, i)
            expr = text[start:end_pos].strip()
            if expr:
                result.append('$' + expr + '$')
            i = end_pos
            continue

        # Variable with subscript/superscript starts a math region
        if text[i].isalpha():
            j = i
            while j < n and text[j].isalnum():
                j += 1
            if j < n and text[j] in '_^':
                start = i
                end_pos = extend_math_region(text, i)
                expr = text[start:end_pos].strip()
                if expr:
                    result.append('$' + expr + '$')
                i = end_pos
                continue
            else:
                result.append(text[i])
                i += 1

        else:
            result.append(text[i])
            i += 1

    output = ''.join(result)

    # Clean up: remove empty math blocks, merge adjacent ones
    output = re.sub(r'\$\s*\$', '', output)
    output = re.sub(r'\$\s+\$', ' ', output)

    return output


# Register filter IMMEDIATELY after templates object creation
templates.env.filters['texify'] = _texify_filter


# Custom logging handler that captures logs to our state
class WebUILogHandler(logging.Handler):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def emit(self, record):
        try:
            msg = self.format(record)
            self.state.add_log(msg)
        except Exception:
            pass


# Global state for the running pipeline
class PipelineState:
    def __init__(self):
        self.is_running = False
        self.run_id: Optional[str] = None
        self.target_count: int = 0
        self.current_count: int = 0
        self.stats: Dict[str, Any] = {}
        self.recent_qa: List[Dict[str, Any]] = []
        self.logs: List[str] = []
        self.error: Optional[str] = None
        self.topics: Optional[List[str]] = None
        self.output_path: str = "output/dataset.jsonl"
        # Per-worker log tracking
        self.worker_logs: Dict[int, List[str]] = {}
        self.active_workers: Set[int] = set()
        # Stop control
        self.stop_requested: bool = False
        self.pipeline_task: Optional[asyncio.Task] = None  # Reference to running task
        # Phase tracking (1 = generation, 2 = audit)
        self.phase: int = 1
        self.phase1_count: int = 0  # QAs that passed Phase 1
        self.phase2_validated: int = 0  # QAs validated in Phase 2
        self.phase2_passed: int = 0  # QAs that passed Phase 2 (final count)

    def reset(self):
        self.is_running = False
        self.run_id = None
        self.target_count = 0
        self.current_count = 0
        self.stats = {}
        self.recent_qa = []
        self.logs = []
        self.error = None
        self.topics = None
        self.output_path = "output/dataset.jsonl"
        self.schema_retries = None
        self.sanity_retries = None
        self.qwen_retries = None
        self.crosscheck_retries = None
        self.final_retries = None
        # Reset per-worker tracking
        self.worker_logs = {}
        self.active_workers = set()
        # Reset stop control
        self.stop_requested = False
        self.pipeline_task = None
        # Reset phase tracking
        self.phase = 1
        self.phase1_count = 0
        self.phase2_validated = 0
        self.phase2_passed = 0

    def add_log(self, message: str):
        self.logs.append(message)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

        # Parse worker ID from message and track per-worker logs
        match = re.search(r'\[Worker (\d+)\]', message)
        if match:
            worker_id = int(match.group(1))
            self.active_workers.add(worker_id)
            if worker_id not in self.worker_logs:
                self.worker_logs[worker_id] = []
            self.worker_logs[worker_id].append(message)
            # Keep per-worker logs bounded
            if len(self.worker_logs[worker_id]) > 100:
                self.worker_logs[worker_id] = self.worker_logs[worker_id][-100:]

    def to_dict(self) -> Dict[str, Any]:
        # Calculate progress percent based on phase
        if self.phase == 1:
            progress_percent = (
                int(self.current_count / self.target_count * 100)
                if self.target_count > 0 else 0
            )
        else:
            # Phase 2: progress based on QAs validated out of phase1_count
            progress_percent = (
                int(self.phase2_validated / self.phase1_count * 100)
                if self.phase1_count > 0 else 0
            )

        return {
            "is_running": self.is_running,
            "run_id": self.run_id,
            "target_count": self.target_count,
            "current_count": self.current_count,
            "progress_percent": progress_percent,
            "stats": self.stats,
            "recent_qa_count": len(self.recent_qa),
            "error": self.error,
            "topics": self.topics,
            "output_path": self.output_path,
            "active_workers": sorted(self.active_workers),
            # Phase tracking
            "phase": self.phase,
            "phase1_count": self.phase1_count,
            "phase2_validated": self.phase2_validated,
            "phase2_passed": self.phase2_passed,
        }


pipeline_state = PipelineState()

# Set up logging to capture to WebUI
log_handler = WebUILogHandler(pipeline_state)
log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S'))
log_handler.setLevel(logging.INFO)

# Only add to root logger - child loggers will propagate up
root_logger = logging.getLogger()
root_logger.addHandler(log_handler)
root_logger.setLevel(logging.INFO)


class GenerateRequest(BaseModel):
    count: int = 50
    output_path: str = "output/dataset.jsonl"
    topics: Optional[List[str]] = None
    # Granular retry limits (None = use default, 0 = unlimited)
    schema_retries: Optional[int] = None
    sanity_retries: Optional[int] = None
    qwen_retries: Optional[int] = None
    crosscheck_retries: Optional[int] = None
    final_retries: Optional[int] = None


def progress_callback(data: Dict[str, Any]):
    """Callback for pipeline progress updates."""
    pipeline_state.current_count = data.get("valid_count", 0)
    pipeline_state.stats = data.get("stats", {})

    # Handle phase tracking (1 = generation only, 2 = audit only, "streaming" = both concurrent)
    phase = data.get("phase", 1)
    pipeline_state.phase = phase

    if phase == 2 or phase == "streaming":
        pipeline_state.phase1_count = data.get("phase1_count", 0)
        pipeline_state.phase2_validated = data.get("phase2_validated", 0)
        pipeline_state.phase2_passed = data.get("phase2_passed", 0)


async def run_pipeline_async(
    count: int,
    output_path: str,
    topics: Optional[List[str]] = None,
    schema_retries: Optional[int] = None,
    sanity_retries: Optional[int] = None,
    qwen_retries: Optional[int] = None,
    crosscheck_retries: Optional[int] = None,
    final_retries: Optional[int] = None,
):
    """Run the pipeline in the background."""
    from dotenv import load_dotenv

    load_dotenv()

    logger = logging.getLogger(__name__)
    logger.info(f"Starting generation: {count} QA pairs")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        pipeline_state.error = "OPENROUTER_API_KEY not set"
        pipeline_state.is_running = False
        return

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    def check_stop_requested() -> bool:
        """Callback to check if stop was requested."""
        return pipeline_state.stop_requested

    try:
        from ..orchestration.pipeline import run_pipeline

        result = await run_pipeline(
            api_key=api_key,
            target_count=count,
            output_path=output_path,
            progress_callback=progress_callback,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            topics=topics,
            schema_max_retries=schema_retries,
            sanity_check_max_retries=sanity_retries,
            qwen_max_retries=qwen_retries,
            crosscheck_max_retries=crosscheck_retries,
            final_validation_max_retries=final_retries,
            stop_check=check_stop_requested,
        )

        pipeline_state.stats = result.get("stats", {})
        logger.info(f"Pipeline complete: {pipeline_state.current_count} valid pairs")

    except asyncio.CancelledError:
        logger.info("Pipeline cancelled by user")
        pipeline_state.error = None  # Not an error, just stopped

    except Exception as e:
        pipeline_state.error = str(e)
        logger.error(f"Pipeline failed: {e}")

    finally:
        pipeline_state.is_running = False
        pipeline_state.pipeline_task = None


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    from ..models.schemas import PhysicsTopic

    topics = [{"value": t.name.lower(), "label": t.value} for t in PhysicsTopic]

    # Get output files for initial render
    output_dir = Path("output")
    files = []
    if output_dir.exists():
        for f in sorted(output_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = f.stat()
            with open(f) as fp:
                line_count = sum(1 for _ in fp)
            files.append({
                "name": f.name,
                "path": str(f),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "count": line_count,
                "is_current": str(f) == pipeline_state.output_path,
            })

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "state": pipeline_state.to_dict(),
            "topics": topics,
            "workers": sorted(pipeline_state.active_workers),
            "logs": pipeline_state.logs[-100:],
            "files": files,
        }
    )


@app.get("/api/status")
async def get_status():
    """Get current pipeline status."""
    return JSONResponse(pipeline_state.to_dict())


@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent logs."""
    return JSONResponse({
        "logs": pipeline_state.logs[-limit:],
    })


@app.get("/api/topics")
async def get_topics():
    """Get available physics topics."""
    from ..models.schemas import PhysicsTopic

    topics = [{"value": t.name.lower(), "label": t.value} for t in PhysicsTopic]
    return JSONResponse({"topics": topics})


def generate_unique_output_path(base_dir: str = "output") -> str:
    """Generate a unique timestamped output file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_dir}/dataset_{timestamp}.jsonl"


@app.post("/api/generate")
async def start_generation(
    background_tasks: BackgroundTasks,
    count: int = Form(default=50),
    output_path: Optional[str] = Form(default=None),
    topics: Optional[str] = Form(default=None),
    schema_retries: Optional[str] = Form(default=None),
    sanity_retries: Optional[str] = Form(default=None),
    qwen_retries: Optional[str] = Form(default=None),
    crosscheck_retries: Optional[str] = Form(default=None),
    final_retries: Optional[str] = Form(default=None),
):
    """Start the generation pipeline."""
    if pipeline_state.is_running:
        raise HTTPException(status_code=400, detail="Pipeline is already running")

    pipeline_state.reset()
    pipeline_state.is_running = True
    pipeline_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_state.target_count = count

    # Generate unique timestamped filename if not specified
    if not output_path or output_path == "output/dataset.jsonl":
        output_path = generate_unique_output_path()

    pipeline_state.output_path = output_path

    topic_list = None
    if topics and topics.strip():
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]

    # Parse retry values - empty string means use default (None), "0" means unlimited
    def parse_retry(val: Optional[str]) -> Optional[int]:
        if val is None or val.strip() == "":
            return None  # Use default
        return int(val)

    schema_retries_int = parse_retry(schema_retries)
    sanity_retries_int = parse_retry(sanity_retries)
    qwen_retries_int = parse_retry(qwen_retries)
    crosscheck_retries_int = parse_retry(crosscheck_retries)
    final_retries_int = parse_retry(final_retries)

    pipeline_state.topics = topic_list
    pipeline_state.schema_retries = schema_retries_int
    pipeline_state.sanity_retries = sanity_retries_int
    pipeline_state.qwen_retries = qwen_retries_int
    pipeline_state.crosscheck_retries = crosscheck_retries_int
    pipeline_state.final_retries = final_retries_int

    logger = logging.getLogger(__name__)
    logger.info(f"Starting with retry limits: schema={schema_retries_int}, sanity={sanity_retries_int}, qwen={qwen_retries_int}, crosscheck={crosscheck_retries_int}, final={final_retries_int}")

    # Create and store the task so we can cancel it later
    pipeline_state.pipeline_task = asyncio.create_task(
        run_pipeline_async(
            count,
            output_path,
            topic_list,
            schema_retries_int,
            sanity_retries_int,
            qwen_retries_int,
            crosscheck_retries_int,
            final_retries_int,
        )
    )

    return JSONResponse({
        "status": "started",
        "run_id": pipeline_state.run_id,
    })


@app.post("/api/stop")
async def stop_generation():
    """Immediately stop the pipeline."""
    if not pipeline_state.is_running:
        raise HTTPException(status_code=400, detail="Pipeline is not running")

    logger = logging.getLogger(__name__)

    # Set flag first
    pipeline_state.stop_requested = True

    # Cancel the task immediately
    if pipeline_state.pipeline_task and not pipeline_state.pipeline_task.done():
        pipeline_state.pipeline_task.cancel()
        logger.info("Pipeline stopped immediately")

    pipeline_state.is_running = False
    return JSONResponse({"status": "stopped"})


@app.get("/api/output")
async def get_output(limit: int = 10, offset: int = 0):
    """Get generated QA pairs from the current output file."""
    output_path = Path(pipeline_state.output_path)

    if not output_path.exists():
        return JSONResponse({"items": [], "total": 0})

    items = []
    total = 0

    with open(output_path) as f:
        lines = f.readlines()
        total = len(lines)

        for line in lines[offset:offset + limit]:
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass

    return JSONResponse({
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
    })


@app.get("/qa/{index}", response_class=HTMLResponse)
async def view_qa(request: Request, index: int):
    """View a specific QA pair."""
    output_path = Path(pipeline_state.output_path)

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="No output file found")

    with open(output_path) as f:
        lines = f.readlines()

        if index < 0 or index >= len(lines):
            raise HTTPException(status_code=404, detail="QA pair not found")

        try:
            qa = json.loads(lines[index].strip())
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse QA pair")

    return templates.TemplateResponse(
        "qa_detail.html",
        {
            "request": request,
            "qa": qa,
            "index": index,
            "total": len(lines),
        }
    )


# HTMX partial endpoints
@app.get("/partials/status", response_class=HTMLResponse)
async def partial_status(request: Request):
    """Partial template for status update."""
    return templates.TemplateResponse(
        "partials/status.html",
        {
            "request": request,
            "state": pipeline_state.to_dict(),
        }
    )


@app.get("/partials/logs", response_class=HTMLResponse)
async def partial_logs(request: Request):
    """Partial template for logs."""
    return templates.TemplateResponse(
        "partials/logs.html",
        {
            "request": request,
            "logs": pipeline_state.logs[-100:],
        }
    )


@app.get("/partials/qa-list", response_class=HTMLResponse)
async def partial_qa_list(request: Request, limit: int = 10):
    """Partial template for QA list."""
    output_path = Path(pipeline_state.output_path)
    items = []

    if output_path.exists():
        with open(output_path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines[-limit:]):
                try:
                    qa = json.loads(line.strip())
                    qa["_index"] = len(lines) - limit + i
                    items.append(qa)
                except json.JSONDecodeError:
                    pass

    return templates.TemplateResponse(
        "partials/qa_list.html",
        {
            "request": request,
            "items": items,
        }
    )


@app.get("/api/workers")
async def get_workers():
    """Get list of active workers and their log counts."""
    return JSONResponse({
        "workers": sorted(pipeline_state.active_workers),
        "logs_count": {
            str(w): len(pipeline_state.worker_logs.get(w, []))
            for w in pipeline_state.active_workers
        }
    })


@app.get("/partials/workers", response_class=HTMLResponse)
async def partial_workers(request: Request):
    """Partial template for worker status cards."""
    return templates.TemplateResponse(
        "partials/workers.html",
        {
            "request": request,
            "workers": sorted(pipeline_state.active_workers),
            "state": pipeline_state.to_dict(),
        }
    )


@app.get("/partials/logs/worker/{worker_id}", response_class=HTMLResponse)
async def partial_worker_logs(request: Request, worker_id: int):
    """Partial template for a specific worker's logs."""
    logs = pipeline_state.worker_logs.get(worker_id, [])[-100:]
    return templates.TemplateResponse(
        "partials/logs.html",
        {
            "request": request,
            "logs": logs,
            "worker_filter": worker_id,
        }
    )


@app.get("/api/output-files")
async def list_output_files():
    """List all output files in the output directory."""
    output_dir = Path("output")
    files = []

    if output_dir.exists():
        for f in sorted(output_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = f.stat()
            # Count lines (QA pairs)
            with open(f) as fp:
                line_count = sum(1 for _ in fp)
            files.append({
                "name": f.name,
                "path": str(f),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "count": line_count,
            })

    return JSONResponse({"files": files})


@app.get("/api/output-file/{filename}")
async def get_output_file_contents(filename: str, limit: int = 20, offset: int = 0):
    """Get contents of a specific output file."""
    output_path = Path("output") / filename

    if not output_path.exists() or not output_path.suffix == ".jsonl":
        raise HTTPException(status_code=404, detail="File not found")

    items = []
    total = 0

    with open(output_path) as f:
        lines = f.readlines()
        total = len(lines)

        for i, line in enumerate(lines[offset:offset + limit]):
            try:
                item = json.loads(line.strip())
                item["_index"] = offset + i
                items.append(item)
            except json.JSONDecodeError:
                pass

    return JSONResponse({
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "filename": filename,
    })


@app.get("/partials/output-browser", response_class=HTMLResponse)
async def partial_output_browser(request: Request):
    """Partial template for output file browser."""
    output_dir = Path("output")
    files = []

    if output_dir.exists():
        for f in sorted(output_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = f.stat()
            with open(f) as fp:
                line_count = sum(1 for _ in fp)
            files.append({
                "name": f.name,
                "path": str(f),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "count": line_count,
                "is_current": str(f) == pipeline_state.output_path,
            })

    return templates.TemplateResponse(
        "partials/output_browser.html",
        {
            "request": request,
            "files": files,
        }
    )


@app.get("/partials/output-content/{filename}", response_class=HTMLResponse)
async def get_output_content(request: Request, filename: str):
    """Get output file content as a partial for inline display."""
    output_path = Path("output") / filename

    if not output_path.exists() or not output_path.suffix == ".jsonl":
        raise HTTPException(status_code=404, detail="File not found")

    items = []
    with open(output_path) as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item["_index"] = i
                items.append(item)
            except json.JSONDecodeError:
                pass

    return templates.TemplateResponse(
        "partials/output_content.html",
        {
            "request": request,
            "filename": filename,
            "items": items,
            "total": len(items),
        }
    )


@app.get("/output/{filename}", response_class=HTMLResponse)
async def view_output_file(request: Request, filename: str):
    """View contents of an output file (full page view)."""
    output_path = Path("output") / filename

    if not output_path.exists() or not output_path.suffix == ".jsonl":
        raise HTTPException(status_code=404, detail="File not found")

    items = []
    with open(output_path) as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item["_index"] = i
                items.append(item)
            except json.JSONDecodeError:
                pass

    return templates.TemplateResponse(
        "output_view.html",
        {
            "request": request,
            "filename": filename,
            "items": items,
            "total": len(items),
        }
    )


def create_app():
    """Factory function for the app."""
    return app
