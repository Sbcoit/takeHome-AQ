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

# Templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")


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
        self.qwen_retries = None
        self.crosscheck_retries = None
        self.final_retries = None
        # Reset per-worker tracking
        self.worker_logs = {}
        self.active_workers = set()
        # Reset stop control
        self.stop_requested = False
        self.pipeline_task = None

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
        return {
            "is_running": self.is_running,
            "run_id": self.run_id,
            "target_count": self.target_count,
            "current_count": self.current_count,
            "progress_percent": (
                int(self.current_count / self.target_count * 100)
                if self.target_count > 0 else 0
            ),
            "stats": self.stats,
            "recent_qa_count": len(self.recent_qa),
            "error": self.error,
            "topics": self.topics,
            "output_path": self.output_path,
            "active_workers": sorted(self.active_workers),
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
    qwen_retries: Optional[int] = None
    crosscheck_retries: Optional[int] = None
    final_retries: Optional[int] = None


def progress_callback(data: Dict[str, Any]):
    """Callback for pipeline progress updates."""
    pipeline_state.current_count = data.get("valid_count", 0)
    pipeline_state.stats = data.get("stats", {})


async def run_pipeline_async(
    count: int,
    output_path: str,
    topics: Optional[List[str]] = None,
    schema_retries: Optional[int] = None,
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

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "state": pipeline_state.to_dict(),
            "topics": topics,
            "workers": sorted(pipeline_state.active_workers),
            "logs": pipeline_state.logs[-100:],
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
    qwen_retries_int = parse_retry(qwen_retries)
    crosscheck_retries_int = parse_retry(crosscheck_retries)
    final_retries_int = parse_retry(final_retries)

    pipeline_state.topics = topic_list
    pipeline_state.schema_retries = schema_retries_int
    pipeline_state.qwen_retries = qwen_retries_int
    pipeline_state.crosscheck_retries = crosscheck_retries_int
    pipeline_state.final_retries = final_retries_int

    logger = logging.getLogger(__name__)
    logger.info(f"Starting with retry limits: schema={schema_retries_int}, qwen={qwen_retries_int}, crosscheck={crosscheck_retries_int}, final={final_retries_int}")

    # Create and store the task so we can cancel it later
    pipeline_state.pipeline_task = asyncio.create_task(
        run_pipeline_async(
            count,
            output_path,
            topic_list,
            schema_retries_int,
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


def create_app():
    """Factory function for the app."""
    return app
