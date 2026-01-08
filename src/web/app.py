"""FastAPI web application for the Physics QA Generator."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

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
        self.mix_topics: bool = False
        self.output_path: str = "output/dataset.jsonl"

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
        self.mix_topics = False
        self.output_path = "output/dataset.jsonl"

    def add_log(self, message: str):
        self.logs.append(message)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

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
            "mix_topics": self.mix_topics,
            "output_path": self.output_path,
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
    mix_topics: bool = False


def progress_callback(data: Dict[str, Any]):
    """Callback for pipeline progress updates."""
    pipeline_state.current_count = data.get("valid_count", 0)
    pipeline_state.stats = data.get("stats", {})


async def run_pipeline_async(count: int, output_path: str, topics: Optional[List[str]] = None, mix_topics: bool = False):
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

    try:
        from ..orchestration.pipeline import run_pipeline

        result = await run_pipeline(
            api_key=api_key,
            target_count=count,
            output_path=output_path,
            progress_callback=progress_callback,
            openai_api_key=openai_api_key,
            topics=topics,
            mix_topics=mix_topics,
        )

        pipeline_state.stats = result.get("stats", {})
        logger.info(f"Pipeline complete: {pipeline_state.current_count} valid pairs")

    except Exception as e:
        pipeline_state.error = str(e)
        logger.error(f"Pipeline failed: {e}")

    finally:
        pipeline_state.is_running = False


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


@app.post("/api/generate")
async def start_generation(
    background_tasks: BackgroundTasks,
    count: int = Form(default=50),
    output_path: str = Form(default="output/dataset.jsonl"),
    topics: Optional[str] = Form(default=None),
    mix_topics: bool = Form(default=False),
):
    """Start the generation pipeline."""
    if pipeline_state.is_running:
        raise HTTPException(status_code=400, detail="Pipeline is already running")

    pipeline_state.reset()
    pipeline_state.is_running = True
    pipeline_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_state.target_count = count
    pipeline_state.output_path = output_path

    topic_list = None
    if topics and topics.strip():
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]

    pipeline_state.topics = topic_list
    pipeline_state.mix_topics = mix_topics

    background_tasks.add_task(run_pipeline_async, count, output_path, topic_list, mix_topics)

    return JSONResponse({
        "status": "started",
        "run_id": pipeline_state.run_id,
    })


@app.post("/api/stop")
async def stop_generation():
    """Request the pipeline to stop."""
    if not pipeline_state.is_running:
        raise HTTPException(status_code=400, detail="Pipeline is not running")

    logging.getLogger(__name__).info("Stop requested")
    return JSONResponse({"status": "stop_requested"})


@app.get("/api/output")
async def get_output(limit: int = 10, offset: int = 0):
    """Get generated QA pairs from the output file."""
    output_path = Path("output/dataset.jsonl")

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
    output_path = Path("output/dataset.jsonl")

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
    output_path = Path("output/dataset.jsonl")
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


def create_app():
    """Factory function for the app."""
    return app
