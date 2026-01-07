"""FastAPI web application for the Physics QA Generator."""

import asyncio
import json
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

    def reset(self):
        self.is_running = False
        self.run_id = None
        self.target_count = 0
        self.current_count = 0
        self.stats = {}
        self.recent_qa = []
        self.logs = []
        self.error = None

    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        # Keep only last 100 logs
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

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
        }


pipeline_state = PipelineState()


class GenerateRequest(BaseModel):
    count: int = 50
    output_path: str = "output/dataset.jsonl"


def progress_callback(data: Dict[str, Any]):
    """Callback for pipeline progress updates."""
    pipeline_state.current_count = data.get("valid_count", 0)
    pipeline_state.stats = data.get("stats", {})

    stage = data.get("stage", "")
    details = data.get("details", {})

    if stage == "generated":
        pipeline_state.add_log(f"Generated QA for: {details.get('topic', 'unknown')}")
    elif stage == "all_passed":
        pipeline_state.add_log(f"QA {details.get('id', '')[:8]}... passed all validation!")

        # Keep track of recent QA (for display)
        # Note: In a real app, we'd load from the output file
        if len(pipeline_state.recent_qa) >= 10:
            pipeline_state.recent_qa.pop(0)


async def run_pipeline_async(count: int, output_path: str):
    """Run the pipeline in the background."""
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pipeline_state.error = "OPENROUTER_API_KEY not set"
        pipeline_state.is_running = False
        return

    # Get OpenAI API key for models not supported by OpenRouter
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    try:
        from ..orchestration.pipeline import run_pipeline

        pipeline_state.add_log(f"Starting pipeline: target={count}, output={output_path}")

        result = await run_pipeline(
            api_key=api_key,
            target_count=count,
            output_path=output_path,
            progress_callback=progress_callback,
            openai_api_key=openai_api_key,
        )

        pipeline_state.stats = result.get("stats", {})
        pipeline_state.add_log(f"Pipeline completed! {pipeline_state.current_count} QA pairs generated.")

    except Exception as e:
        pipeline_state.error = str(e)
        pipeline_state.add_log(f"Pipeline error: {e}")

    finally:
        pipeline_state.is_running = False


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "state": pipeline_state.to_dict(),
        }
    )


@app.get("/api/status")
async def get_status():
    """Get current pipeline status."""
    return JSONResponse(pipeline_state.to_dict())


@app.get("/api/logs")
async def get_logs(limit: int = 20):
    """Get recent logs."""
    return JSONResponse({
        "logs": pipeline_state.logs[-limit:],
    })


@app.post("/api/generate")
async def start_generation(
    background_tasks: BackgroundTasks,
    count: int = Form(default=50),
    output_path: str = Form(default="output/dataset.jsonl"),
):
    """Start the generation pipeline."""
    if pipeline_state.is_running:
        raise HTTPException(status_code=400, detail="Pipeline is already running")

    pipeline_state.reset()
    pipeline_state.is_running = True
    pipeline_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_state.target_count = count

    pipeline_state.add_log(f"Generation requested: {count} QA pairs")

    # Run in background
    background_tasks.add_task(run_pipeline_async, count, output_path)

    return JSONResponse({
        "status": "started",
        "run_id": pipeline_state.run_id,
    })


@app.post("/api/stop")
async def stop_generation():
    """Request the pipeline to stop."""
    if not pipeline_state.is_running:
        raise HTTPException(status_code=400, detail="Pipeline is not running")

    pipeline_state.add_log("Stop requested...")
    # Note: In a real implementation, we'd signal the pipeline to stop
    # For now, this is a placeholder

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
            "logs": pipeline_state.logs[-20:],
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
