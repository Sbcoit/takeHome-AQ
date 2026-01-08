"""CLI entry point for the Physics QA Generator."""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    name="physics-qa",
    help="Synthetic Graduate-Level Physics QA Generator",
    add_completion=False,
)
console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    # Suppress noisy loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_api_key() -> str:
    """Get OpenRouter API key from environment."""
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print(
            "[red]Error: OPENROUTER_API_KEY not set.[/red]\n"
            "Set it in your environment or create a .env file.\n"
            "See .env.example for reference."
        )
        raise typer.Exit(1)

    return api_key


def generate_unique_output_path(base_dir: str = "output") -> str:
    """Generate a unique timestamped output file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_dir}/dataset_{timestamp}.jsonl"


@app.command()
def generate(
    count: int = typer.Option(50, "--count", "-n", help="Number of QA pairs to generate"),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path. If not specified, creates a unique timestamped file (e.g., output/dataset_20260107_143052.jsonl)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    topic: Optional[list[str]] = typer.Option(
        None,
        "--topic",
        "-t",
        help="Specific topic(s) to generate questions for (can be used multiple times). "
             "Use 'physics-qa topics' to see available topics."
    ),
    schema_retries: Optional[int] = typer.Option(
        None,
        "--schema-retries",
        help="Max retries for schema validation. Default 3, 0 = unlimited."
    ),
    qwen_retries: Optional[int] = typer.Option(
        None,
        "--qwen-retries",
        help="Max retries for Qwen difficulty check. Default 5, 0 = unlimited."
    ),
    crosscheck_retries: Optional[int] = typer.Option(
        None,
        "--crosscheck-retries",
        help="Max retries for cross-check validation. Default 5, 0 = unlimited."
    ),
    final_retries: Optional[int] = typer.Option(
        None,
        "--final-retries",
        help="Max retries for final 5x blind validation. Default 10, 0 = unlimited."
    ),
):
    """Generate and validate physics QA pairs using parallel workers."""
    setup_logging(verbose)

    api_key = get_api_key()

    # Generate unique timestamped filename if not specified
    if output is None:
        output = generate_unique_output_path()
        console.print(f"[dim]Auto-generated output path: {output}[/dim]\n")

    # Format topics for display
    topics_display = ", ".join(topic) if topic else "All topics"

    # Helper to format retry display
    def format_retries(value: Optional[int], default: int) -> str:
        effective = value if value is not None else default
        return "Unlimited" if effective == 0 else str(effective)

    # Format retry limits for display
    schema_display = format_retries(schema_retries, 3)
    qwen_display = format_retries(qwen_retries, 5)
    crosscheck_display = format_retries(crosscheck_retries, 5)
    final_display = format_retries(final_retries, 10)

    console.print(
        Panel.fit(
            f"[bold blue]Physics QA Generator[/bold blue]\n\n"
            f"Target: [green]{count}[/green] QA pairs\n"
            f"Topics: [yellow]{topics_display}[/yellow]\n"
            f"Mode: [magenta]Parallel ({count} workers)[/magenta]\n"
            f"Retry Limits: [cyan]schema={schema_display}, qwen={qwen_display}, crosscheck={crosscheck_display}, final={final_display}[/cyan]\n"
            f"Output: [cyan]{output}[/cyan]",
            title="Configuration",
        )
    )

    # Progress tracking
    progress_data = {
        "current": 0,
        "generated": 0,
        "failed": 0,
        "phase": 1,
        "phase1_count": 0,
        "phase2_validated": 0,
        "phase2_passed": 0,
    }

    def progress_callback(data):
        progress_data["current"] = data.get("valid_count", 0)
        progress_data["generated"] = data.get("stats", {}).get("total_generated", 0)
        progress_data["failed"] = data.get("stats", {}).get("failed", 0)
        progress_data["phase"] = data.get("phase", 1)
        progress_data["phase1_count"] = data.get("phase1_count", 0)
        progress_data["phase2_validated"] = data.get("phase2_validated", 0)
        progress_data["phase2_passed"] = data.get("phase2_passed", 0)

    async def run():
        from .orchestration.pipeline import run_pipeline

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Phase 1: Generating QA pairs...", total=count)
            current_phase = 1

            # Run pipeline with progress updates
            async def run_with_updates():
                result = await run_pipeline(
                    api_key=api_key,
                    target_count=count,
                    output_path=output,
                    progress_callback=progress_callback,
                    topics=topic,
                    schema_max_retries=schema_retries,
                    qwen_max_retries=qwen_retries,
                    crosscheck_max_retries=crosscheck_retries,
                    final_validation_max_retries=final_retries,
                )

                return result

            # Create task for pipeline
            pipeline_task = asyncio.create_task(run_with_updates())

            # Update progress while running
            while not pipeline_task.done():
                phase = progress_data["phase"]

                if phase == "streaming":
                    # Streaming mode: both phases run concurrently
                    phase1_count = progress_data["phase1_count"]
                    phase2_passed = progress_data["phase2_passed"]
                    phase2_validated = progress_data["phase2_validated"]

                    # Show progress toward final target
                    # Gen: how many passed Phase 1 / target
                    # Audit: how many passed Phase 2 / target (the final goal)
                    if phase2_validated == 0:
                        audit_status = f"0/{count} (waiting...)"
                    else:
                        audit_status = f"{phase2_passed}/{count}"

                    progress.update(
                        task,
                        completed=phase2_passed,
                        description=f"[cyan]Gen: {phase1_count}/{count}[/cyan] | [yellow]Audit: {audit_status}[/yellow]"
                    )
                elif phase == 2 and current_phase == 1:
                    # Transition to Phase 2 (legacy sequential mode)
                    current_phase = 2
                    phase1_count = progress_data["phase1_count"]
                    progress.update(task, total=phase1_count, completed=0)
                    progress.update(
                        task,
                        description=f"[yellow]Phase 2: Auditing ({progress_data['phase2_passed']} passed)..."
                    )
                elif phase == 1:
                    progress.update(task, completed=progress_data["current"])
                elif phase == 2:
                    # Phase 2: update progress based on validated count
                    progress.update(
                        task,
                        completed=progress_data["phase2_validated"],
                        description=f"[yellow]Phase 2: Auditing ({progress_data['phase2_passed']} passed)..."
                    )

                await asyncio.sleep(0.5)

            result = await pipeline_task

            # Final update
            if progress_data["phase"] == "streaming" or progress_data["phase"] == 2:
                progress.update(
                    task,
                    completed=progress_data["phase2_passed"],
                    description=f"[green]Complete: {progress_data['phase2_passed']} QA pairs fully validated"
                )
            else:
                progress.update(task, completed=count)

            return result

    try:
        result = asyncio.run(run())

        # Display results
        console.print("\n")
        stats = result.get("stats", {})

        table = Table(title="Generation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Mode", "Parallel")
        table.add_row("Total Generated", str(stats.get("total_generated", 0)))
        table.add_row("Passed All Validation", str(stats.get("passed_all", 0)))
        table.add_row("Failed", str(stats.get("failed", 0)))
        table.add_row("Pass Rate", stats.get("pass_rate", "0%"))

        # Qwen constraint info
        qwen_info = result.get("qwen_constraint", {})
        if qwen_info:
            table.add_row("", "")
            table.add_row("[bold]Qwen Constraint (5% max easy)[/bold]", "")
            table.add_row("  Easy Questions", f"{qwen_info.get('easy_questions', 0)}/{qwen_info.get('total_questions', 0)}")
            table.add_row("  Easy Rate", qwen_info.get("easy_rate", "0%"))
            table.add_row("  Status", "[green]PASSED[/green]" if qwen_info.get("passed") else "[red]FAILED[/red]")

        # By stage (for sequential mode)
        by_stage = stats.get("by_stage", {})
        if by_stage:
            table.add_row("", "")
            table.add_row("[bold]By Validation Stage[/bold]", "")
            table.add_row("  Schema", str(by_stage.get("schema", 0)))
            table.add_row("  Qwen Constraint", str(by_stage.get("qwen_constraint", 0)))
            table.add_row("  Cross-check", str(by_stage.get("crosscheck", 0)))
            table.add_row("  Correctness", str(by_stage.get("correctness", 0)))

        console.print(table)

        console.print(f"\n[green]Output written to:[/green] {result.get('output_path')}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted by user.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the web UI server."""
    setup_logging(verbose=False)

    # Check API key first
    get_api_key()

    console.print(
        Panel.fit(
            f"[bold blue]Physics QA Generator - Web UI[/bold blue]\n\n"
            f"Server: [cyan]http://{host}:{port}[/cyan]\n"
            f"Dashboard: [cyan]http://{host}:{port}/[/cyan]",
            title="Starting Server",
        )
    )

    import uvicorn

    uvicorn.run(
        "src.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def validate(
    input_file: str = typer.Argument(..., help="JSONL file to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Validate an existing dataset file."""
    setup_logging(verbose)

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    import json
    from .validation.schema import SchemaValidator

    validator = SchemaValidator()
    valid_count = 0
    invalid_count = 0
    errors = []

    with open(input_path) as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                is_valid, errs = validator.validate(data)

                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    errors.append((i, errs))
            except json.JSONDecodeError as e:
                invalid_count += 1
                errors.append((i, [f"JSON parse error: {e}"]))

    # Display results
    table = Table(title=f"Validation Results: {input_file}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Lines", str(valid_count + invalid_count))
    table.add_row("Valid", str(valid_count))
    table.add_row("Invalid", str(invalid_count))
    table.add_row(
        "Validity Rate",
        f"{valid_count / (valid_count + invalid_count) * 100:.1f}%" if (valid_count + invalid_count) > 0 else "N/A"
    )

    console.print(table)

    if errors and verbose:
        console.print("\n[yellow]Errors:[/yellow]")
        for line_num, errs in errors[:10]:  # Show first 10 errors
            console.print(f"  Line {line_num}: {', '.join(errs)}")
        if len(errors) > 10:
            console.print(f"  ... and {len(errors) - 10} more errors")


@app.command()
def info():
    """Show system information and configuration."""
    from dotenv import load_dotenv

    load_dotenv()

    from . import __version__
    from .config import Settings

    console.print(
        Panel.fit(
            f"[bold blue]Physics QA Generator[/bold blue]\n"
            f"Version: {__version__}",
            title="System Info",
        )
    )

    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    api_status = "[green]Set[/green]" if api_key else "[red]Not set[/red]"

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("API Key", api_status)

    try:
        settings = Settings()
        table.add_row("Generation Model", settings.generation_model)
        table.add_row("Judge Model", settings.judge_model)
        table.add_row("Qwen Model", settings.qwen_model)
        table.add_row("Cross-check Models", ", ".join(settings.crosscheck_models))
        table.add_row("Target Count", str(settings.target_count))
        table.add_row("Output Dir", settings.output_dir)
    except Exception as e:
        table.add_row("Settings Error", f"[red]{e}[/red]")

    console.print(table)


@app.command()
def topics():
    """List available physics topics."""
    from .generation.topics import TopicSampler
    from .models.schemas import PhysicsTopic

    table = Table(title="Available Physics Topics")
    table.add_column("Topic", style="cyan")
    table.add_column("Subtopics", style="green")

    sampler = TopicSampler()

    for topic in PhysicsTopic:
        subtopics = sampler.SUBTOPICS.get(topic, [])
        table.add_row(
            topic.value,
            f"{len(subtopics)} subtopics"
        )

    console.print(table)

    console.print(f"\n[dim]Total: {len(PhysicsTopic)} topics, "
                  f"{sum(len(s) for s in sampler.SUBTOPICS.values())} subtopics[/dim]")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
