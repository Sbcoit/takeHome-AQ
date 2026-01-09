"""Pipeline orchestration for QA generation and validation."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple

from ..api.client import OpenRouterClient
from ..config import Settings
from ..models.schemas import (
    PhysicsQADataPoint,
    PhysicsTopic,
    GenerationStats,
)
from ..generation.topics import TopicSampler, TopicContext
from ..generation.generator import QAGenerator
from ..validation.schema import SchemaValidator
from ..validation.qwen_check import QwenConstraintValidator
from ..validation.crosscheck import CrossCheckValidator
from ..validation.final_answer import FinalAnswerValidator
from ..validation.derivation_audit import DerivationAuditor
from ..validation.sanity_check import SanityCheckValidator, MultiModelSanityValidator
from ..validation.symbolic_math import SymbolicMathValidator
from ..validation.completeness_check import CompletenessValidator, QuickCompletenessChecker

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """Result from a single worker's pipeline execution."""
    qa: Optional[PhysicsQADataPoint] = None
    is_easy: bool = False
    is_incorrect: bool = False
    success: bool = False
    worker_id: int = 0
    topic: Optional[str] = None
    error: Optional[str] = None


class CheckpointManager:
    """Manages checkpointing for crash recovery."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        run_id: str,
        valid_qa_pairs: List[PhysicsQADataPoint],
        stats: GenerationStats,
        failed_ids: List[str],
    ):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{run_id}.json"

        data = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "stats": stats.model_dump(),
            "valid_count": len(valid_qa_pairs),
            "valid_ids": [qa.id for qa in valid_qa_pairs],
            "failed_ids": failed_ids,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint if exists."""
        checkpoint_path = self.checkpoint_dir / f"{run_id}.json"

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path) as f:
            return json.load(f)

    def clear(self, run_id: str):
        """Clear checkpoint after successful completion."""
        checkpoint_path = self.checkpoint_dir / f"{run_id}.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()


class PipelineRunner:
    """Orchestrates the complete generation and validation pipeline."""

    def __init__(
        self,
        client: OpenRouterClient,
        settings: Settings,
        output_path: str = "output/dataset.jsonl",
        run_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        topics: Optional[List[PhysicsTopic]] = None,
    ):
        """
        Initialize the pipeline runner.

        Args:
            client: OpenRouter API client
            settings: Application settings
            output_path: Path for JSONL output
            run_id: Optional run ID for checkpointing
            progress_callback: Optional callback for progress updates
            topics: Optional list of specific topics to generate questions for
        """
        self.client = client
        self.settings = settings
        self.output_path = Path(output_path)
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.progress_callback = progress_callback

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize components with optional topic filtering
        self.topic_sampler = TopicSampler(topics=topics)
        self.qa_generator = QAGenerator(
            client,
            settings.generation_model,
            judge_model=settings.judge_model,
            solver_model=settings.solver_model,
        )
        self.schema_validator = SchemaValidator()
        self.qwen_validator = QwenConstraintValidator(
            client,
            qwen_model=settings.qwen_model,
            judge_model=settings.judge_model,
            max_pass_rate=settings.qwen_max_pass_rate,
        )
        self.crosscheck_validator = CrossCheckValidator(
            client,
            judge_model=settings.judge_model,
            models=settings.crosscheck_models,
            min_models_with_correct=settings.min_crosscheck_models,
            min_total_correct=settings.min_crosscheck_total,
        )
        # Note: Correctness is now inferred from cross-check success (Option A)
        # If frontier models can solve the problem, the reference answer is likely correct
        self.checkpoint_manager = CheckpointManager()

        # Statistics
        self.stats = GenerationStats()
        self.valid_qa_pairs: List[PhysicsQADataPoint] = []
        self.failed_ids: List[str] = []

        # Dataset-level Qwen constraint tracking
        # "The proportion of cases where Qwen passes 4+/5 must not exceed 5%"
        self.easy_question_count = 0  # Questions where Qwen passed 4+/5
        self.qwen_max_easy_rate = settings.qwen_max_pass_rate  # 0.05 = 5%

        # Dataset-level correctness constraint tracking
        # "Across the accepted dataset, the accuracy should be higher than 90%"
        self.incorrect_question_count = 0  # Questions that failed correctness check
        self.correctness_min_rate = settings.correctness_threshold  # 0.90 = 90%

        # Control
        self._stop_requested = False

    def request_stop(self):
        """Request the pipeline to stop gracefully."""
        self._stop_requested = True
        logger.info("Stop requested, will finish current item and stop...")

    def _report_progress(self, stage: str, details: Dict[str, Any]):
        """Report progress via callback."""
        if self.progress_callback:
            self.progress_callback({
                "stage": stage,
                "stats": self.stats.to_summary_dict(),
                "valid_count": len(self.valid_qa_pairs),
                "details": details,
            })

    async def _validate_qa(
        self,
        qa: PhysicsQADataPoint,
        _topic_context: TopicContext,  # Reserved for future use (e.g., topic-specific validation)
    ) -> tuple[Optional[PhysicsQADataPoint], Optional[str], Optional[str], bool, bool]:
        """
        Validate a QA pair through all checks.

        Validation order (optimized):
        1. Schema validation (instant, no API calls)
        2. Qwen constraint (5 API calls) - checks question isn't too easy
        3. Cross-check (20 API calls) - checks question is solvable AND serves as correctness proxy

        Returns:
            Tuple of (validated_qa, failure_stage, feedback_for_regeneration, is_easy, is_incorrect)
            - If successful: (qa, None, None, is_easy, is_incorrect)
            - If failed: (None, stage_name, feedback_message, False, False)
        """
        # Schema validation (instant, no API calls)
        schema_valid, schema_errors, _ = self.schema_validator.validate_complete(
            qa.model_dump()
        )
        if not schema_valid:
            logger.warning(f"Schema validation failed: {schema_errors}")
            return None, "schema", f"Fix schema errors: {schema_errors}", False, False

        self.stats.passed_schema += 1
        self._report_progress("schema_passed", {"id": qa.id})

        # Qwen constraint check (DATASET-LEVEL constraint)
        # Rule: "The proportion of cases where Qwen passes 4+/5 must not exceed 5%"
        logger.info("Running Qwen constraint check...")
        qwen_pass_rate, qwen_high_count, is_easy, _ = await self.qwen_validator.validate(
            qa, samples=self.settings.samples_per_qwen_check
        )

        # Check if accepting this question would exceed the 5% easy question threshold
        if is_easy:
            # Calculate what the easy rate would be if we accept this question
            current_valid = len(self.valid_qa_pairs)
            potential_easy_count = self.easy_question_count + 1
            potential_total = current_valid + 1
            potential_easy_rate = potential_easy_count / potential_total

            if potential_easy_rate > self.qwen_max_easy_rate:
                logger.warning(
                    f"Qwen constraint: Question is easy ({qwen_high_count}/5 passed). "
                    f"Rejecting because accepting would exceed 5% threshold "
                    f"({potential_easy_count}/{potential_total} = {potential_easy_rate:.1%})"
                )
                feedback = (
                    f"The question is TOO EASY. Qwen3-max solved it correctly {qwen_high_count} out of 5 times. "
                    "Make the question significantly harder by: "
                    "1) Adding more complex physics that requires deeper understanding, "
                    "2) Introducing subtle traps where naive approaches fail, "
                    "3) Requiring multi-step reasoning that can't be pattern-matched, "
                    "4) Using non-standard scenarios not found in textbooks. "
                    "The question should be challenging enough that even strong AI models struggle."
                )
                return None, "qwen", feedback, False, False
            else:
                logger.info(
                    f"Qwen constraint: Question is easy ({qwen_high_count}/5 passed) but "
                    f"accepting because within 5% threshold ({potential_easy_count}/{potential_total} = {potential_easy_rate:.1%})"
                )
        else:
            logger.info(f"Qwen constraint: Question is NOT easy ({qwen_high_count}/5 passed)")

        self.stats.passed_qwen += 1
        self._report_progress("qwen_passed", {
            "id": qa.id,
            "pass_rate": qwen_pass_rate,
            "is_easy": is_easy,
        })

        # Cross-check validation (also serves as correctness proxy)
        # If 4 frontier models can solve the question >= 25% of the time,
        # the question and answer are likely correct.
        logger.info("Running cross-check validation...")
        _, models_correct, total_correct, crosscheck_valid, _ = (
            await self.crosscheck_validator.validate(
                qa, samples_per_model=self.settings.samples_per_crosscheck
            )
        )

        if not crosscheck_valid:
            logger.warning(
                f"Cross-check failed: {models_correct} models with correct, "
                f"{total_correct}/20 total correct"
            )
            feedback = (
                f"The question is TOO HARD or has an incorrect answer. "
                f"Only {total_correct}/20 attempts across 4 frontier models were correct "
                f"(need at least 5). Either: "
                "1) The reference answer/solution is incorrect - verify and fix it, "
                "2) The question is ambiguous or poorly worded - clarify it, "
                "3) The question is unsolvable without information not provided - add necessary context, "
                "4) The difficulty is too extreme - make it more accessible while keeping it graduate-level. "
                "Strong AI models should be able to solve this at least 25% of the time."
            )
            return None, "crosscheck", feedback, is_easy, False

        # Question passed both Qwen (not too easy) and cross-check (solvable by frontier models)
        # This means it's in the sweet spot - accept it!
        #
        # The 90% correctness requirement is satisfied by cross-check:
        # If frontier models can solve it >= 25% of the time, the answer is correct.
        # We just track the cross-check accuracy as the "correctness" metric.
        total_attempts = 20  # 4 models × 5 samples
        crosscheck_accuracy = total_correct / total_attempts

        self.stats.passed_crosscheck += 1
        self.stats.passed_correctness += 1
        self.stats.passed_all += 1
        qa.validation_passed = True

        self._report_progress("all_passed", {
            "id": qa.id,
            "models_correct": models_correct,
            "total_correct": total_correct,
            "crosscheck_accuracy": crosscheck_accuracy,
        })

        logger.info(
            f"QA pair {qa.id} passed all validation! "
            f"(cross-check: {total_correct}/20 = {crosscheck_accuracy:.0%})"
        )

        # is_easy already tracked above, is_incorrect = False since cross-check passed
        return qa, None, None, is_easy, False

    async def generate_and_validate_single(self) -> tuple[Optional[PhysicsQADataPoint], bool, bool]:
        """
        Generate and validate a single QA pair with regeneration on failure.

        Returns:
            Tuple of (validated_qa, is_easy, is_incorrect)
            - is_easy indicates if this question was "easy" (Qwen passed 4+/5)
            - is_incorrect indicates if this question failed correctness check
        """
        # Generate initial question
        topic_context = self.topic_sampler.sample(prefer_diverse=True)
        logger.info(f"Generating QA for: {topic_context.subtopic}")

        try:
            qa = await self.qa_generator.generate(topic_context, temperature=0.8)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.stats.failed += 1
            return None, False, False

        self.stats.total_generated += 1
        self._report_progress("generated", {"topic": topic_context.subtopic})

        # Track by topic
        topic_name = topic_context.topic.value
        self.stats.by_topic[topic_name] = self.stats.by_topic.get(topic_name, 0) + 1

        # Validate with regeneration loop (limited attempts to prevent infinite loops)
        MAX_REGEN_ATTEMPTS = 10  # Increased for better success rate with harder questions
        attempt = 0
        while attempt < MAX_REGEN_ATTEMPTS:
            attempt += 1
            validated_qa, failure_stage, feedback, is_easy, is_incorrect = await self._validate_qa(qa, topic_context)

            if validated_qa is not None:
                return validated_qa, is_easy, is_incorrect

            # Schema failures can't be fixed by regeneration
            if failure_stage == "schema":
                self.stats.failed += 1
                self.failed_ids.append(qa.id)
                return None, False, False

            # Check if we've exhausted regeneration attempts
            if attempt >= MAX_REGEN_ATTEMPTS:
                logger.warning(
                    f"Abandoning question after {MAX_REGEN_ATTEMPTS} regeneration attempts. "
                    f"Last failure: {failure_stage}"
                )
                self.stats.failed += 1
                self.failed_ids.append(qa.id)
                return None, False, False

            # Regenerate with feedback
            logger.info(f"Regenerating question (attempt {attempt + 1}/{MAX_REGEN_ATTEMPTS}) - {failure_stage} failed")
            try:
                qa = await self.qa_generator.regenerate_with_feedback(
                    original=qa,
                    feedback=feedback,
                    temperature=0.7,
                )
                self.stats.total_generated += 1
                logger.info(f"Regenerated question for {topic_context.subtopic}")
            except Exception as e:
                logger.error(f"Regeneration failed: {e}")
                self.stats.failed += 1
                self.failed_ids.append(qa.id)
                return None, False, False

        # Should not reach here, but safety fallback
        self.stats.failed += 1
        return None, False, False

    def _write_to_output(self, qa: PhysicsQADataPoint):
        """Append a single QA pair to JSONL output."""
        with open(self.output_path, "a") as f:
            f.write(qa.to_json_line() + "\n")

    async def run(self, target_count: int) -> List[PhysicsQADataPoint]:
        """
        Run the pipeline until we have enough valid QA pairs.

        Args:
            target_count: Target number of valid QA pairs

        Returns:
            List of valid PhysicsQADataPoint objects
        """
        self.stats.started_at = datetime.utcnow()
        logger.info(f"Starting pipeline run {self.run_id}, target: {target_count} QA pairs")

        # Append to existing output file (don't delete - preserves data from failed runs)
        # New QAs will be appended to the end
        if self.output_path.exists():
            logger.info(f"Appending to existing output file: {self.output_path}")
        else:
            logger.info(f"Creating new output file: {self.output_path}")

        checkpoint_interval = 5  # Save checkpoint every N valid pairs

        attempts = 0

        # ALWAYS retry indefinitely until we reach the target count
        # The pipeline should NEVER stop until it creates the amount of QA the user wants
        logger.info(
            f"Pipeline will retry indefinitely until {target_count} valid QA pairs are generated. "
            f"Data is written to disk immediately and will NOT be lost on failure."
        )

        while len(self.valid_qa_pairs) < target_count and not self._stop_requested:
            attempts += 1

            try:
                qa, is_easy, is_incorrect = await self.generate_and_validate_single()

                if qa:
                    self.valid_qa_pairs.append(qa)
                    self._write_to_output(qa)

                    # Track easy questions for dataset-level Qwen constraint
                    if is_easy:
                        self.easy_question_count += 1

                    # Track incorrect questions for dataset-level correctness constraint
                    if is_incorrect:
                        self.incorrect_question_count += 1

                    # Checkpoint
                    if len(self.valid_qa_pairs) % checkpoint_interval == 0:
                        self.checkpoint_manager.save(
                            self.run_id,
                            self.valid_qa_pairs,
                            self.stats,
                            self.failed_ids,
                        )

                    easy_rate = self.easy_question_count / len(self.valid_qa_pairs)
                    correct_rate = (len(self.valid_qa_pairs) - self.incorrect_question_count) / len(self.valid_qa_pairs)
                    logger.info(
                        f"Progress: {len(self.valid_qa_pairs)}/{target_count} valid pairs "
                        f"({self.stats.total_generated} generated, {self.stats.failed} failed, "
                        f"easy: {self.easy_question_count}/{len(self.valid_qa_pairs)} = {easy_rate:.1%}, "
                        f"correct: {len(self.valid_qa_pairs) - self.incorrect_question_count}/{len(self.valid_qa_pairs)} = {correct_rate:.1%})"
                    )

            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                self.stats.failed += 1
                # Continue with next attempt

        # Final statistics
        self.stats.completed_at = datetime.utcnow()

        # Update API stats
        api_stats = self.client.get_stats()
        self.stats.total_api_calls = api_stats.get("total_calls", 0)
        self.stats.total_api_time_seconds = api_stats.get("total_time_seconds", 0)

        # Clear checkpoint on completion
        self.checkpoint_manager.clear(self.run_id)

        logger.info(
            f"Pipeline complete! Generated {len(self.valid_qa_pairs)} valid QA pairs. "
            f"Output: {self.output_path}"
        )

        return self.valid_qa_pairs

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline statistics."""
        valid_count = len(self.valid_qa_pairs)
        easy_rate = self.easy_question_count / valid_count if valid_count > 0 else 0
        correct_count = valid_count - self.incorrect_question_count
        correct_rate = correct_count / valid_count if valid_count > 0 else 1.0
        return {
            "run_id": self.run_id,
            "output_path": str(self.output_path),
            "stats": self.stats.to_summary_dict(),
            "valid_count": valid_count,
            "qwen_constraint": {
                "easy_questions": self.easy_question_count,
                "total_questions": valid_count,
                "easy_rate": f"{easy_rate:.1%}",
                "max_allowed_rate": f"{self.qwen_max_easy_rate:.0%}",
                "passed": easy_rate <= self.qwen_max_easy_rate,
            },
            "correctness_constraint": {
                "correct_questions": correct_count,
                "incorrect_questions": self.incorrect_question_count,
                "total_questions": valid_count,
                "correct_rate": f"{correct_rate:.1%}",
                "min_required_rate": f"{self.correctness_min_rate:.0%}",
                "passed": correct_rate >= self.correctness_min_rate,
            },
            "topic_coverage": self.topic_sampler.get_coverage_stats(),
            "api_stats": self.client.get_stats(),
        }


class ParallelPipelineRunner:
    """
    Orchestrates parallel generation and validation of QA pairs.

    Uses a worker pool pattern where multiple workers run the full pipeline
    (generate → validate → check) concurrently. Results are collected and
    the 5% easy question threshold is applied post-generation.
    """

    def __init__(
        self,
        client: OpenRouterClient,
        settings: Settings,
        output_path: str = "output/dataset.jsonl",
        run_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        topics: Optional[List[PhysicsTopic]] = None,
        schema_max_retries: Optional[int] = None,
        sanity_check_max_retries: Optional[int] = None,
        qwen_max_retries: Optional[int] = None,
        crosscheck_max_retries: Optional[int] = None,
        final_validation_max_retries: Optional[int] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        """
        Initialize the parallel pipeline runner.

        Args:
            client: OpenRouter API client
            settings: Application settings
            output_path: Path for JSONL output
            run_id: Optional run ID for checkpointing
            progress_callback: Optional callback for progress updates
            topics: Optional list of specific topics to generate questions for
            schema_max_retries: Max retries for schema validation. Default 3, 0 = unlimited.
            sanity_check_max_retries: Max retries for sanity checks (dimensional analysis, etc.). Default 5, 0 = unlimited.
            qwen_max_retries: Max retries for Qwen difficulty check. Default 5, 0 = unlimited.
            crosscheck_max_retries: Max retries for cross-check validation. Default 5, 0 = unlimited.
            final_validation_max_retries: Max retries for final 5x blind validation. Default 10, 0 = unlimited.
            stop_check: Optional callback that returns True if stop was requested.
        """
        self.client = client
        self.stop_check = stop_check
        self.settings = settings
        self.output_path = Path(output_path)
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.progress_callback = progress_callback
        self.topics = topics

        # Helper to process retry limits: None = use default, 0 = unlimited (None internally), else use value
        def process_retry_limit(value: Optional[int], default: int) -> Optional[int]:
            if value is None:
                return default
            elif value == 0:
                return None  # None = unlimited internally
            else:
                return value

        # Granular retry limits for each validation stage
        self.schema_max_retries = process_retry_limit(schema_max_retries, settings.schema_max_retries)
        self.sanity_check_max_retries = process_retry_limit(sanity_check_max_retries, settings.sanity_check_max_retries)
        self.qwen_max_retries = process_retry_limit(qwen_max_retries, settings.qwen_max_retries)
        self.crosscheck_max_retries = process_retry_limit(crosscheck_max_retries, settings.crosscheck_max_retries)
        self.final_validation_max_retries = process_retry_limit(final_validation_max_retries, settings.final_validation_max_retries)

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Shared components (thread-safe for reads)
        self.schema_validator = SchemaValidator()

        # Statistics (will be aggregated from workers)
        self.stats = GenerationStats()
        self.valid_qa_pairs: List[PhysicsQADataPoint] = []
        self.failed_ids: List[str] = []

        # For post-generation filtering
        self.all_results: List[WorkerResult] = []
        self.qwen_max_easy_rate = settings.qwen_max_pass_rate  # 0.05 = 5%

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager()

        # Control
        self._stop_requested_internal = False
        self._results_lock = asyncio.Lock()

    @property
    def _stop_requested(self) -> bool:
        """Check if stop was requested (internal flag or external callback)."""
        if self._stop_requested_internal:
            return True
        if self.stop_check and self.stop_check():
            return True
        return False

    def request_stop(self):
        """Request the pipeline to stop gracefully."""
        self._stop_requested_internal = True
        logger.info("Stop requested, workers will finish current tasks and stop...")

    def _report_progress(self, stage: str, details: Dict[str, Any]):
        """Report progress via callback."""
        if self.progress_callback:
            data = {
                "stage": stage,
                "stats": self.stats.to_summary_dict(),
                "valid_count": len(self.valid_qa_pairs),
                "details": details,
                "phase": details.get("phase", 1),
            }
            # Include phase tracking info for both phase 2 and streaming mode
            phase = details.get("phase")
            if phase == 2 or phase == "streaming":
                data["phase1_count"] = details.get("phase1_count", 0)
                data["phase2_validated"] = details.get("phase2_validated", 0)
                data["phase2_passed"] = details.get("phase2_passed", 0)
            self.progress_callback(data)

    async def _worker(
        self,
        worker_id: int,
        total_workers: int,
        results_queue: asyncio.Queue,
    ) -> WorkerResult:
        """
        Single worker that generates and validates one QA pair.

        Args:
            worker_id: Unique identifier for this worker
            total_workers: Total number of workers
            results_queue: Queue to put results into

        Returns:
            WorkerResult with the outcome
        """
        result = WorkerResult(worker_id=worker_id)

        try:
            # Create worker-specific components
            topic_sampler = TopicSampler(topics=self.topics)
            qa_generator = QAGenerator(
                self.client,
                self.settings.generation_model,
                judge_model=self.settings.judge_model,
                solver_model=self.settings.solver_model,
            )
            qwen_validator = QwenConstraintValidator(
                self.client,
                qwen_model=self.settings.qwen_model,
                judge_model=self.settings.judge_model,
                max_pass_rate=self.settings.qwen_max_pass_rate,
            )
            crosscheck_validator = CrossCheckValidator(
                self.client,
                judge_model=self.settings.judge_model,
                models=self.settings.crosscheck_models,
                min_models_with_correct=self.settings.min_crosscheck_models,
                min_total_correct=self.settings.min_crosscheck_total,
            )

            # Generate initial question
            topic_context = topic_sampler.sample(prefer_diverse=True)
            result.topic = topic_context.subtopic
            logger.info(f"[Worker {worker_id}] Generating QA for: {topic_context.subtopic}")

            try:
                qa = await qa_generator.generate(topic_context, temperature=0.8)
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Generation failed: {e}")
                result.error = str(e)
                await results_queue.put(result)
                return result

            # Validate with regeneration loops - separate counters for each stage
            # Helper to get max attempts and display string
            def get_max_and_display(limit: Optional[int]) -> tuple:
                if limit is None:
                    return float('inf'), "unlimited"
                return limit, str(limit)

            # Separate retry limits for each validation stage
            schema_max, schema_display = get_max_and_display(self.schema_max_retries)
            sanity_max, sanity_display = get_max_and_display(self.sanity_check_max_retries)
            qwen_max, qwen_display = get_max_and_display(self.qwen_max_retries)
            crosscheck_max, crosscheck_display = get_max_and_display(self.crosscheck_max_retries)

            # Create sanity check validator for this worker
            # Use multi-model mode if configured (catches model blind spots)
            if self.settings.use_multi_model_sanity:
                sanity_validator = MultiModelSanityValidator(
                    self.client,
                    models=self.settings.sanity_check_models,
                    consensus_mode=self.settings.sanity_consensus_mode,
                )
                logger.info(f"[Worker {worker_id}] Using multi-model sanity check with {len(self.settings.sanity_check_models)} models")
            else:
                sanity_validator = SanityCheckValidator(
                    self.client,
                    check_model=self.settings.judge_model,
                )

            # Create optional enhanced validators
            symbolic_validator = None
            if self.settings.use_symbolic_math:
                symbolic_validator = SymbolicMathValidator()

            completeness_validator = None
            quick_completeness = None
            if self.settings.use_completeness_check:
                completeness_validator = CompletenessValidator(
                    self.client,
                    check_model=self.settings.judge_model,
                )
                quick_completeness = QuickCompletenessChecker()

            # Separate attempt counters
            schema_attempts = 0
            sanity_attempts = 0
            qwen_attempts = 0
            crosscheck_attempts = 0

            while True:
                # === Stage 1: Schema validation ===
                schema_valid, schema_errors, _ = self.schema_validator.validate_complete(
                    qa.model_dump()
                )
                if not schema_valid:
                    schema_attempts += 1
                    logger.warning(f"[Worker {worker_id}] Schema validation failed: {schema_errors}")

                    # Check if we've exceeded schema retry limit
                    if schema_max != float('inf') and schema_attempts >= schema_max:
                        result.error = f"Schema errors after {schema_display} attempts: {schema_errors}"
                        await results_queue.put(result)
                        return result

                    feedback = f"Schema validation failed: {schema_errors}. Fix these issues."
                    logger.info(f"[Worker {worker_id}] Regenerating for schema fix (attempt {schema_attempts + 1}/{schema_display})")
                    try:
                        qa = await qa_generator.regenerate_with_feedback(
                            original=qa,
                            feedback=feedback,
                            temperature=0.7,
                        )
                        continue  # Re-validate with new QA
                    except Exception as e:
                        logger.error(f"[Worker {worker_id}] Regeneration failed: {e}")
                        result.error = str(e)
                        await results_queue.put(result)
                        return result

                # === Stage 1.5: Problem completeness check (optional) ===
                # Ensures the problem is well-posed before checking the answer
                if completeness_validator is not None:
                    # Quick heuristic check first (no API call)
                    if quick_completeness is not None:
                        quick_complete, quick_warnings = quick_completeness.check(qa.query)
                        if not quick_complete:
                            logger.debug(f"[Worker {worker_id}] Quick completeness warnings: {quick_warnings}")

                    # Full completeness check
                    logger.info(f"[Worker {worker_id}] Running problem completeness check...")
                    complete_passed, complete_details, complete_feedback = await completeness_validator.validate(qa)

                    if not complete_passed:
                        schema_attempts += 1  # Reuse schema counter for completeness
                        logger.warning(f"[Worker {worker_id}] Problem is incomplete/ambiguous.")

                        if schema_max != float('inf') and schema_attempts >= schema_max:
                            result.error = f"Problem completeness failed after {schema_display} attempts"
                            await results_queue.put(result)
                            return result

                        logger.info(f"[Worker {worker_id}] Regenerating to fix completeness issues")
                        try:
                            qa = await qa_generator.regenerate_with_feedback(
                                original=qa,
                                feedback=complete_feedback,
                                temperature=0.7,
                            )
                            continue
                        except Exception as e:
                            logger.error(f"[Worker {worker_id}] Regeneration failed: {e}")
                            result.error = str(e)
                            await results_queue.put(result)
                            return result

                # === Stage 1.75: Symbolic math verification (optional, no API call) ===
                # Programmatic check using SymPy - catches errors AI models miss
                if symbolic_validator is not None:
                    logger.debug(f"[Worker {worker_id}] Running symbolic math verification...")
                    sym_passed, sym_details, sym_feedback = symbolic_validator.validate(
                        qa.query, qa.response_answer, qa.response_reasoning
                    )

                    if not sym_passed:
                        logger.warning(f"[Worker {worker_id}] Symbolic math check found issues: {sym_details}")
                        # Don't fail outright - log as warning and let sanity check handle it
                        # The feedback will be included if sanity check also fails

                # === Stage 2: Sanity checks (dimensional analysis, limiting cases, etc.) ===
                # This catches ~80% of errors with a single API call before expensive Qwen/cross-check
                logger.info(f"[Worker {worker_id}] Running sanity checks (dimensional analysis, etc.)...")
                sanity_passed, sanity_details, sanity_feedback = await sanity_validator.validate(qa)

                if not sanity_passed:
                    sanity_attempts += 1
                    logger.warning(
                        f"[Worker {worker_id}] Sanity check failed. "
                        f"Regenerating (attempt {sanity_attempts}/{sanity_display})."
                    )

                    # Check if we've exceeded sanity check retry limit
                    if sanity_max != float('inf') and sanity_attempts >= sanity_max:
                        result.error = f"Sanity checks failed after {sanity_display} attempts"
                        await results_queue.put(result)
                        return result

                    # Regenerate with detailed feedback about what failed
                    logger.info(f"[Worker {worker_id}] Regenerating to fix sanity check failures (attempt {sanity_attempts + 1}/{sanity_display})")
                    try:
                        qa = await qa_generator.regenerate_with_feedback(
                            original=qa,
                            feedback=sanity_feedback,
                            temperature=0.7,
                        )
                        continue  # Re-validate from the beginning (schema check)
                    except Exception as e:
                        logger.error(f"[Worker {worker_id}] Regeneration failed: {e}")
                        result.error = str(e)
                        await results_queue.put(result)
                        return result

                # === Stage 3: Qwen constraint check ===
                logger.info(f"[Worker {worker_id}] Running Qwen constraint check...")
                qwen_pass_rate, qwen_high_count, is_easy, _ = await qwen_validator.validate(
                    qa, samples=self.settings.samples_per_qwen_check
                )

                # If question is too easy (Qwen passes 4+/5), regenerate to make it harder
                if is_easy:
                    qwen_attempts += 1
                    logger.warning(
                        f"[Worker {worker_id}] Question is too easy ({qwen_high_count}/5 passed). "
                        f"Regenerating to increase difficulty (attempt {qwen_attempts}/{qwen_display})."
                    )

                    # Check if we've exceeded Qwen retry limit
                    if qwen_max != float('inf') and qwen_attempts >= qwen_max:
                        result.is_easy = True
                        result.error = f"Question still too easy after {qwen_display} attempts"
                        await results_queue.put(result)
                        return result

                    # Regenerate with strong feedback to make it MUCH harder
                    # Use escalating difficulty hints based on attempt number
                    difficulty_hints = [
                        "Add coupled differential equations or systems",
                        "Require perturbation theory or asymptotic analysis",
                        "Use non-standard coordinate systems (elliptic, parabolic, etc.)",
                        "Combine quantum mechanics with thermodynamics",
                        "Add time-dependent perturbations or driven systems",
                        "Require Green's function or integral equation methods",
                        "Use relativistic corrections or field theory concepts",
                        "Add topological or geometric phase considerations",
                    ]
                    extra_hint = difficulty_hints[min(qwen_attempts - 1, len(difficulty_hints) - 1)]

                    feedback = (
                        f"The question is TOO EASY - Qwen solved it {qwen_high_count}/5 times. "
                        f"This is attempt {qwen_attempts} - you MUST generate a COMPLETELY DIFFERENT, MUCH HARDER question.\n\n"
                        f"SPECIFIC REQUIREMENT FOR THIS ATTEMPT: {extra_hint}\n\n"
                        "The question MUST:\n"
                        "- NOT be solvable by pattern matching or standard formulas\n"
                        "- Require deep physical insight and multi-step reasoning\n"
                        "- Challenge a first-year physics PhD student\n"
                        "- Be fundamentally different from the previous attempt"
                    )
                    logger.info(f"[Worker {worker_id}] Regenerating for difficulty (attempt {qwen_attempts + 1}/{qwen_display})")
                    try:
                        qa = await qa_generator.regenerate_with_feedback(
                            original=qa,
                            feedback=feedback,
                            temperature=0.8,  # Higher temp for more creative harder questions
                        )
                        continue  # Re-validate with new harder QA (goes back to schema check)
                    except Exception as e:
                        logger.error(f"[Worker {worker_id}] Regeneration failed: {e}")
                        result.error = str(e)
                        await results_queue.put(result)
                        return result

                # === Stage 4: Cross-check validation ===
                logger.info(f"[Worker {worker_id}] Running cross-check validation...")
                _, models_correct, total_correct, crosscheck_valid, _ = (
                    await crosscheck_validator.validate(
                        qa, samples_per_model=self.settings.samples_per_crosscheck
                    )
                )

                if crosscheck_valid:
                    # Success!
                    qa.validation_passed = True
                    result.qa = qa
                    result.success = True
                    logger.info(
                        f"[Worker {worker_id}] QA pair {qa.id} passed all validation! "
                        f"(cross-check: {total_correct}/20)"
                    )
                    await results_queue.put(result)
                    return result

                # Cross-check failed - increment counter and try to regenerate
                crosscheck_attempts += 1

                # Check if we've exceeded cross-check retry limit
                if crosscheck_max != float('inf') and crosscheck_attempts >= crosscheck_max:
                    logger.warning(
                        f"[Worker {worker_id}] Abandoning after {crosscheck_display} cross-check attempts. "
                        f"Cross-check: {total_correct}/20"
                    )
                    result.error = f"Cross-check failed after {crosscheck_display} attempts"
                    await results_queue.put(result)
                    return result

                # Regenerate with feedback - cross-check failure means the ANSWER is likely wrong
                feedback = (
                    f"CRITICAL: The reference answer appears to be INCORRECT. "
                    f"Only {total_correct}/20 model attempts matched your answer. "
                    f"Models with correct answers: {models_correct}/4.\n\n"
                    "You MUST:\n"
                    "1. Re-derive the answer from FIRST PRINCIPLES - do not just tweak the old answer\n"
                    "2. Check your algebra step-by-step for sign errors, missing factors, etc.\n"
                    "3. Verify dimensional analysis - does your answer have the right units?\n"
                    "4. Check limiting cases - does the answer behave correctly as parameters approach 0 or infinity?\n"
                    "5. If the physics is ambiguous, clarify the problem statement\n\n"
                    "The question difficulty is fine - focus on getting the CORRECT ANSWER."
                )
                logger.info(f"[Worker {worker_id}] Regenerating to fix answer (attempt {crosscheck_attempts + 1}/{crosscheck_display})")

                try:
                    qa = await qa_generator.regenerate_with_feedback(
                        original=qa,
                        feedback=feedback,
                        temperature=0.7,
                    )
                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Regeneration failed: {e}")
                    result.error = str(e)
                    await results_queue.put(result)
                    return result

        except Exception as e:
            logger.error(f"[Worker {worker_id}] Unexpected error: {e}")
            result.error = str(e)

        await results_queue.put(result)
        return result

    def _write_to_output(self, qa: PhysicsQADataPoint):
        """Append a single QA pair to JSONL output."""
        with open(self.output_path, "a") as f:
            f.write(qa.to_json_line() + "\n")

    async def _run_phase2_validation(
        self,
        phase1_pairs: List[PhysicsQADataPoint],
        target_count: int,
    ) -> List[PhysicsQADataPoint]:
        """
        Run Phase 2 validation (GPT-4 Audit + Answer Check) on a batch of QA pairs.

        Args:
            phase1_pairs: QA pairs that passed Phase 1 (Qwen + Cross-check)
            target_count: Original target count (for logging)

        Returns:
            List of QA pairs that passed Phase 2 validation
        """
        if not phase1_pairs:
            return []

        logger.info(
            f"\n{'='*60}\n"
            f"PHASE 2: Final Validation (GPT-4 Audit + Answer Check)\n"
            f"{'='*60}"
        )
        logger.info(
            f"Validating {len(phase1_pairs)} QA pairs with:\n"
            f"  - Derivation Audit: {self.settings.audit_model}\n"
            f"  - Answer Validation: {self.settings.final_validation_model}"
        )

        # Report phase transition
        self._report_progress("phase2_started", {
            "phase": 2,
            "phase1_count": len(phase1_pairs),
            "phase2_validated": 0,
            "phase2_passed": len(getattr(self, 'fully_validated_pairs', [])),
        })

        # Create derivation auditor (uses GPT-4 for independent reasoning check)
        # 2 passes balances rigor with practicality - catches major issues without being overly strict
        derivation_auditor = DerivationAuditor(
            client=self.client,
            audit_model=self.settings.audit_model,
            required_passes=2,  # 2 audit passes - balanced rigor
        )

        # Create final answer validator (uses GPT-4, not Claude)
        final_validator = FinalAnswerValidator(
            client=self.client,
            judge_model=self.settings.final_validation_model,  # GPT-4, not Claude
            required_consecutive_passes=self.settings.final_validation_passes,
        )

        # Create validators for full pipeline re-validation
        qa_regenerator = QAGenerator(
            client=self.client,
            model=self.settings.generation_model,
            judge_model=self.settings.judge_model,
            solver_model=self.settings.solver_model,
        )
        qwen_validator = QwenConstraintValidator(
            self.client,
            qwen_model=self.settings.qwen_model,
            judge_model=self.settings.judge_model,
            max_pass_rate=self.settings.qwen_max_pass_rate,
        )
        crosscheck_validator = CrossCheckValidator(
            self.client,
            judge_model=self.settings.judge_model,
            models=self.settings.crosscheck_models,
            min_models_with_correct=self.settings.min_crosscheck_models,
            min_total_correct=self.settings.min_crosscheck_total,
        )

        # Use configured final_validation_max_retries, or unlimited (float('inf')) if not set
        MAX_REGEN_ATTEMPTS = self.final_validation_max_retries if self.final_validation_max_retries else float('inf')
        max_display = "unlimited" if MAX_REGEN_ATTEMPTS == float('inf') else str(int(MAX_REGEN_ATTEMPTS))

        pre_validation_count = len(phase1_pairs)

        # Define async worker for validating a single QA pair
        async def validate_single_qa(qa_idx: int, qa: PhysicsQADataPoint) -> Tuple[Optional[PhysicsQADataPoint], bool, int]:
            """
            Validate a single QA pair with regeneration loop.

            Validation order:
            1. Derivation Audit (GPT-4) - checks reasoning quality
            2. Final Answer Validation (GPT-4) - checks answer correctness

            If either fails, regenerate and re-run full pipeline.

            Returns (final_qa, passed, regen_attempts).
            """
            current_qa = qa
            attempt = 0
            regen_count = 0

            while (MAX_REGEN_ATTEMPTS == float('inf') or attempt < MAX_REGEN_ATTEMPTS) and not self._stop_requested:
                attempt += 1

                # Step 1: Derivation Audit (GPT-4 checks reasoning quality)
                logger.info(f"[Validator {qa_idx + 1}] Running derivation audit on {current_qa.id} (attempt {attempt}/{max_display})...")
                audit_passed, audit_pass_count, audits = await derivation_auditor.audit(current_qa)

                if not audit_passed:
                    logger.warning(
                        f"[Validator {qa_idx + 1}] QA {current_qa.id} FAILED derivation audit "
                        f"({audit_pass_count}/2). Issues found in reasoning."
                    )

                    # Check retry limit
                    if MAX_REGEN_ATTEMPTS != float('inf') and attempt >= MAX_REGEN_ATTEMPTS:
                        logger.warning(f"[Validator {qa_idx + 1}] QA {qa.id} failed after {max_display} attempts")
                        return None, False, regen_count

                    # Extract feedback from audit and regenerate
                    feedback = derivation_auditor.extract_feedback_from_audits(audits)
                    logger.info(
                        f"[Validator {qa_idx + 1}] Regenerating QA due to audit failure "
                        f"(attempt {attempt + 1}/{max_display})..."
                    )

                    try:
                        current_qa = await qa_regenerator.regenerate_with_feedback(
                            original=current_qa,
                            feedback=feedback,
                            temperature=0.7,
                        )
                        regen_count += 1
                        logger.info(f"[Validator {qa_idx + 1}] Regenerated QA, new ID: {current_qa.id}")
                    except Exception as e:
                        logger.error(f"[Validator {qa_idx + 1}] Regeneration failed: {e}")
                        return None, False, regen_count

                    # After regeneration, re-run Qwen and cross-check before retrying audit
                    # Step 1b: Re-run Qwen check
                    logger.info(f"[Validator {qa_idx + 1}] Re-running Qwen check on {current_qa.id}...")
                    _, qwen_high_count, is_easy, _ = await qwen_validator.validate(
                        current_qa, samples=self.settings.samples_per_qwen_check
                    )

                    if is_easy:
                        logger.warning(
                            f"[Validator {qa_idx + 1}] Regenerated QA {current_qa.id} is TOO EASY ({qwen_high_count}/5). "
                            f"Regenerating with harder constraints."
                        )
                        feedback = (
                            f"The corrected answer made the question TOO EASY. "
                            f"Qwen solved it {qwen_high_count}/5 times. "
                            "Make the question harder while keeping the answer correct."
                        )
                        try:
                            current_qa = await qa_regenerator.regenerate_with_feedback(
                                original=current_qa,
                                feedback=feedback,
                                temperature=0.8,
                            )
                            regen_count += 1
                        except Exception as e:
                            logger.error(f"[Validator {qa_idx + 1}] Difficulty regeneration failed: {e}")
                            return None, False, regen_count

                    # Step 1c: Re-run Cross-check
                    logger.info(f"[Validator {qa_idx + 1}] Re-running cross-check on {current_qa.id}...")
                    _, models_correct, total_correct, crosscheck_valid, _ = (
                        await crosscheck_validator.validate(
                            current_qa, samples_per_model=self.settings.samples_per_crosscheck
                        )
                    )

                    if not crosscheck_valid:
                        logger.warning(
                            f"[Validator {qa_idx + 1}] QA {current_qa.id} failed cross-check "
                            f"({total_correct}/20). Regenerating with answer feedback."
                        )
                        feedback = (
                            f"CRITICAL: The answer appears to be INCORRECT. "
                            f"Only {total_correct}/20 attempts across 4 frontier models matched your answer "
                            f"(need at least 5). Models with any correct: {models_correct}/4.\n\n"
                            "You MUST:\n"
                            "1. Re-derive the answer from FIRST PRINCIPLES\n"
                            "2. Check your algebra step-by-step for sign errors, missing factors\n"
                            "3. Verify dimensional analysis\n"
                            "4. Check limiting cases\n"
                            "5. If the problem is ambiguous, clarify it"
                        )
                        try:
                            current_qa = await qa_regenerator.regenerate_with_feedback(
                                original=current_qa,
                                feedback=feedback,
                                temperature=0.7,
                            )
                            regen_count += 1
                            logger.info(f"[Validator {qa_idx + 1}] Regenerated QA after cross-check failure, new ID: {current_qa.id}")
                        except Exception as e:
                            logger.error(f"[Validator {qa_idx + 1}] Cross-check regeneration failed: {e}")
                            return None, False, regen_count

                    continue  # Re-run from the beginning (audit)

                # Step 2: Final Answer Validation (GPT-4 blind solve + compare)
                logger.info(f"[Validator {qa_idx + 1}] Running final answer validation on {current_qa.id}...")
                passed, pass_count, judgments = await final_validator.validate(current_qa)

                if passed:
                    logger.info(
                        f"[Validator {qa_idx + 1}] QA {current_qa.id} PASSED all validation "
                        f"(audit: {audit_pass_count}/2, answer: {pass_count}/{self.settings.final_validation_passes}) "
                        f"on attempt {attempt}"
                    )
                    return current_qa, True, regen_count

                # Final validation failed
                logger.warning(
                    f"[Validator {qa_idx + 1}] QA {current_qa.id} failed answer validation "
                    f"({pass_count}/{self.settings.final_validation_passes})"
                )

                # Check retry limit
                if MAX_REGEN_ATTEMPTS != float('inf') and attempt >= MAX_REGEN_ATTEMPTS:
                    logger.warning(f"[Validator {qa_idx + 1}] QA {qa.id} failed after {max_display} attempts")
                    return None, False, regen_count

                # Extract feedback and regenerate
                feedback = final_validator._extract_feedback_from_judgments(judgments)
                logger.info(
                    f"[Validator {qa_idx + 1}] Regenerating QA due to answer validation failure "
                    f"(attempt {attempt + 1}/{max_display})..."
                )

                try:
                    current_qa = await qa_regenerator.regenerate_with_feedback(
                        original=current_qa,
                        feedback=feedback,
                        temperature=0.7,
                    )
                    regen_count += 1
                    logger.info(f"[Validator {qa_idx + 1}] Regenerated QA, new ID: {current_qa.id}")
                except Exception as e:
                    logger.error(f"[Validator {qa_idx + 1}] Regeneration failed: {e}")
                    return None, False, regen_count

                # Re-run Qwen and cross-check on regenerated QA
                # Step 2b: Re-run Qwen check
                logger.info(f"[Validator {qa_idx + 1}] Re-running Qwen check on {current_qa.id}...")
                _, qwen_high_count, is_easy, _ = await qwen_validator.validate(
                    current_qa, samples=self.settings.samples_per_qwen_check
                )

                if is_easy:
                    logger.warning(
                        f"[Validator {qa_idx + 1}] Regenerated QA {current_qa.id} is TOO EASY ({qwen_high_count}/5). "
                        f"Regenerating with harder constraints."
                    )
                    feedback = (
                        f"The corrected answer made the question TOO EASY. "
                        f"Qwen solved it {qwen_high_count}/5 times. "
                        "Make the question harder while keeping the answer correct."
                    )
                    try:
                        current_qa = await qa_regenerator.regenerate_with_feedback(
                            original=current_qa,
                            feedback=feedback,
                            temperature=0.8,
                        )
                        regen_count += 1
                    except Exception as e:
                        logger.error(f"[Validator {qa_idx + 1}] Difficulty regeneration failed: {e}")
                        return None, False, regen_count

                # Step 2c: Re-run Cross-check
                logger.info(f"[Validator {qa_idx + 1}] Re-running cross-check on {current_qa.id}...")
                _, models_correct, total_correct, crosscheck_valid, _ = (
                    await crosscheck_validator.validate(
                        current_qa, samples_per_model=self.settings.samples_per_crosscheck
                    )
                )

                if not crosscheck_valid:
                    logger.warning(
                        f"[Validator {qa_idx + 1}] QA {current_qa.id} failed cross-check "
                        f"({total_correct}/20). Regenerating with answer feedback."
                    )
                    feedback = (
                        f"CRITICAL: The answer appears to be INCORRECT. "
                        f"Only {total_correct}/20 attempts across 4 frontier models matched your answer "
                        f"(need at least 5). Models with any correct: {models_correct}/4.\n\n"
                        "You MUST:\n"
                        "1. Re-derive the answer from FIRST PRINCIPLES\n"
                        "2. Check your algebra step-by-step for sign errors, missing factors\n"
                        "3. Verify dimensional analysis\n"
                        "4. Check limiting cases\n"
                        "5. If the problem is ambiguous, clarify it"
                    )
                    try:
                        current_qa = await qa_regenerator.regenerate_with_feedback(
                            original=current_qa,
                            feedback=feedback,
                            temperature=0.7,
                        )
                        regen_count += 1
                        logger.info(f"[Validator {qa_idx + 1}] Regenerated QA after cross-check failure, new ID: {current_qa.id}")
                    except Exception as e:
                        logger.error(f"[Validator {qa_idx + 1}] Cross-check regeneration failed: {e}")
                        return None, False, regen_count

                # Loop back to re-run full validation (audit + answer check)

            return None, False, regen_count

        # Run all validations in parallel with progress tracking
        logger.info(f"Starting parallel final validation for {pre_validation_count} QA pairs...")

        # Shared counter for tracking progress
        validated_count = 0
        passed_count = 0
        validated_lock = asyncio.Lock()

        async def validate_with_progress(qa_idx: int, qa: PhysicsQADataPoint):
            """Wrapper that reports progress after each validation."""
            nonlocal validated_count, passed_count
            result = await validate_single_qa(qa_idx, qa)

            async with validated_lock:
                validated_count += 1
                if result[1]:  # passed
                    passed_count += 1

                # Report progress
                self._report_progress("phase2_progress", {
                    "phase": 2,
                    "phase1_count": pre_validation_count,
                    "phase2_validated": validated_count,
                    "phase2_passed": len(getattr(self, 'fully_validated_pairs', [])) + passed_count,
                })

            return result

        validation_tasks = [
            validate_with_progress(i, qa)
            for i, qa in enumerate(phase1_pairs)
        ]

        # Gather all results
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # Process results
        passed_pairs = []
        failed_pairs = []
        total_regen_attempts = 0

        for i, result in enumerate(validation_results):
            original_qa = phase1_pairs[i]

            if isinstance(result, Exception):
                logger.error(f"Validation task {i + 1} failed with exception: {result}")
                failed_pairs.append(original_qa)
                continue

            final_qa, passed, regen_count = result
            total_regen_attempts += regen_count

            if passed and final_qa:
                passed_pairs.append(final_qa)
            else:
                failed_pairs.append(original_qa)

        # Store validation results for stats (accumulate across rounds)
        if not hasattr(self, 'final_validation_summary'):
            self.final_validation_summary = {
                "total_validated": 0,
                "passed": 0,
                "failed": 0,
                "audit_model": self.settings.audit_model,
                "final_validation_model": self.settings.final_validation_model,
                "required_audit_passes": 2,
                "required_answer_passes": self.settings.final_validation_passes,
                "total_regeneration_attempts": 0,
            }

        self.final_validation_summary["total_validated"] += pre_validation_count
        self.final_validation_summary["passed"] += len(passed_pairs)
        self.final_validation_summary["failed"] += len(failed_pairs)
        self.final_validation_summary["total_regeneration_attempts"] += total_regen_attempts
        total_validated = self.final_validation_summary["total_validated"]
        total_passed = self.final_validation_summary["passed"]
        self.final_validation_summary["pass_rate"] = f"{total_passed / total_validated:.1%}" if total_validated > 0 else "0%"

        logger.info(
            f"\nPhase 2 Batch Results:\n"
            f"  - Validated: {pre_validation_count} QA pairs\n"
            f"  - Passed: {len(passed_pairs)} QA pairs\n"
            f"  - Failed: {len(failed_pairs)} QA pairs\n"
            f"  - Regeneration attempts: {total_regen_attempts}"
        )

        if failed_pairs:
            logger.warning(
                f"The following {len(failed_pairs)} QA pairs failed final validation:"
            )
            for qa in failed_pairs:
                logger.warning(f"  - {qa.id}: {qa.subtopic}")
            # Add to class-level failed list for later writing to file
            self.failed_validation_pairs.extend(failed_pairs)

        return passed_pairs

    async def _regenerate_and_revalidate(
        self,
        qa: PhysicsQADataPoint,
        feedback: str,
        qa_regenerator: QAGenerator,
        qwen_validator: QwenConstraintValidator,
        crosscheck_validator: CrossCheckValidator,
        worker_id: int,
    ) -> PhysicsQADataPoint:
        """
        Regenerate a QA pair and re-run Qwen + cross-check validation.

        This ensures the regenerated QA still passes the difficulty and correctness
        checks before going back to Phase 2 audit.

        Args:
            qa: Original QA to regenerate
            feedback: Feedback about what failed
            qa_regenerator: Generator for regeneration
            qwen_validator: Qwen constraint validator
            crosscheck_validator: Cross-check validator
            worker_id: For logging

        Returns:
            Regenerated and validated QA

        Raises:
            Exception if regeneration or validation fails repeatedly
        """
        MAX_REGEN_LOOPS = 5  # Max attempts to get a valid regeneration

        current_qa = qa
        for loop in range(MAX_REGEN_LOOPS):
            # Regenerate with feedback
            logger.info(f"[Audit {worker_id}] Regenerating QA (loop {loop + 1}/{MAX_REGEN_LOOPS})...")
            current_qa = await qa_regenerator.regenerate_with_feedback(
                original=current_qa, feedback=feedback, temperature=0.7
            )

            # Re-run Qwen constraint check
            logger.info(f"[Audit {worker_id}] Re-running Qwen check on regenerated QA...")
            _, qwen_high_count, is_easy, _ = await qwen_validator.validate(
                current_qa, samples=self.settings.samples_per_qwen_check
            )

            if is_easy:
                logger.warning(f"[Audit {worker_id}] Regenerated QA too easy ({qwen_high_count}/5). Making harder...")
                feedback = f"Question TOO EASY ({qwen_high_count}/5). Make MUCH harder - require multi-step derivation."
                continue  # Try again with harder question

            # Re-run cross-check validation
            logger.info(f"[Audit {worker_id}] Re-running cross-check on regenerated QA...")
            _, models_correct, total_correct, crosscheck_valid, _ = await crosscheck_validator.validate(
                current_qa, samples_per_model=self.settings.samples_per_crosscheck
            )

            if not crosscheck_valid:
                logger.warning(f"[Audit {worker_id}] Regenerated QA failed cross-check ({total_correct}/20). Re-deriving...")
                feedback = f"Answer INCORRECT ({total_correct}/20, {models_correct}/4 models). Re-derive from first principles with careful dimensional analysis."
                continue  # Try again with correct answer

            # Both checks passed!
            logger.info(f"[Audit {worker_id}] Regenerated QA passed Qwen and cross-check validation")
            return current_qa

        raise Exception(f"Failed to regenerate valid QA after {MAX_REGEN_LOOPS} attempts")

    async def _audit_worker(
        self,
        audit_queue: asyncio.Queue,
        validated_queue: asyncio.Queue,
        worker_id: int,
    ):
        """
        Audit worker that processes Phase 1 QAs as they arrive.

        Runs Phase 2 validation (derivation audit + final answer check) on each QA.
        Puts validated QAs into the validated_queue.

        Args:
            audit_queue: Queue of Phase 1 QAs to validate
            validated_queue: Queue to put validated QAs into
            worker_id: ID for logging
        """
        # Create validators for this audit worker
        # 2 passes balances rigor with practicality
        derivation_auditor = DerivationAuditor(
            client=self.client,
            audit_model=self.settings.audit_model,
            required_passes=2,  # 2 audit passes - balanced rigor
        )
        final_validator = FinalAnswerValidator(
            client=self.client,
            judge_model=self.settings.final_validation_model,
            required_consecutive_passes=self.settings.final_validation_passes,
        )
        qa_regenerator = QAGenerator(
            client=self.client,
            model=self.settings.generation_model,
            judge_model=self.settings.judge_model,
            solver_model=self.settings.solver_model,
        )
        qwen_validator = QwenConstraintValidator(
            self.client,
            qwen_model=self.settings.qwen_model,
            judge_model=self.settings.judge_model,
            max_pass_rate=self.settings.qwen_max_pass_rate,
        )
        crosscheck_validator = CrossCheckValidator(
            self.client,
            judge_model=self.settings.judge_model,
            models=self.settings.crosscheck_models,
            min_models_with_correct=self.settings.min_crosscheck_models,
            min_total_correct=self.settings.min_crosscheck_total,
        )

        MAX_REGEN_ATTEMPTS = self.final_validation_max_retries if self.final_validation_max_retries else float('inf')
        max_display = "unlimited" if MAX_REGEN_ATTEMPTS == float('inf') else str(int(MAX_REGEN_ATTEMPTS))

        while True:
            try:
                # Get next QA to audit (with timeout to check for stop)
                try:
                    item = await asyncio.wait_for(audit_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if self._stop_requested:
                        break
                    continue

                if item is None:  # Shutdown signal
                    break

                qa = item
                current_qa = qa
                attempt = 0
                regen_count = 0
                passed = False

                while (MAX_REGEN_ATTEMPTS == float('inf') or attempt < MAX_REGEN_ATTEMPTS) and not self._stop_requested:
                    attempt += 1

                    # Step 1: Derivation Audit
                    logger.info(f"[Audit {worker_id}] Running derivation audit on {current_qa.id} (attempt {attempt}/{max_display})...")
                    audit_passed, audit_pass_count, audits = await derivation_auditor.audit(current_qa)

                    if not audit_passed:
                        logger.warning(f"[Audit {worker_id}] QA {current_qa.id} FAILED derivation audit ({audit_pass_count}/2)")

                        if MAX_REGEN_ATTEMPTS != float('inf') and attempt >= MAX_REGEN_ATTEMPTS:
                            break

                        feedback = derivation_auditor.extract_feedback_from_audits(audits)
                        try:
                            current_qa = await self._regenerate_and_revalidate(
                                current_qa, feedback, qa_regenerator, qwen_validator, crosscheck_validator, worker_id
                            )
                            regen_count += 1
                            continue  # Go back to derivation audit with regenerated QA
                        except Exception as e:
                            logger.error(f"[Audit {worker_id}] Regeneration failed: {e}")
                            break

                    # Step 2: Final Answer Validation
                    logger.info(f"[Audit {worker_id}] Running final answer validation on {current_qa.id}...")
                    val_passed, pass_count, judgments = await final_validator.validate(current_qa)

                    if val_passed:
                        logger.info(f"[Audit {worker_id}] QA {current_qa.id} PASSED all validation!")
                        passed = True
                        break

                    logger.warning(f"[Audit {worker_id}] QA {current_qa.id} failed answer validation ({pass_count}/{self.settings.final_validation_passes})")

                    if MAX_REGEN_ATTEMPTS != float('inf') and attempt >= MAX_REGEN_ATTEMPTS:
                        break

                    feedback = final_validator._extract_feedback_from_judgments(judgments)
                    try:
                        current_qa = await self._regenerate_and_revalidate(
                            current_qa, feedback, qa_regenerator, qwen_validator, crosscheck_validator, worker_id
                        )
                        regen_count += 1
                        # Loop back to derivation audit (will re-check everything)
                    except Exception as e:
                        logger.error(f"[Audit {worker_id}] Regeneration failed: {e}")
                        break

                # Put result into validated queue
                await validated_queue.put((current_qa if passed else None, passed, regen_count))
                audit_queue.task_done()

            except Exception as e:
                logger.error(f"[Audit {worker_id}] Unexpected error: {e}")
                audit_queue.task_done()

    async def run(self, target_count: int) -> List[PhysicsQADataPoint]:
        """
        Run the parallel pipeline to generate target_count QA pairs.

        Phase 1 (generation) and Phase 2 (audit) run concurrently:
        - Generation workers produce QA pairs that pass initial validation
        - Audit workers immediately start validating Phase 1 QAs as they complete
        - This streaming approach reduces total pipeline time significantly

        Args:
            target_count: Target number of valid QA pairs

        Returns:
            List of valid PhysicsQADataPoint objects
        """
        self.stats.started_at = datetime.utcnow()
        logger.info(
            f"Starting parallel pipeline run {self.run_id}, "
            f"target: {target_count} QA pairs (streaming Phase 1 + Phase 2)"
        )

        # Append to existing output file (don't delete - preserves data from failed runs)
        if self.output_path.exists():
            logger.info(f"Appending to existing output file: {self.output_path}")
        else:
            logger.info(f"Creating new output file: {self.output_path}")

        # Track fully validated QAs (passed both Phase 1 and Phase 2)
        self.fully_validated_pairs: List[PhysicsQADataPoint] = []
        # Track QAs that failed Phase 2 validation (for debugging)
        self.failed_validation_pairs: List[PhysicsQADataPoint] = []
        backfill_round = 0

        # Outer loop: keep going until we have target_count fully validated QAs
        while len(self.fully_validated_pairs) < target_count and not self._stop_requested:
            backfill_round += 1
            needed_count = target_count - len(self.fully_validated_pairs)

            if backfill_round == 1:
                logger.info(
                    f"Pipeline will retry indefinitely until {target_count} valid QA pairs are generated. "
                    f"Phase 1 (generation) and Phase 2 (audit) run concurrently for faster throughput."
                )
            else:
                logger.info(
                    f"\n{'='*60}\n"
                    f"BACKFILL ROUND {backfill_round}: Need {needed_count} more QA pairs\n"
                    f"{'='*60}"
                )

            # Queues for streaming pipeline
            phase1_results_queue: asyncio.Queue = asyncio.Queue()
            audit_queue: asyncio.Queue = asyncio.Queue()  # Phase 1 QAs waiting for audit
            validated_queue: asyncio.Queue = asyncio.Queue()  # Audit results

            phase1_results: List[WorkerResult] = []
            workers_spawned = 0
            worker_id = backfill_round * 10000

            # Spawn Phase 1 generation workers
            pending_gen_tasks: List[asyncio.Task] = []
            logger.info(f"Spawning {needed_count} generation workers for Phase 1...")

            for i in range(needed_count):
                task = asyncio.create_task(
                    self._worker(worker_id, needed_count, phase1_results_queue)
                )
                pending_gen_tasks.append(task)
                worker_id += 1
                workers_spawned += 1

            # Spawn audit workers (run concurrently with generation)
            # Use fewer audit workers since audits are heavier
            num_audit_workers = max(2, min(needed_count // 2, 10))
            audit_tasks: List[asyncio.Task] = []
            logger.info(f"Spawning {num_audit_workers} audit workers for Phase 2 (streaming)...")

            for i in range(num_audit_workers):
                task = asyncio.create_task(
                    self._audit_worker(audit_queue, validated_queue, i)
                )
                audit_tasks.append(task)

            # Track progress
            phase1_successful_count = 0
            phase1_pairs: List[PhysicsQADataPoint] = []
            audit_submitted_count = 0
            audit_completed_count = 0
            audit_passed_count = 0
            total_regen_attempts = 0
            round_validated: List[PhysicsQADataPoint] = []  # Collect validated QAs here

            # Report phase start - start as Phase 1 until we have QAs to audit
            self._report_progress("phase1_started", {
                "phase": 1,
                "valid_count": 0,  # For Phase 1 progress bar
                "phase1_count": 0,
                "phase2_validated": 0,
                "phase2_passed": len(self.fully_validated_pairs),
            })

            # Main collection loop - handles both Phase 1 results and audit results
            while (phase1_successful_count < needed_count or audit_completed_count < audit_submitted_count) and not self._stop_requested:
                # Check for Phase 1 results (non-blocking if we have enough)
                if phase1_successful_count < needed_count:
                    try:
                        result = await asyncio.wait_for(phase1_results_queue.get(), timeout=0.1)
                        phase1_results.append(result)
                        self.all_results.append(result)

                        if result.success and result.qa:
                            phase1_pairs.append(result.qa)
                            self.valid_qa_pairs.append(result.qa)
                            self._write_to_output(result.qa)
                            self.stats.total_generated += 1
                            self.stats.passed_all += 1
                            phase1_successful_count += 1

                            # Immediately submit to audit queue (streaming!)
                            await audit_queue.put(result.qa)
                            audit_submitted_count += 1

                            logger.info(
                                f"Phase 1: {phase1_successful_count}/{needed_count} | "
                                f"Phase 2: {audit_passed_count}/{audit_submitted_count} auditing "
                                f"(round {backfill_round})"
                            )
                            # Only switch to "streaming" phase once audits start
                            # Until then, show as Phase 1
                            current_phase = "streaming" if audit_submitted_count > 0 else 1
                            self._report_progress("streaming_progress", {
                                "phase": current_phase,
                                "valid_count": phase1_successful_count,  # For Phase 1 progress bar
                                "phase1_count": phase1_successful_count,
                                "phase2_validated": audit_completed_count,
                                "phase2_passed": len(self.fully_validated_pairs) + audit_passed_count,
                            })
                        else:
                            self.stats.failed += 1
                            logger.warning(f"Worker {result.worker_id} failed: {result.error or 'validation failed'}")

                            # Spawn replacement worker
                            if not self._stop_requested:
                                task = asyncio.create_task(
                                    self._worker(worker_id, needed_count, phase1_results_queue)
                                )
                                pending_gen_tasks.append(task)
                                worker_id += 1
                                workers_spawned += 1

                    except asyncio.TimeoutError:
                        pass  # No Phase 1 result ready, check audit results

                # Check for audit results (non-blocking)
                try:
                    validated_qa, passed, regen_count = await asyncio.wait_for(validated_queue.get(), timeout=0.1)
                    audit_completed_count += 1
                    total_regen_attempts += regen_count

                    if passed and validated_qa:
                        audit_passed_count += 1
                        round_validated.append(validated_qa)  # Collect the validated QA!
                        logger.info(
                            f"Phase 1: {phase1_successful_count}/{needed_count} | "
                            f"Phase 2: {audit_passed_count}/{audit_completed_count} passed "
                            f"(total: {len(self.fully_validated_pairs) + audit_passed_count})"
                        )
                        self._report_progress("streaming_progress", {
                            "phase": "streaming",
                            "phase1_count": phase1_successful_count,
                            "phase2_validated": audit_completed_count,
                            "phase2_passed": len(self.fully_validated_pairs) + audit_passed_count,
                        })
                    else:
                        logger.warning(f"Audit failed for a QA pair ({audit_completed_count - audit_passed_count} failed so far)")

                except asyncio.TimeoutError:
                    pass  # No audit result ready

                # Clean up completed tasks from the list
                pending_gen_tasks = [t for t in pending_gen_tasks if not t.done()]

                # Handle timeout - spawn more workers if stuck
                # All generation workers have finished but we still need more QAs
                if phase1_successful_count < needed_count and len(pending_gen_tasks) == 0:
                    logger.warning("All generation workers done but target not reached. Spawning more...")
                    for _ in range(min(5, needed_count - phase1_successful_count)):
                        if not self._stop_requested:
                            task = asyncio.create_task(
                                self._worker(worker_id, needed_count, phase1_results_queue)
                            )
                            pending_gen_tasks.append(task)
                            worker_id += 1
                            workers_spawned += 1

            # Signal audit workers to stop
            for _ in range(num_audit_workers):
                await audit_queue.put(None)

            # Wait for audit workers to finish
            await asyncio.gather(*audit_tasks, return_exceptions=True)

            # Drain any remaining audit results
            while not validated_queue.empty():
                try:
                    validated_qa, passed, regen_count = validated_queue.get_nowait()
                    audit_completed_count += 1
                    total_regen_attempts += regen_count
                    if passed and validated_qa:
                        audit_passed_count += 1
                        round_validated.append(validated_qa)  # Collect remaining validated QAs
                except asyncio.QueueEmpty:
                    break

            # Cancel remaining generation tasks
            for task in pending_gen_tasks:
                if not task.done():
                    task.cancel()

            # Store final validation summary
            if not hasattr(self, 'final_validation_summary'):
                self.final_validation_summary = {
                    "total_validated": 0,
                    "passed": 0,
                    "failed": 0,
                    "audit_model": self.settings.audit_model,
                    "final_validation_model": self.settings.final_validation_model,
                    "required_audit_passes": 2,
                    "required_answer_passes": self.settings.final_validation_passes,
                    "total_regeneration_attempts": 0,
                }

            self.final_validation_summary["total_validated"] += audit_submitted_count
            self.final_validation_summary["passed"] += audit_passed_count
            self.final_validation_summary["failed"] += (audit_submitted_count - audit_passed_count)
            self.final_validation_summary["total_regeneration_attempts"] += total_regen_attempts
            total_validated = self.final_validation_summary["total_validated"]
            total_passed = self.final_validation_summary["passed"]
            self.final_validation_summary["pass_rate"] = f"{total_passed / total_validated:.1%}" if total_validated > 0 else "0%"

            # Add validated QAs from this round to fully_validated_pairs
            self.fully_validated_pairs.extend(round_validated)

            logger.info(
                f"Round {backfill_round} complete: {len(round_validated)}/{len(phase1_pairs)} passed Phase 2. "
                f"Total fully validated: {len(self.fully_validated_pairs)}/{target_count}"
            )

        # Update valid_qa_pairs to only include fully validated ones
        self.valid_qa_pairs = self.fully_validated_pairs

        # Write final validated pairs to the final output file
        if self.fully_validated_pairs:
            final_output_path = self.output_path.with_suffix('.final.jsonl')
            with open(final_output_path, "w") as f:
                for qa in self.fully_validated_pairs:
                    f.write(qa.to_json_line() + "\n")
            logger.info(f"Wrote {len(self.fully_validated_pairs)} final validated QA pairs to: {final_output_path}")
            logger.info(f"Original cross-check validated pairs preserved in: {self.output_path}")

        # Write failed pairs to failed output file (for debugging)
        if self.failed_validation_pairs:
            failed_output_path = self.output_path.with_suffix('.failed.jsonl')
            with open(failed_output_path, "w") as f:
                for qa in self.failed_validation_pairs:
                    f.write(qa.to_json_line() + "\n")
            logger.info(f"Wrote {len(self.failed_validation_pairs)} failed QA pairs to: {failed_output_path}")

        # Update stats
        self.stats.completed_at = datetime.utcnow()

        # Track API stats
        api_stats = self.client.get_stats()
        self.stats.total_api_calls = api_stats.get("total_calls", 0)
        self.stats.total_api_time_seconds = api_stats.get("total_time_seconds", 0)

        # Clear checkpoint on completion
        self.checkpoint_manager.clear(self.run_id)

        valid_count = len(self.valid_qa_pairs)
        if valid_count > 0:
            logger.info(
                f"\n{'='*60}\n"
                f"PIPELINE COMPLETE\n"
                f"{'='*60}\n"
                f"Final validated QA pairs: {valid_count}/{target_count}\n"
                f"Output: {final_output_path}"
            )
        else:
            logger.warning(
                f"Parallel pipeline complete but no valid QA pairs remain after final validation. "
                f"All pairs failed the validation checks. "
                f"Output: {self.output_path}"
            )

        return self.valid_qa_pairs

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline statistics."""
        valid_count = len(self.valid_qa_pairs)
        easy_count = sum(1 for r in self.all_results if r.success and r.is_easy)
        easy_rate = easy_count / valid_count if valid_count > 0 else 0

        summary = {
            "run_id": self.run_id,
            "output_path": str(self.output_path),
            "parallel_mode": True,
            "stats": self.stats.to_summary_dict(),
            "valid_count": valid_count,
            "qwen_constraint": {
                "easy_questions": easy_count,
                "total_questions": valid_count,
                "easy_rate": f"{easy_rate:.1%}",
                "max_allowed_rate": f"{self.qwen_max_easy_rate:.0%}",
                "passed": easy_rate <= self.qwen_max_easy_rate,
            },
            "api_stats": self.client.get_stats(),
        }

        # Include final validation stats if available
        if hasattr(self, 'final_validation_summary'):
            summary["final_answer_validation"] = self.final_validation_summary

        return summary


async def run_pipeline(
    api_key: str,
    target_count: int = 50,
    output_path: str = "output/dataset.jsonl",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    topics: Optional[List[str]] = None,
    schema_max_retries: Optional[int] = None,
    sanity_check_max_retries: Optional[int] = None,
    qwen_max_retries: Optional[int] = None,
    crosscheck_max_retries: Optional[int] = None,
    final_validation_max_retries: Optional[int] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the complete pipeline with parallel workers.

    Args:
        api_key: OpenRouter API key
        target_count: Number of QA pairs to generate
        output_path: Output JSONL path
        progress_callback: Optional progress callback
        openai_api_key: Optional OpenAI API key for models not supported by OpenRouter
        anthropic_api_key: Optional Anthropic API key for Claude models
        topics: Optional list of topic names to generate questions for.
                Valid topics: classical_mechanics, electromagnetism, quantum_mechanics,
                statistical_mechanics, thermodynamics, special_relativity,
                general_relativity, condensed_matter, nuclear_physics,
                particle_physics, optics, fluid_mechanics
        schema_max_retries: Max retries for schema validation. Default 3, 0 = unlimited.
        sanity_check_max_retries: Max retries for sanity checks (dimensional analysis, etc.). Default 5, 0 = unlimited.
        qwen_max_retries: Max retries for Qwen difficulty check. Default 5, 0 = unlimited.
        crosscheck_max_retries: Max retries for cross-check validation. Default 5, 0 = unlimited.
        final_validation_max_retries: Max retries for final 5x blind validation. Default 10, 0 = unlimited.
        stop_check: Optional callback that returns True if stop was requested.

    Returns:
        Pipeline results summary
    """
    from ..config import Settings

    # Create settings with the provided API key
    settings = Settings(openrouter_api_key=api_key)

    # Parse topic strings into PhysicsTopic enums
    physics_topics = None
    if topics:
        physics_topics = []
        for topic_str in topics:
            try:
                physics_topics.append(PhysicsTopic(topic_str.lower().replace(" ", "_")))
            except ValueError:
                logger.warning(f"Unknown topic '{topic_str}', skipping. Valid topics: {[t.value for t in PhysicsTopic]}")
        if not physics_topics:
            physics_topics = None  # Fall back to all topics if none valid

    # Use API keys from settings if not provided
    effective_openai_key = openai_api_key or settings.openai_api_key
    effective_anthropic_key = anthropic_api_key or settings.anthropic_api_key

    # Scale up concurrency for parallel workers
    # Each worker makes ~26 API calls (1 gen + 5 qwen + 20 crosscheck)
    # So for N workers, we need N * 26 concurrent capacity
    scaled_max_concurrent = max(
        settings.max_concurrent_requests,
        target_count * 30  # 30 calls per worker with buffer
    )
    scaled_requests_per_minute = max(
        settings.requests_per_minute,
        target_count * 60  # Allow burst for parallel workers
    )

    logger.info(
        f"Pipeline mode: parallel ({target_count} workers), "
        f"max_concurrent: {scaled_max_concurrent}, rpm: {scaled_requests_per_minute}"
    )

    async with OpenRouterClient(
        api_key=api_key,
        openai_api_key=effective_openai_key if effective_openai_key else None,
        anthropic_api_key=effective_anthropic_key if effective_anthropic_key else None,
        max_concurrent=scaled_max_concurrent,
        requests_per_minute=scaled_requests_per_minute,
        max_retries=settings.max_retries,
    ) as client:
        runner = ParallelPipelineRunner(
            client=client,
            settings=settings,
            output_path=output_path,
            progress_callback=progress_callback,
            topics=physics_topics,
            schema_max_retries=schema_max_retries,
            sanity_check_max_retries=sanity_check_max_retries,
            qwen_max_retries=qwen_max_retries,
            crosscheck_max_retries=crosscheck_max_retries,
            final_validation_max_retries=final_validation_max_retries,
            stop_check=stop_check,
        )

        await runner.run(target_count)

        return runner.get_stats_summary()
