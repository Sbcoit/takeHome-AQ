"""Pipeline orchestration for QA generation and validation."""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from ..api.client import OpenRouterClient
from ..config import Settings
from ..models.schemas import (
    PhysicsQADataPoint,
    PhysicsTopic,
    ValidationResult,
    GenerationStats,
)
from ..generation.topics import TopicSampler, TopicContext
from ..generation.generator import QAGenerator
from ..validation.schema import SchemaValidator
from ..validation.qwen_check import QwenConstraintValidator
from ..validation.crosscheck import CrossCheckValidator
from ..validation.correctness import CorrectnessJudge

logger = logging.getLogger(__name__)


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
        self.qa_generator = QAGenerator(client, settings.generation_model)
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
        self.correctness_judge = CorrectnessJudge(
            client,
            judge_model=settings.judge_model,
            correctness_threshold=settings.correctness_threshold,
        )
        self.checkpoint_manager = CheckpointManager()

        # Statistics
        self.stats = GenerationStats()
        self.valid_qa_pairs: List[PhysicsQADataPoint] = []
        self.failed_ids: List[str] = []

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
        topic_context: TopicContext,
    ) -> tuple[Optional[PhysicsQADataPoint], Optional[str], Optional[str]]:
        """
        Validate a QA pair through all checks.

        Returns:
            Tuple of (validated_qa, failure_stage, feedback_for_regeneration)
            - If successful: (qa, None, None)
            - If failed: (None, stage_name, feedback_message)
        """
        # Schema validation
        schema_valid, schema_errors, _ = self.schema_validator.validate_complete(
            qa.model_dump()
        )
        if not schema_valid:
            logger.warning(f"Schema validation failed: {schema_errors}")
            return None, "schema", f"Fix schema errors: {schema_errors}"

        self.stats.passed_schema += 1
        self._report_progress("schema_passed", {"id": qa.id})

        # Qwen constraint check
        logger.info("Running Qwen constraint check...")
        qwen_pass_rate, qwen_high_count, qwen_valid, qwen_details = await self.qwen_validator.validate(
            qa, samples=self.settings.samples_per_qwen_check
        )

        if not qwen_valid:
            logger.warning(f"Qwen constraint failed: {qwen_high_count}/5 attempts passed (too easy)")
            feedback = (
                f"The question is TOO EASY. Qwen3-max solved it correctly {qwen_high_count} out of 5 times. "
                "Make the question significantly harder by: "
                "1) Adding more complex physics that requires deeper understanding, "
                "2) Introducing subtle traps where naive approaches fail, "
                "3) Requiring multi-step reasoning that can't be pattern-matched, "
                "4) Using non-standard scenarios not found in textbooks. "
                "The question should be challenging enough that even strong AI models struggle."
            )
            return None, "qwen", feedback

        self.stats.passed_qwen += 1
        self._report_progress("qwen_passed", {
            "id": qa.id,
            "pass_rate": qwen_pass_rate,
        })

        # Cross-check validation
        logger.info("Running cross-check validation...")
        crosscheck_results, models_correct, total_correct, crosscheck_valid, crosscheck_summary = (
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
            return None, "crosscheck", feedback

        self.stats.passed_crosscheck += 1
        self._report_progress("crosscheck_passed", {
            "id": qa.id,
            "models_correct": models_correct,
            "total_correct": total_correct,
        })

        # Correctness check
        logger.info("Running correctness check...")
        correctness_passed, correctness_score, correctness_details = (
            await self.correctness_judge.judge_single(qa)
        )

        if not correctness_passed:
            logger.warning(f"Correctness check failed: score {correctness_score:.0%}")
            feedback = (
                f"The answer/reasoning has correctness issues (score: {correctness_score:.0%}). "
                "Please verify: "
                "1) The physics principles applied are correct, "
                "2) The mathematical derivations are accurate, "
                "3) The final answer has correct units and reasonable magnitude, "
                "4) The reasoning fully supports the answer. "
                "Fix any errors in the solution."
            )
            return None, "correctness", feedback

        self.stats.passed_correctness += 1
        self.stats.passed_all += 1
        qa.validation_passed = True

        self._report_progress("all_passed", {
            "id": qa.id,
            "correctness_score": correctness_score,
        })

        logger.info(f"QA pair {qa.id} passed all validation!")
        return qa, None, None

    async def generate_and_validate_single(self) -> Optional[PhysicsQADataPoint]:
        """Generate and validate a single QA pair with regeneration on failure."""
        # Generate initial question
        topic_context = self.topic_sampler.sample(prefer_diverse=True)
        logger.info(f"Generating QA for: {topic_context.subtopic}")

        try:
            qa = await self.qa_generator.generate(topic_context, temperature=0.8)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.stats.failed += 1
            return None

        self.stats.total_generated += 1
        self._report_progress("generated", {"topic": topic_context.subtopic})

        # Track by topic
        topic_name = topic_context.topic.value
        self.stats.by_topic[topic_name] = self.stats.by_topic.get(topic_name, 0) + 1

        # Validate with regeneration loop (no limit - keep trying until success)
        attempt = 0
        while True:
            attempt += 1
            validated_qa, failure_stage, feedback = await self._validate_qa(qa, topic_context)

            if validated_qa is not None:
                return validated_qa

            # Schema failures can't be fixed by regeneration
            if failure_stage == "schema":
                self.stats.failed += 1
                self.failed_ids.append(qa.id)
                return None

            # Regenerate with feedback
            logger.info(f"Regenerating question (attempt {attempt + 1}) - {failure_stage} failed")
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
                return None

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

        # Clear output file if starting fresh
        if self.output_path.exists():
            self.output_path.unlink()

        checkpoint_interval = 5  # Save checkpoint every N valid pairs
        max_attempts_multiplier = 5  # Give up after target * multiplier attempts

        attempts = 0
        max_attempts = target_count * max_attempts_multiplier

        while len(self.valid_qa_pairs) < target_count and not self._stop_requested:
            attempts += 1

            if attempts > max_attempts:
                logger.warning(
                    f"Reached max attempts ({max_attempts}). "
                    f"Only generated {len(self.valid_qa_pairs)}/{target_count} valid pairs."
                )
                break

            try:
                qa = await self.generate_and_validate_single()

                if qa:
                    self.valid_qa_pairs.append(qa)
                    self._write_to_output(qa)

                    # Checkpoint
                    if len(self.valid_qa_pairs) % checkpoint_interval == 0:
                        self.checkpoint_manager.save(
                            self.run_id,
                            self.valid_qa_pairs,
                            self.stats,
                            self.failed_ids,
                        )

                    logger.info(
                        f"Progress: {len(self.valid_qa_pairs)}/{target_count} valid pairs "
                        f"({self.stats.total_generated} generated, {self.stats.failed} failed)"
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

        # Final dataset-level correctness check
        if self.valid_qa_pairs:
            logger.info("Running final dataset-level correctness verification...")
            accuracy, passed, _ = await self.correctness_judge.validate_dataset(
                self.valid_qa_pairs
            )

            if not passed:
                logger.warning(
                    f"Dataset accuracy {accuracy:.1%} is below threshold "
                    f"{self.settings.correctness_threshold:.0%}"
                )
            else:
                logger.info(f"Dataset accuracy {accuracy:.1%} meets threshold!")

        # Clear checkpoint on completion
        self.checkpoint_manager.clear(self.run_id)

        logger.info(
            f"Pipeline complete! Generated {len(self.valid_qa_pairs)} valid QA pairs. "
            f"Output: {self.output_path}"
        )

        return self.valid_qa_pairs

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline statistics."""
        return {
            "run_id": self.run_id,
            "output_path": str(self.output_path),
            "stats": self.stats.to_summary_dict(),
            "valid_count": len(self.valid_qa_pairs),
            "topic_coverage": self.topic_sampler.get_coverage_stats(),
            "api_stats": self.client.get_stats(),
        }


async def run_pipeline(
    api_key: str,
    target_count: int = 50,
    output_path: str = "output/dataset.jsonl",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    openai_api_key: Optional[str] = None,
    topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the complete pipeline.

    Args:
        api_key: OpenRouter API key
        target_count: Number of QA pairs to generate
        output_path: Output JSONL path
        progress_callback: Optional progress callback
        openai_api_key: Optional OpenAI API key for models not supported by OpenRouter
        topics: Optional list of topic names to generate questions for.
                Valid topics: classical_mechanics, electromagnetism, quantum_mechanics,
                statistical_mechanics, thermodynamics, special_relativity,
                general_relativity, condensed_matter, nuclear_physics,
                particle_physics, optics, fluid_mechanics

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

    # Use OpenAI key from settings if not provided
    effective_openai_key = openai_api_key or settings.openai_api_key

    async with OpenRouterClient(
        api_key=api_key,
        openai_api_key=effective_openai_key if effective_openai_key else None,
        max_concurrent=settings.max_concurrent_requests,
        requests_per_minute=settings.requests_per_minute,
        max_retries=settings.max_retries,
    ) as client:
        runner = PipelineRunner(
            client=client,
            settings=settings,
            output_path=output_path,
            progress_callback=progress_callback,
            topics=physics_topics,
        )

        await runner.run(target_count)

        return runner.get_stats_summary()
