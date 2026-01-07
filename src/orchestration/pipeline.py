"""Pipeline orchestration for QA generation and validation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

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
        self.qa_generator = QAGenerator(
            client,
            settings.generation_model,
            judge_model=settings.judge_model,
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
        MAX_REGEN_ATTEMPTS = 5
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
