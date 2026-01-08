"""Frontier model cross-check validation."""

import asyncio
import json
import logging
from typing import Tuple, Dict, List, Any

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint, ModelTestResult
from ..utils import extract_json_from_response
from ..prompts import JSON_INSTRUCTION, LATEX_FORMAT_GUIDE, ANSWER_FORMAT_GUIDE

logger = logging.getLogger(__name__)


class CrossCheckValidator:
    """
    Validates questions with multiple frontier models.

    Rule: Test with 4 models × 5 samples = 20 attempts.
    - At least 2 models must have accuracy > 0
    - Total correct across 20 attempts must be >= 5 (25% accuracy)
    """

    # Default models for cross-checking
    DEFAULT_MODELS = [
        "deepseek/deepseek-r1",
        "openai/o4-mini",
        "google/gemini-2.5-pro",
        "x-ai/grok-4",
    ]

    SOLVE_PROMPT = f"""Solve this physics problem completely. Show your work and provide your final answer.

PROBLEM:
{{query}}
{LATEX_FORMAT_GUIDE}
{ANSWER_FORMAT_GUIDE}

Use markdown formatting for clarity. Mark your final answer with **bold**.
{JSON_INSTRUCTION}

Expected format:
{{{{
    "solution": "## Solution\\n\\n### Approach\\nMethod description.\\n\\n### Derivation\\n1. First step: **equation**\\n2. Second step...\\n\\n### Final Answer\\n**symbolic_answer**",
    "final_answer": "symbolic answer using LaTeX (e.g., \\\\frac{{m \\\\omega^2 R^2}}{{2}}, \\\\frac{{4}}{{3}}, \\\\frac{{\\\\hbar \\\\omega}}{{2}})"
}}}}"""

    JUDGE_PROMPT = f"""You are an expert physics professor grading a student's solution. You must evaluate BOTH the reasoning AND the final answer.

QUESTION:
{{query}}

REFERENCE ANSWER:
{{reference_answer}}

REFERENCE SOLUTION:
{{reference_reasoning}}

GRADING RUBRIC (10-point scale):
{{rubric}}

STUDENT'S ANSWER:
{{student_answer}}

GRADING INSTRUCTIONS:
1. First, check for AUTOMATIC ZERO conditions (from rubric)
2. Grade the FINAL ANSWER (3 points max):
   - Exactly correct or equivalent form: 3 points
   - Close but missing a factor (2, pi, etc.): 1 point
   - Wrong: 0 points
3. Grade each KEY STEP (7 points total):
   - Award points only if the step is clearly demonstrated
   - Partial credit allowed within each step
4. Calculate TOTAL SCORE out of 10

PASS CRITERIA (GENEROUS for frontier models):
- Must score >= 6/10 points total
- Must have correct final answer (at least 2/3 points)
- Both conditions required to pass

BE GENEROUS in your evaluation:
- Accept equivalent answers (e.g., different but equivalent forms, reasonable rounding)
- Accept valid alternative approaches that reach the same answer
- Minor calculation errors are acceptable if the method is correct and answer is close
- Focus on whether the physics is RIGHT, not whether it matches the reference exactly
{JSON_INSTRUCTION}

Expected format:
{{{{
    "final_answer_score": {{{{"points": 0-3, "explanation": "why this score"}}}},
    "key_steps_score": {{{{"points": 0-7, "breakdown": "which steps were demonstrated"}}}},
    "total_score": 0-10,
    "automatic_zero_triggered": false,
    "is_correct": true/false,
    "explanation": "One sentence summary"
}}}}"""

    def __init__(
        self,
        client: OpenRouterClient,
        judge_model: str = "anthropic/claude-sonnet-4",
        models: List[str] | None = None,
        min_models_with_correct: int = 2,
        min_total_correct: int = 5,
    ):
        """
        Initialize the cross-check validator.

        Args:
            client: OpenRouter API client
            judge_model: Model to use for judging correctness
            models: List of models to test with (default: DEFAULT_MODELS)
            min_models_with_correct: Minimum models that must get >= 1 correct
            min_total_correct: Minimum total correct across all attempts
        """
        self.client = client
        self.judge_model = judge_model
        self.models = models or self.DEFAULT_MODELS
        self.min_models_with_correct = min_models_with_correct
        self.min_total_correct = min_total_correct

    async def _get_model_answer(self, model: str, qa: PhysicsQADataPoint) -> str:
        """Get an answer from a specific model."""
        prompt = self.SOLVE_PROMPT.replace("{{query}}", qa.query)

        try:
            response = await self.client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=8192,
                reasoning=True,  # Enable thinking/reasoning mode for better problem solving
                response_format={"type": "json_object"},
            )
            # Handle different response structures
            message = response["choices"][0]["message"]
            content = message.get("content", "")

            # Some models return content as a list or other structure
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            elif not isinstance(content, str):
                content = str(content)

            return content if content else "ERROR: Empty response"
        except Exception as e:
            import traceback
            logger.warning(f"Failed to get answer from {model}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return f"ERROR: {e}"

    async def _judge_correctness(
        self,
        qa: PhysicsQADataPoint,
        student_answer: str,
    ) -> Tuple[bool, str]:
        """Judge if a student answer is correct using rubric-based grading."""
        if student_answer.startswith("ERROR:"):
            return False, "Model failed to generate answer"

        # Include rubric in the prompt
        rubric_text = json.dumps(qa.rubric.model_dump(), indent=2)

        prompt = (self.JUDGE_PROMPT
            .replace("{{query}}", qa.query)
            .replace("{{reference_answer}}", qa.response_answer)
            .replace("{{reference_reasoning}}", qa.response_reasoning)
            .replace("{{student_answer}}", student_answer)
            .replace("{{rubric}}", rubric_text)
        )

        max_judge_retries = 3
        last_error = None

        for attempt in range(max_judge_retries):
            try:
                response = await self.client.chat_completion(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )

                # Debug: log the full response structure
                logger.debug(f"Judge response structure: {type(response)}, keys: {response.keys() if isinstance(response, dict) else 'N/A'}")

                choices = response.get("choices", [])
                if not choices:
                    logger.warning(f"No choices in judge response: {response}")
                    last_error = "No choices in response"
                    await asyncio.sleep(1)
                    continue

                message = choices[0].get("message", {})
                content = message.get("content", "")

                # Handle content that might be a list (some models return this)
                if isinstance(content, list):
                    content = " ".join(
                        item.get("text", str(item)) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                elif not isinstance(content, str):
                    logger.warning(f"Unexpected content type: {type(content)}, value: {content}")
                    content = str(content) if content else ""

                # Handle empty responses - retry
                if not content or not content.strip():
                    logger.warning(f"Empty response from judge (attempt {attempt + 1}/{max_judge_retries})")
                    last_error = "Empty response from judge model"
                    await asyncio.sleep(1)
                    continue

                logger.debug(f"Attempting to parse JSON from content (first 500 chars): {content[:500]}")
                data = extract_json_from_response(content)
                logger.debug(f"Parsed data type: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'N/A'}")

                # Parse the point-based grading response
                is_correct = data.get("is_correct", False)
                explanation = data.get("explanation", "")

                # Extract point-based scores for logging
                final_answer_data = data.get("final_answer_score", {})
                final_answer_points = final_answer_data.get("points", 0) if isinstance(final_answer_data, dict) else 0

                key_steps_data = data.get("key_steps_score", {})
                key_steps_points = key_steps_data.get("points", 0) if isinstance(key_steps_data, dict) else 0

                total_score = data.get("total_score", final_answer_points + key_steps_points)
                automatic_zero = data.get("automatic_zero_triggered", False)

                # If automatic zero was triggered, override
                if automatic_zero:
                    is_correct = False

                # Verify pass criteria (generous: >= 6/10 AND >= 2/3 on answer)
                computed_correct = (total_score >= 6) and (final_answer_points >= 2) and not automatic_zero

                if is_correct != computed_correct:
                    logger.debug(
                        f"Correctness mismatch: judge={is_correct}, computed={computed_correct} "
                        f"(score={total_score}/10, answer={final_answer_points}/3)"
                    )

                logger.debug(
                    f"Grading: answer={final_answer_points}/3, steps={key_steps_points}/7, "
                    f"total={total_score}/10, auto_zero={automatic_zero}. Correct: {is_correct}"
                )

                return is_correct, explanation

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}/{max_judge_retries}): {e}")
                last_error = str(e)
                await asyncio.sleep(1)
                continue
            except Exception as e:
                import traceback
                logger.error(f"Failed to judge correctness: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                last_error = str(e)
                break

        logger.error(f"Judge failed after {max_judge_retries} attempts: {last_error}")
        return False, f"Judging failed: {last_error}"

    async def _test_single_sample(
        self,
        model: str,
        qa: PhysicsQADataPoint,
        sample_idx: int,
    ) -> Tuple[str, bool, str]:
        """Test a single sample: get answer then immediately judge it."""
        try:
            answer = await self._get_model_answer(model, qa)
            is_correct, explanation = await self._judge_correctness(qa, answer)
            logger.debug(f"{model} attempt {sample_idx+1}: {'CORRECT' if is_correct else 'INCORRECT'}")
            return answer, is_correct, explanation
        except Exception as e:
            import traceback
            logger.error(f"{model} attempt {sample_idx+1} exception: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise

    async def _test_single_model(
        self,
        model: str,
        qa: PhysicsQADataPoint,
        samples: int,
    ) -> ModelTestResult:
        """Test a single model with multiple samples - each sample gets answer then judge."""
        logger.debug(f"Testing {model} with {samples} samples...")

        # Run all samples in parallel - each sample does: get answer -> judge
        tasks = [
            self._test_single_sample(model, qa, i)
            for i in range(samples)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        correct_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"{model} attempt {i+1} failed: {result}")
                responses.append(f"ERROR: {result}")
            else:
                answer, is_correct, _ = result
                responses.append(answer)
                if is_correct:
                    correct_count += 1

        return ModelTestResult(
            model=model,
            samples=samples,
            correct_count=correct_count,
            responses=responses,
        )

    async def validate(
        self,
        qa: PhysicsQADataPoint,
        samples_per_model: int = 5,
    ) -> Tuple[Dict[str, ModelTestResult], int, int, bool, Dict[str, Any]]:
        """
        Validate a question with multiple frontier models.

        Args:
            qa: The QA data point to validate
            samples_per_model: Number of samples per model (default 5)

        Returns:
            Tuple of:
            - results: Dict of model -> ModelTestResult
            - models_with_correct: Number of models with >= 1 correct
            - total_correct: Total correct across all attempts
            - is_valid: True if question passes the cross-check
            - summary: Summary statistics
        """
        logger.info(
            f"Running cross-check with {len(self.models)} models × {samples_per_model} samples..."
        )

        # Test all models concurrently for faster execution
        tasks = [
            self._test_single_model(model, qa, samples_per_model)
            for model in self.models
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results: Dict[str, ModelTestResult] = {}
        models_with_correct = 0
        total_correct = 0

        for model, result in zip(self.models, results_list):
            if isinstance(result, Exception):
                logger.error(f"Cross-check failed for {model}: {result}")
                results[model] = ModelTestResult(
                    model=model,
                    samples=samples_per_model,
                    correct_count=0,
                    error=str(result),
                )
                logger.info(f"  {model}: 0/{samples_per_model} correct (ERROR)")
            else:
                results[model] = result
                if result.correct_count > 0:
                    models_with_correct += 1
                total_correct += result.correct_count
                logger.info(f"  {model}: {result.correct_count}/{samples_per_model} correct ({result.accuracy:.0%})")

        # Check validation criteria
        is_valid = (
            models_with_correct >= self.min_models_with_correct
            and total_correct >= self.min_total_correct
        )

        total_attempts = len(self.models) * samples_per_model
        overall_accuracy = total_correct / total_attempts if total_attempts > 0 else 0

        summary = {
            "models_tested": len(self.models),
            "samples_per_model": samples_per_model,
            "total_attempts": total_attempts,
            "models_with_correct": models_with_correct,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "by_model": {
                model: {
                    "correct": r.correct_count,
                    "samples": r.samples,
                    "accuracy": r.accuracy,
                }
                for model, r in results.items()
            },
        }

        logger.info(
            f"Cross-check result: {models_with_correct} models with correct, "
            f"{total_correct}/{total_attempts} total correct ({overall_accuracy:.0%}). "
            f"Valid: {is_valid}"
        )

        return results, models_with_correct, total_correct, is_valid, summary
