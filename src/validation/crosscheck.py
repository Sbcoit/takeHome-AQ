"""Frontier model cross-check validation."""

import asyncio
import json
import logging
import re
from typing import Tuple, Dict, List, Any

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint, ModelTestResult

logger = logging.getLogger(__name__)


def extract_json_from_response(content: str) -> dict:
    """
    Extract JSON from a response that may contain markdown or other text.

    Handles cases like:
    - Pure JSON: {"key": "value"}
    - Markdown wrapped: ```json\n{"key": "value"}\n```
    - Text before/after JSON: "Here is the result: {"key": "value"}"
    """
    content = content.strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(json_block_pattern, content)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to find JSON object in the text (find first { and last })
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(content[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Nothing worked, raise error with content preview
    raise json.JSONDecodeError(
        f"Could not extract JSON from response: {content[:200]}...",
        content, 0
    )


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

    SOLVE_PROMPT = """Solve this physics problem completely. Show your work and provide your final answer.

PROBLEM:
{query}

Respond with JSON in this exact format:
{{
    "solution": "Your complete step-by-step solution with all mathematical derivations",
    "final_answer": "Your final numerical or symbolic answer with units"
}}"""

    JUDGE_PROMPT = """You are an expert physics professor evaluating whether a student demonstrated understanding of a physics problem.

QUESTION:
{query}

REFERENCE ANSWER:
{reference_answer}

REFERENCE SOLUTION:
{reference_reasoning}

STUDENT'S ANSWER:
{student_answer}

Your task is to determine if the student's answer demonstrates correct understanding and arrives at the right answer, even if their approach differs from the reference solution.

EVALUATION CRITERIA:
1. **Correct Physics**: Did the student identify and apply the correct physical principles?
2. **Correct Answer**: Did the student arrive at the correct (or equivalent) final answer?
3. **Sound Reasoning**: Is the mathematical/logical approach valid, even if different from reference?

BE GENEROUS in your evaluation:
- Accept equivalent answers (e.g., different but equivalent forms, reasonable rounding)
- Accept valid alternative approaches that reach the same answer
- Focus on whether the physics is RIGHT, not whether it matches the reference exactly
- Minor calculation errors are acceptable if the method is correct and the answer is close

Mark as CORRECT if:
- The student gets the right answer (or equivalent) with valid reasoning, OR
- The student uses correct physics and makes only minor errors

Mark as INCORRECT only if:
- The physics approach is fundamentally wrong, OR
- The final answer is significantly wrong (wrong order of magnitude, wrong units, wrong sign for a meaningful quantity)

IMPORTANT: You MUST respond with ONLY a JSON object. Do not include any text before or after the JSON. Do not use markdown formatting. Do not analyze parts separately.

{{
    "correct_physics": {{"passed": true/false, "explanation": "one sentence"}},
    "correct_answer": {{"passed": true/false, "explanation": "one sentence"}},
    "sound_reasoning": {{"passed": true/false, "explanation": "one sentence"}},
    "is_correct": true/false,
    "explanation": "One sentence summary"
}}"""

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
        prompt = self.SOLVE_PROMPT.format(query=qa.query)

        try:
            response = await self.client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=65536,
                response_format={"type": "json_object"},
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Failed to get answer from {model}: {e}")
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

        prompt = self.JUDGE_PROMPT.format(
            query=qa.query,
            reference_answer=qa.response_answer,
            reference_reasoning=qa.response_reasoning,
            student_answer=student_answer,
            rubric=rubric_text,
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

                content = response["choices"][0]["message"]["content"]

                # Handle empty responses - retry
                if not content or not content.strip():
                    logger.warning(f"Empty response from judge (attempt {attempt + 1}/{max_judge_retries})")
                    last_error = "Empty response from judge model"
                    await asyncio.sleep(1)
                    continue

                data = extract_json_from_response(content)

                # Parse the simplified grading response
                is_correct = data.get("is_correct", False)
                explanation = data.get("explanation", "")

                # Log detailed breakdown for debugging
                correct_physics = data.get("correct_physics", {}).get("passed", False)
                correct_answer = data.get("correct_answer", {}).get("passed", False)
                sound_reasoning = data.get("sound_reasoning", {}).get("passed", False)

                logger.debug(
                    f"Grading: physics={correct_physics}, answer={correct_answer}, "
                    f"reasoning={sound_reasoning}. Overall: {is_correct}"
                )

                return is_correct, explanation

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}/{max_judge_retries}): {e}")
                last_error = str(e)
                await asyncio.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Failed to judge correctness: {e}")
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
        answer = await self._get_model_answer(model, qa)
        is_correct, explanation = await self._judge_correctness(qa, answer)
        logger.debug(f"{model} attempt {sample_idx+1}: {'CORRECT' if is_correct else 'INCORRECT'}")
        return answer, is_correct, explanation

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
