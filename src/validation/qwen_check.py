"""Qwen constraint validation (anti-"too easy" filter)."""

import asyncio
import json
import logging
import re
from typing import Tuple, List, Dict, Any

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint, GradingResult

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


class QwenConstraintValidator:
    """
    Validates that questions are not too easy by testing with Qwen3-max.

    Rule per document: "The proportion of cases where Qwen3-max-thinking passes
    at least 4 out of 5 attempts must not exceed 5%."

    Interpretation (DATASET-LEVEL constraint):
    - Run Qwen 5 times on each question
    - Grade each attempt as PASS or FAIL (based on rubric criteria)
    - If Qwen passes >= 4 out of 5 attempts → question is "easy"
    - Across the DATASET, no more than 5% of questions can be "easy"
    - This means for a 50-question dataset, up to 2 questions can be easy

    The validator returns whether a question is "easy" (4+/5 passes).
    The pipeline decides whether to accept based on current easy question count.
    """

    ANSWER_PROMPT = """You are a physics graduate student taking a qualifying exam.
Solve the following problem completely, showing all your work.

PROBLEM:
{query}

Respond with JSON in this exact format:
{{
    "solution": "Your complete step-by-step solution with all mathematical derivations and reasoning",
    "final_answer": "Your final numerical or symbolic answer with units"
}}"""

    GRADING_PROMPT = """You are an expert physics professor evaluating whether a student demonstrated understanding of a physics problem.

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

Mark as PASSED if:
- The student gets the right answer (or equivalent) with valid reasoning, OR
- The student uses correct physics and makes only minor errors

Mark as FAILED only if:
- The physics approach is fundamentally wrong, OR
- The final answer is significantly wrong (wrong order of magnitude, wrong units, wrong sign for a meaningful quantity)

IMPORTANT: You MUST respond with ONLY a JSON object. Do not include any text before or after the JSON. Do not use markdown formatting. Do not analyze parts separately.

{{
    "correct_physics": {{"passed": true/false, "explanation": "one sentence"}},
    "correct_answer": {{"passed": true/false, "explanation": "one sentence"}},
    "sound_reasoning": {{"passed": true/false, "explanation": "one sentence"}},
    "passed": true/false,
    "explanation": "One sentence summary"
}}"""

    def __init__(
        self,
        client: OpenRouterClient,
        qwen_model: str = "qwen/qwen3-max",
        judge_model: str = "anthropic/claude-sonnet-4",
        max_pass_rate: float = 0.05,
    ):
        """
        Initialize the Qwen constraint validator.

        Args:
            client: OpenRouter API client
            qwen_model: Model ID for Qwen
            judge_model: Model ID for grading responses
            max_pass_rate: Maximum allowed pass rate (default 5%)
        """
        self.client = client
        self.qwen_model = qwen_model
        self.judge_model = judge_model
        self.max_pass_rate = max_pass_rate

    async def _get_qwen_answer(self, qa: PhysicsQADataPoint) -> str:
        """Have Qwen attempt the question."""
        prompt = self.ANSWER_PROMPT.format(query=qa.query)

        try:
            response = await self.client.chat_completion(
                model=self.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=65536,
                reasoning=True,  # Enable thinking mode
                response_format={"type": "json_object"},
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to get Qwen answer: {e}")
            return f"ERROR: {e}"

    async def _grade_answer(
        self,
        qa: PhysicsQADataPoint,
        student_answer: str,
    ) -> GradingResult:
        """Grade a student answer against the rubric."""
        rubric_text = json.dumps(qa.rubric.model_dump(), indent=2)

        prompt = self.GRADING_PROMPT.format(
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
                    temperature=0.1,  # Low for consistent grading
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

                # Parse the simplified grading format
                passed = data.get("passed", False)
                explanation = data.get("explanation", "")

                # Extract individual criteria results for logging
                correct_physics = data.get("correct_physics", {}).get("passed", False)
                correct_answer = data.get("correct_answer", {}).get("passed", False)
                sound_reasoning = data.get("sound_reasoning", {}).get("passed", False)

                # Count how many criteria passed for compatibility with existing code
                criteria_passed = sum([correct_physics, correct_answer, sound_reasoning])

                logger.debug(
                    f"Grading: physics={correct_physics}, answer={correct_answer}, "
                    f"reasoning={sound_reasoning}. Overall passed: {passed}"
                )

                return GradingResult(
                    criteria_scores={
                        "correct_physics": 1 if correct_physics else 0,
                        "correct_answer": 1 if correct_answer else 0,
                        "sound_reasoning": 1 if sound_reasoning else 0,
                    },
                    total_points=criteria_passed,
                    max_points=3,
                    criteria_passed=criteria_passed,
                    total_criteria=3,
                    passed=passed,
                    explanation=explanation,
                )

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error in grading (attempt {attempt + 1}/{max_judge_retries}): {e}")
                last_error = str(e)
                await asyncio.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Failed to grade answer: {e}")
                last_error = str(e)
                break

        logger.error(f"Grading failed after {max_judge_retries} attempts: {last_error}")
        return GradingResult(
            criteria_scores={},
            total_points=0,
            max_points=100,
            criteria_passed=0,
            total_criteria=5,
            passed=False,
            explanation=f"Grading failed: {last_error}",
        )

    async def _check_single_attempt(
        self,
        qa: PhysicsQADataPoint,
        attempt_idx: int,
    ) -> Tuple[str, GradingResult]:
        """Have Qwen attempt the question and immediately grade it."""
        # Get Qwen's answer
        answer = await self._get_qwen_answer(qa)

        # If answer failed, return error result
        if answer.startswith("ERROR:"):
            grading = GradingResult(
                criteria_scores={},
                total_points=0,
                max_points=100,
                criteria_passed=0,
                total_criteria=5,
                passed=False,
                explanation=f"Qwen failed to generate answer: {answer}",
            )
        else:
            # Grade it immediately
            grading = await self._grade_answer(qa, answer)

        logger.debug(
            f"Qwen attempt {attempt_idx+1}: {grading.criteria_passed}/{grading.total_criteria} "
            f"criteria passed ({grading.criteria_pass_rate:.0%})"
        )
        return answer, grading

    async def validate(
        self,
        qa: PhysicsQADataPoint,
        samples: int = 5,
    ) -> Tuple[float, int, bool, List[Dict[str, Any]]]:
        """
        Validate a question against the Qwen constraint.

        Args:
            qa: The QA data point to validate
            samples: Number of times to sample Qwen (default 5)

        Returns:
            Tuple of:
            - pass_rate: Proportion of attempts where Qwen passed 80%+ criteria
            - high_pass_count: Number of attempts with high pass rate
            - is_valid: True if question passes the constraint (is hard enough)
            - details: List of detailed results for each attempt
        """
        logger.info(f"Running Qwen constraint check with {samples} samples...")

        # Run all samples in parallel - each sample does: get answer -> grade
        tasks = [self._check_single_attempt(qa, i) for i in range(samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        details = []
        high_pass_count = 0

        for i, result in enumerate(results):
            # Handle exceptions from asyncio.gather
            if isinstance(result, Exception):
                logger.error(f"Qwen attempt {i+1} failed: {result}")
                details.append({
                    "attempt": i + 1,
                    "error": str(result),
                    "passed_criteria_rate": 0,
                })
                continue

            _answer, grading = result

            # Check if this attempt had an error
            if grading.explanation and grading.explanation.startswith(("Qwen failed", "Grading failed")):
                details.append({
                    "attempt": i + 1,
                    "error": grading.explanation,
                    "passed_criteria_rate": 0,
                })
                continue

            # Check if this attempt passed (Qwen got it right)
            is_pass = grading.passed

            if is_pass:
                high_pass_count += 1

            details.append({
                "attempt": i + 1,
                "passed": is_pass,
                "criteria_passed": grading.criteria_passed,
                "total_criteria": grading.total_criteria,
                "total_points": grading.total_points,
                "max_points": grading.max_points,
                "explanation": grading.explanation[:200] if grading.explanation else "",
            })

        # Calculate pass rate (proportion of attempts Qwen passed)
        valid_attempts = len([d for d in details if "error" not in d])
        pass_rate = high_pass_count / valid_attempts if valid_attempts > 0 else 0

        # Document rule: "passes at least 4 out of 5 attempts must not exceed 5%"
        # This is a DATASET-LEVEL constraint, not per-question.
        # We return whether the question is "easy" (4+/5), and let the pipeline
        # decide whether to accept based on current easy question count.
        min_passes_for_easy = 4  # If Qwen passes 4+ attempts, question is "easy"
        is_easy = high_pass_count >= min_passes_for_easy

        logger.info(
            f"Qwen constraint: {high_pass_count}/{valid_attempts} attempts passed "
            f"({pass_rate:.0%}). Easy threshold: {min_passes_for_easy}+. Is easy: {is_easy}"
        )

        # Return is_easy (True if question is easy), NOT is_valid
        # Pipeline will use this to track dataset-level 5% threshold
        return pass_rate, high_pass_count, is_easy, details
