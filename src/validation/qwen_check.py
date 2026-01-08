"""Qwen constraint validation (anti-"too easy" filter)."""

import asyncio
import json
import logging
from typing import Tuple, List, Dict, Any

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint, GradingResult
from ..utils import extract_json_from_response
from ..prompts import JSON_INSTRUCTION, LATEX_FORMAT_GUIDE, ANSWER_FORMAT_GUIDE

logger = logging.getLogger(__name__)


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

    ANSWER_PROMPT = f"""You are a physics graduate student taking a qualifying exam.
Solve the following problem completely, showing all your work.

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

    GRADING_PROMPT = f"""You are a STRICT physics professor grading a student's solution. You must evaluate BOTH the reasoning AND the final answer.

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

PASS CRITERIA (STRICT):
- Must score >= 7/10 points total
- Must have correct final answer (at least 2/3 points)
- Both conditions required to pass

BE STRICT - this is for discriminating between models:
- Do NOT give benefit of the doubt
- Missing steps = missing points
- Wrong reasoning with right answer = max 3 points (answer only)
- Right reasoning with wrong answer = max 7 points
{JSON_INSTRUCTION}

Expected format:
{{{{
    "final_answer_score": {{{{"points": 0-3, "explanation": "why this score"}}}},
    "key_steps_score": {{{{"points": 0-7, "breakdown": "which steps were demonstrated"}}}},
    "total_score": 0-10,
    "automatic_zero_triggered": false,
    "passed": true/false,
    "explanation": "One sentence summary"
}}}}"""

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
        prompt = self.ANSWER_PROMPT.replace("{{query}}", qa.query)

        try:
            response = await self.client.chat_completion(
                model=self.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=8192,
                reasoning=True,  # Enable thinking mode
                response_format={"type": "json_object"},
            )
            # Handle different response structures
            message = response["choices"][0]["message"]
            content = message.get("content", "")

            # Some models return content as a list or other structure
            if isinstance(content, list):
                # Extract text from content array
                content = " ".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            elif not isinstance(content, str):
                content = str(content)

            return content if content else "ERROR: Empty response"
        except Exception as e:
            import traceback
            logger.error(f"Failed to get Qwen answer: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return f"ERROR: {e}"

    async def _grade_answer(
        self,
        qa: PhysicsQADataPoint,
        student_answer: str,
    ) -> GradingResult:
        """Grade a student answer against the rubric."""
        rubric_text = json.dumps(qa.rubric.model_dump(), indent=2)

        prompt = (self.GRADING_PROMPT
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
                    temperature=0.1,  # Low for consistent grading
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

                # Parse the point-based grading format
                passed = data.get("passed", False)
                explanation = data.get("explanation", "")

                # Extract point-based scores
                final_answer_data = data.get("final_answer_score", {})
                final_answer_points = final_answer_data.get("points", 0) if isinstance(final_answer_data, dict) else 0

                key_steps_data = data.get("key_steps_score", {})
                key_steps_points = key_steps_data.get("points", 0) if isinstance(key_steps_data, dict) else 0

                total_score = data.get("total_score", final_answer_points + key_steps_points)
                automatic_zero = data.get("automatic_zero_triggered", False)

                # If automatic zero was triggered, override the score
                if automatic_zero:
                    total_score = 0
                    passed = False

                # Determine pass: needs >= 7/10 AND >= 2/3 on final answer
                # (The judge should already compute this, but we double-check)
                computed_passed = (total_score >= 7) and (final_answer_points >= 2) and not automatic_zero

                # Use judge's passed value but log discrepancy
                if passed != computed_passed:
                    logger.warning(
                        f"Pass criteria mismatch: judge={passed}, computed={computed_passed} "
                        f"(score={total_score}/10, answer={final_answer_points}/3)"
                    )

                logger.debug(
                    f"Grading: answer={final_answer_points}/3, steps={key_steps_points}/7, "
                    f"total={total_score}/10, auto_zero={automatic_zero}. Passed: {passed}"
                )

                return GradingResult(
                    criteria_scores={
                        "final_answer": final_answer_points,
                        "key_steps": key_steps_points,
                    },
                    total_points=total_score,
                    max_points=10,
                    criteria_passed=2 if passed else (1 if final_answer_points >= 2 or key_steps_points >= 5 else 0),
                    total_criteria=2,
                    passed=passed,
                    explanation=explanation,
                )

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error in grading (attempt {attempt + 1}/{max_judge_retries}): {e}")
                last_error = str(e)
                await asyncio.sleep(1)
                continue
            except Exception as e:
                import traceback
                logger.error(f"Failed to grade answer: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                last_error = str(e)
                break

        logger.error(f"Grading failed after {max_judge_retries} attempts: {last_error}")
        return GradingResult(
            criteria_scores={"final_answer": 0, "key_steps": 0},
            total_points=0,
            max_points=10,
            criteria_passed=0,
            total_criteria=2,
            passed=False,
            explanation=f"Grading failed: {last_error}",
        )

    async def _check_single_attempt(
        self,
        qa: PhysicsQADataPoint,
        attempt_idx: int,
    ) -> Tuple[str, GradingResult]:
        """Have Qwen attempt the question and immediately grade it."""
        try:
            # Get Qwen's answer
            answer = await self._get_qwen_answer(qa)

            # If answer failed, return error result
            if answer.startswith("ERROR:"):
                grading = GradingResult(
                    criteria_scores={"final_answer": 0, "key_steps": 0},
                    total_points=0,
                    max_points=10,
                    criteria_passed=0,
                    total_criteria=2,
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
        except Exception as e:
            import traceback
            logger.error(f"Qwen attempt {attempt_idx+1} exception: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise

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
