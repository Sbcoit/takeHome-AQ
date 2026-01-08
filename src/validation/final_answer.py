"""Final answer validation using Claude Opus as self-judge.

This module implements a post-generation validation step where Claude Opus
judges its own generated answers 5 consecutive times. A QA pair only passes
if all 5 judgments confirm the answer is correct.

If validation fails, the system regenerates the QA pair with feedback and
tries again, up to a maximum number of attempts.
"""

import asyncio
import json
import logging
from typing import Tuple, List, Dict, Any, Optional

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint
from ..utils import extract_json_from_response
from ..prompts import JSON_INSTRUCTION

logger = logging.getLogger(__name__)


class FinalAnswerValidator:
    """
    Validates QA pairs using a TWO-STEP BLIND judgment approach.

    Step 1 (Blind Solve): Claude sees ONLY the question and solves it independently
    Step 2 (Compare): Claude compares its derived answer to the provided answer

    The QA pair only passes if all 5 consecutive judgments confirm correctness.
    This ensures high confidence in the final answer's validity without bias.
    """

    # Step 1: Blind solve - Claude only sees the question
    BLIND_SOLVE_PROMPT = f"""You are an expert physics professor solving a graduate-level physics problem.

QUESTION:
{{query}}

Solve this problem completely and rigorously. Show your work step by step.

REQUIREMENTS:
1. Derive the answer from first principles
2. Show all mathematical steps clearly
3. Check dimensional analysis - verify units are consistent
4. Check limiting cases where applicable (e.g., what happens as parameters -> 0 or -> infinity)
5. State your final answer clearly

{JSON_INSTRUCTION}

Respond with JSON:
{{{{
    "approach": "Brief description of your solution approach",
    "derivation": "Your step-by-step derivation",
    "dimensional_analysis": {{{{
        "passed": true/false,
        "explanation": "Units analysis"
    }}}},
    "limiting_cases": {{{{
        "cases_checked": ["case1", "case2"],
        "results": "Results of limiting case checks"
    }}}},
    "final_answer": "Your final answer (use LaTeX notation if needed)",
    "confidence": 0.0-1.0
}}}}"""

    # Step 2: Compare answers
    COMPARE_PROMPT = f"""You are an expert physics professor comparing two solutions to the same problem.

QUESTION:
{{query}}

YOUR INDEPENDENTLY DERIVED ANSWER:
{{blind_answer}}

YOUR DERIVATION APPROACH:
{{blind_derivation}}

PROVIDED ANSWER TO VERIFY:
{{response_answer}}

PROVIDED SOLUTION/REASONING:
{{response_reasoning}}

Compare your independently derived answer to the provided answer.

IMPORTANT: Be rigorous but fair. Equivalent forms of the same answer are acceptable:
- \\frac{{1}}{{2}} = 0.5 = 1/2
- 2\\pi\\omega = \\omega \\cdot 2\\pi
- \\frac{{mv^2}}{{2}} = \\frac{{1}}{{2}}mv^2
- Answers that differ only by constant factors that cancel out
- Different but mathematically equivalent expressions

{JSON_INSTRUCTION}

Respond with JSON:
{{{{
    "my_answer": "Your answer (restated)",
    "provided_answer": "The provided answer (restated)",
    "answers_equivalent": true/false,
    "equivalence_explanation": "Detailed explanation of why the answers are or are not equivalent",
    "mathematical_issues_in_provided": ["list any mathematical errors found in the provided solution"],
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "overall_explanation": "Final verdict explanation"
}}}}"""

    def __init__(
        self,
        client: OpenRouterClient,
        judge_model: str = "anthropic/claude-opus-4",
        required_consecutive_passes: int = 5,
    ):
        """
        Initialize the final answer validator.

        Args:
            client: OpenRouter API client
            judge_model: Model to use for self-judging (should be same as generation model)
            required_consecutive_passes: Number of consecutive passes required (default 5)
        """
        self.client = client
        self.judge_model = judge_model
        self.required_passes = required_consecutive_passes

    async def _blind_solve(
        self,
        qa: PhysicsQADataPoint,
        attempt_num: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Step 1: Have Claude solve the problem BLIND (without seeing the provided answer).

        Args:
            qa: The QA data point (only the question is used)
            attempt_num: Which attempt this is (for logging)

        Returns:
            Tuple of (success, solve_details with final_answer and derivation)
        """
        prompt = self.BLIND_SOLVE_PROMPT.replace("{{query}}", qa.query)

        try:
            response = await self.client.chat_completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )

            content = response["choices"][0]["message"]["content"]
            data = extract_json_from_response(content)

            final_answer = data.get("final_answer", "")
            derivation = data.get("derivation", "")
            confidence = data.get("confidence", 0.0)

            logger.debug(
                f"  Blind solve {attempt_num}: answer='{final_answer[:50]}...', "
                f"confidence={confidence:.2f}"
            )

            return True, data

        except Exception as e:
            logger.error(f"Blind solve {attempt_num} failed: {e}")
            return False, {"error": str(e)}

    async def _compare_answers(
        self,
        qa: PhysicsQADataPoint,
        blind_result: Dict[str, Any],
        attempt_num: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Step 2: Compare the blind-derived answer to the provided answer.

        Args:
            qa: The QA data point
            blind_result: The result from the blind solve step
            attempt_num: Which attempt this is (for logging)

        Returns:
            Tuple of (is_correct, comparison_details)
        """
        blind_answer = blind_result.get("final_answer", "")
        blind_derivation = blind_result.get("derivation", "")

        prompt = (self.COMPARE_PROMPT
            .replace("{{query}}", qa.query)
            .replace("{{blind_answer}}", blind_answer)
            .replace("{{blind_derivation}}", blind_derivation)
            .replace("{{response_answer}}", qa.response_answer)
            .replace("{{response_reasoning}}", qa.response_reasoning)
        )

        try:
            response = await self.client.chat_completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            content = response["choices"][0]["message"]["content"]
            data = extract_json_from_response(content)

            is_correct = data.get("is_correct", False)
            answers_equivalent = data.get("answers_equivalent", False)
            confidence = data.get("confidence", 0.0)

            logger.debug(
                f"  Compare {attempt_num}: correct={is_correct}, "
                f"equivalent={answers_equivalent}, confidence={confidence:.2f}"
            )

            # Include the blind solve result in the judgment for feedback extraction
            data["blind_solve_answer"] = blind_answer
            data["blind_solve_derivation"] = blind_derivation

            return is_correct, data

        except Exception as e:
            logger.error(f"Compare {attempt_num} failed: {e}")
            return False, {"error": str(e), "is_correct": False}

    async def _single_judgment(
        self,
        qa: PhysicsQADataPoint,
        attempt_num: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform a single TWO-STEP BLIND judgment of the QA pair.

        Step 1: Claude solves the problem without seeing the answer
        Step 2: Claude compares its answer to the provided answer

        Args:
            qa: The QA data point to judge
            attempt_num: Which attempt this is (for logging)

        Returns:
            Tuple of (is_correct, judgment_details)
        """
        # Step 1: Blind solve
        logger.debug(f"  Judgment {attempt_num} Step 1: Blind solving...")
        solve_success, blind_result = await self._blind_solve(qa, attempt_num)

        if not solve_success:
            return False, {
                "error": "Blind solve failed",
                "is_correct": False,
                "blind_result": blind_result
            }

        # Step 2: Compare answers
        logger.debug(f"  Judgment {attempt_num} Step 2: Comparing answers...")
        is_correct, compare_result = await self._compare_answers(qa, blind_result, attempt_num)

        # Merge results for comprehensive feedback
        judgment = {
            **compare_result,
            "blind_solve": blind_result,
            "my_derived_answer": blind_result.get("final_answer", ""),
        }

        logger.debug(
            f"  Judgment {attempt_num}: correct={is_correct}, "
            f"equivalent={compare_result.get('answers_equivalent', False)}"
        )

        return is_correct, judgment

    async def validate(
        self,
        qa: PhysicsQADataPoint,
    ) -> Tuple[bool, int, List[Dict[str, Any]]]:
        """
        Validate a QA pair with 5 consecutive judgments.

        Args:
            qa: The QA data point to validate

        Returns:
            Tuple of:
            - passed: True if all 5 judgments confirm correctness
            - pass_count: Number of judgments that passed
            - judgments: List of all judgment details
        """
        logger.info(f"Running final answer validation for QA {qa.id} ({self.required_passes}x verification)...")

        judgments = []
        pass_count = 0

        for i in range(self.required_passes):
            is_correct, details = await self._single_judgment(qa, i + 1)
            judgments.append(details)

            if is_correct:
                pass_count += 1
            else:
                # Early exit - if any judgment fails, the QA fails
                logger.warning(
                    f"QA {qa.id} failed judgment {i + 1}/{self.required_passes}. "
                    f"Reason: {details.get('overall_explanation', 'Unknown')}"
                )
                # Continue to get all judgments for analysis

        passed = pass_count == self.required_passes

        if passed:
            logger.info(f"QA {qa.id} PASSED final answer validation ({pass_count}/{self.required_passes})")
        else:
            logger.warning(f"QA {qa.id} FAILED final answer validation ({pass_count}/{self.required_passes})")

        return passed, pass_count, judgments

    def _extract_feedback_from_judgments(
        self,
        judgments: List[Dict[str, Any]],
    ) -> str:
        """
        Extract actionable feedback from failed judgments for regeneration.

        Works with the two-step blind judgment structure.

        Args:
            judgments: List of judgment details

        Returns:
            Feedback string for regeneration
        """
        issues = []
        derived_answers = []

        for i, j in enumerate(judgments):
            if not j.get("is_correct", False):
                # Collect the model's independently derived answer (from blind solve)
                if j.get("my_derived_answer"):
                    derived_answers.append(f"Blind judgment {i+1}: {j['my_derived_answer']}")

                # Collect equivalence explanation (from compare step)
                if j.get("equivalence_explanation"):
                    issues.append(f"- {j['equivalence_explanation']}")

                # Collect overall explanation
                if j.get("overall_explanation"):
                    issues.append(f"- {j['overall_explanation']}")

                # Collect mathematical issues found in the provided solution
                math_issues = j.get("mathematical_issues_in_provided", [])
                if math_issues:
                    issues.extend([f"- Math error: {issue}" for issue in math_issues])

                # Check blind solve dimensional analysis
                blind_solve = j.get("blind_solve", {})
                if isinstance(blind_solve, dict):
                    dim_check = blind_solve.get("dimensional_analysis", {})
                    if isinstance(dim_check, dict) and not dim_check.get("passed", True):
                        issues.append(f"- Dimensional analysis: {dim_check.get('explanation', 'failed')}")

                    # Check limiting cases from blind solve
                    limit_check = blind_solve.get("limiting_cases", {})
                    if isinstance(limit_check, dict):
                        results = limit_check.get("results", "")
                        if results:
                            issues.append(f"- Limiting cases: {results}")

        feedback = "VALIDATION FAILED - The answer appears to be INCORRECT.\n\n"

        if derived_answers:
            feedback += "INDEPENDENT DERIVATIONS found different answers:\n"
            feedback += "\n".join(derived_answers[:3])  # Show up to 3
            feedback += "\n\n"

        if issues:
            feedback += "ISSUES IDENTIFIED:\n"
            feedback += "\n".join(issues[:5])  # Show up to 5 issues
            feedback += "\n\n"

        feedback += (
            "You MUST:\n"
            "1. Re-derive the answer completely from first principles\n"
            "2. Check your algebra step-by-step\n"
            "3. Verify dimensional analysis\n"
            "4. Check limiting cases (e.g., what happens as parameters -> 0 or -> infinity)\n"
            "5. If there's ambiguity in the problem, clarify it\n"
        )

        return feedback

    async def validate_with_regeneration(
        self,
        qa: PhysicsQADataPoint,
        regenerator,  # QAGenerator instance
        max_regen_attempts: int = 3,
    ) -> Tuple[Optional[PhysicsQADataPoint], bool, int, Dict[str, Any]]:
        """
        Validate a QA pair, regenerating if it fails.

        Args:
            qa: The QA data point to validate
            regenerator: QAGenerator instance for regeneration
            max_regen_attempts: Maximum regeneration attempts

        Returns:
            Tuple of:
            - final_qa: The validated QA (or None if all attempts failed)
            - passed: Whether validation ultimately passed
            - attempts_used: Number of validation/regen cycles used
            - details: Validation details
        """
        current_qa = qa
        attempts_used = 0

        for attempt in range(max_regen_attempts + 1):  # +1 for initial attempt
            attempts_used = attempt + 1

            # Validate current QA
            passed, pass_count, judgments = await self.validate(current_qa)

            if passed:
                logger.info(
                    f"QA {current_qa.id} passed final validation on attempt {attempts_used}"
                )
                return current_qa, True, attempts_used, {
                    "passed": True,
                    "attempts": attempts_used,
                    "final_pass_count": pass_count,
                }

            # If this was the last attempt, give up
            if attempt >= max_regen_attempts:
                logger.warning(
                    f"QA {current_qa.id} failed final validation after {attempts_used} attempts"
                )
                return None, False, attempts_used, {
                    "passed": False,
                    "attempts": attempts_used,
                    "final_pass_count": pass_count,
                    "last_judgments": judgments,
                }

            # Extract feedback and regenerate
            feedback = self._extract_feedback_from_judgments(judgments)
            logger.info(
                f"QA {current_qa.id} failed validation ({pass_count}/{self.required_passes}), "
                f"regenerating (attempt {attempt + 2}/{max_regen_attempts + 1})..."
            )

            try:
                current_qa = await regenerator.regenerate_with_feedback(
                    original=current_qa,
                    feedback=feedback,
                    temperature=0.7,
                )
                logger.info(f"Regenerated QA, new ID: {current_qa.id}")
            except Exception as e:
                logger.error(f"Regeneration failed: {e}")
                return None, False, attempts_used, {
                    "passed": False,
                    "attempts": attempts_used,
                    "error": str(e),
                }

        # Should not reach here
        return None, False, attempts_used, {"passed": False, "attempts": attempts_used}

    async def validate_batch(
        self,
        qa_pairs: List[PhysicsQADataPoint],
        regenerator=None,  # Optional QAGenerator for regeneration
        max_regen_attempts: int = 3,
    ) -> Tuple[List[PhysicsQADataPoint], List[PhysicsQADataPoint], Dict[str, Any]]:
        """
        Validate a batch of QA pairs after generation is complete.

        If a regenerator is provided, failed QA pairs will be regenerated
        and re-validated up to max_regen_attempts times.

        Args:
            qa_pairs: List of QA pairs to validate
            regenerator: Optional QAGenerator for regenerating failed pairs
            max_regen_attempts: Max regeneration attempts per QA pair

        Returns:
            Tuple of:
            - passed_pairs: QA pairs that passed all 5 judgments
            - failed_pairs: QA pairs that failed validation (even after regeneration)
            - summary: Summary statistics
        """
        logger.info(f"Starting final answer validation for {len(qa_pairs)} QA pairs...")
        if regenerator:
            logger.info(f"Regeneration enabled: up to {max_regen_attempts} attempts per failed QA")

        passed_pairs = []
        failed_pairs = []
        all_results = []
        total_regen_attempts = 0

        for i, qa in enumerate(qa_pairs):
            logger.info(f"Validating QA {i + 1}/{len(qa_pairs)} (ID: {qa.id})...")

            if regenerator:
                # Validate with regeneration on failure
                final_qa, passed, attempts, details = await self.validate_with_regeneration(
                    qa=qa,
                    regenerator=regenerator,
                    max_regen_attempts=max_regen_attempts,
                )
                total_regen_attempts += max(0, attempts - 1)  # Count only regen attempts

                result = {
                    "id": qa.id,
                    "final_id": final_qa.id if final_qa else None,
                    "passed": passed,
                    "attempts": attempts,
                    "topic": qa.topic.value if qa.topic else None,
                    "subtopic": qa.subtopic,
                }
                all_results.append(result)

                if passed and final_qa:
                    passed_pairs.append(final_qa)
                else:
                    failed_pairs.append(qa)
            else:
                # Simple validation without regeneration
                passed, pass_count, judgments = await self.validate(qa)

                result = {
                    "id": qa.id,
                    "passed": passed,
                    "pass_count": pass_count,
                    "required": self.required_passes,
                    "topic": qa.topic.value if qa.topic else None,
                    "subtopic": qa.subtopic,
                }
                all_results.append(result)

                if passed:
                    passed_pairs.append(qa)
                else:
                    failed_pairs.append(qa)

        # Summary statistics
        total = len(qa_pairs)
        passed_count = len(passed_pairs)
        pass_rate = passed_count / total if total > 0 else 0

        summary = {
            "total_validated": total,
            "passed": passed_count,
            "failed": len(failed_pairs),
            "pass_rate": f"{pass_rate:.1%}",
            "required_consecutive_passes": self.required_passes,
            "regeneration_enabled": regenerator is not None,
            "total_regeneration_attempts": total_regen_attempts,
            "results": all_results,
        }

        logger.info(
            f"Final answer validation complete: {passed_count}/{total} passed ({pass_rate:.1%})"
        )
        if regenerator:
            logger.info(f"Total regeneration attempts: {total_regen_attempts}")

        return passed_pairs, failed_pairs, summary
