"""Answer verification - independently solve and verify reference answers."""

import asyncio
import logging
from typing import Tuple, Optional

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint
from ..utils import extract_json_from_response
from ..prompts import JSON_INSTRUCTION, ANSWER_FORMAT_GUIDE, FINAL_ANSWER_SCHEMA, EQUIVALENCE_SCHEMA

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """
    Verifies that generated reference answers are correct by having an
    independent model solve the question and compare answers.

    This catches cases where the generator model made calculation errors
    in its reference answer.
    """

    SOLVE_PROMPT = f"""Solve this physics problem.

PROBLEM:
{{query}}
{ANSWER_FORMAT_GUIDE}
{JSON_INSTRUCTION}

{FINAL_ANSWER_SCHEMA}"""

    COMPARE_PROMPT = f"""Are these two physics answers equivalent?

ANSWER A: {{reference_answer}}
ANSWER B: {{verification_answer}}

Equivalent = same value, algebraically equal, or differ only in notation.
{JSON_INSTRUCTION}

{EQUIVALENCE_SCHEMA}"""

    def __init__(
        self,
        client: OpenRouterClient,
        solver_model: str = "anthropic/claude-sonnet-4",
        judge_model: str = "anthropic/claude-sonnet-4",
    ):
        """
        Initialize the answer verifier.

        Args:
            client: OpenRouter API client
            solver_model: Model to independently solve the problem
            judge_model: Model to compare the answers
        """
        self.client = client
        self.solver_model = solver_model
        self.judge_model = judge_model

    async def _solve_independently(self, query: str) -> Tuple[str, str]:
        """
        Have an independent model solve the problem.

        Returns:
            Tuple of (solution, final_answer)
        """
        prompt = self.SOLVE_PROMPT.replace("{{query}}", query)

        try:
            response = await self.client.chat_completion(
                model=self.solver_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent solving
                max_tokens=32000,
                response_format={"type": "json_object"},
            )

            content = response["choices"][0]["message"]["content"]
            data = extract_json_from_response(content)

            final_answer = data.get("final_answer", "")

            return "", final_answer  # No solution needed, just the answer

        except Exception as e:
            logger.error(f"Failed to solve independently: {e}")
            return "", f"ERROR: {e}"

    async def _compare_answers(
        self,
        reference_answer: str,
        verification_answer: str,
    ) -> Tuple[bool, str]:
        """
        Compare two answers to determine if they're equivalent.

        Returns:
            Tuple of (equivalent, explanation)
        """
        prompt = (self.COMPARE_PROMPT
            .replace("{{reference_answer}}", reference_answer)
            .replace("{{verification_answer}}", verification_answer)
        )

        try:
            response = await self.client.chat_completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            content = response["choices"][0]["message"]["content"]
            data = extract_json_from_response(content)

            equivalent = data.get("equivalent", False)
            explanation = data.get("explanation", "")

            return equivalent, explanation

        except Exception as e:
            logger.error(f"Failed to compare answers: {e}")
            return False, f"ERROR: {e}"

    async def verify(
        self,
        qa: PhysicsQADataPoint,
        num_attempts: int = 2,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Verify that the reference answer is correct by independently solving.

        Args:
            qa: The QA pair to verify
            num_attempts: Number of independent solve attempts (majority vote)

        Returns:
            Tuple of (is_valid, explanation, corrected_answer)
            - is_valid: True if reference answer is verified correct
            - explanation: Why it passed or failed
            - corrected_answer: If invalid, the correct answer (or None)
        """
        logger.info(f"Verifying answer for question: {qa.query[:100]}...")

        # Solve independently multiple times
        tasks = [self._solve_independently(qa.query) for _ in range(num_attempts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        verified_answers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Solve attempt {i+1} failed: {result}")
                continue

            solution, answer = result
            if answer.startswith("ERROR:"):
                logger.warning(f"Solve attempt {i+1} error: {answer}")
                continue

            # Compare this answer with the reference
            equivalent, explanation = await self._compare_answers(
                qa.response_answer, answer
            )

            logger.debug(
                f"Attempt {i+1}: answer='{answer}', "
                f"equivalent={equivalent}, explanation={explanation}"
            )

            verified_answers.append({
                "answer": answer,
                "solution": solution,
                "equivalent": equivalent,
                "explanation": explanation,
            })

        if not verified_answers:
            logger.warning("All verification attempts failed")
            return False, "All verification attempts failed", None

        # Check if majority agree with reference
        equivalent_count = sum(1 for v in verified_answers if v["equivalent"])
        total = len(verified_answers)

        logger.info(
            f"Answer verification: {equivalent_count}/{total} attempts "
            f"agree with reference answer '{qa.response_answer}'"
        )

        if equivalent_count >= (total / 2):
            # Majority agree - reference is likely correct
            return True, f"{equivalent_count}/{total} verifications agree with reference", None
        else:
            # Majority disagree - reference is likely wrong
            # Return the most common alternative answer
            alternative = verified_answers[0]["answer"]
            explanation = verified_answers[0]["explanation"]
            return (
                False,
                f"Reference answer appears incorrect. {explanation}",
                alternative,
            )
