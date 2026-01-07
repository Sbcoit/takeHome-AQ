"""Correctness validation using LLM-as-judge."""

import asyncio
import json
import logging
import re
from typing import Tuple, Dict, Any, List

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint

logger = logging.getLogger(__name__)


def extract_json_from_response(content: str) -> dict:
    """
    Extract JSON from a response that may contain markdown or other text.
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

    # Try to find JSON object in the text
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(content[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        f"Could not extract JSON from response: {content[:200]}...",
        content, 0
    )


class CorrectnessJudge:
    """
    Validates that QA pairs are correct and well-posed.

    Rule: The final dataset should have 90%+ accuracy where "correct" means:
    - The query is well-posed and internally consistent
    - The response_answer is correct
    - The response_reasoning is correct and supports the answer
    """

    JUDGE_PROMPT = """You are an expert physics professor reviewing exam materials for quality and correctness.

Evaluate this physics question-answer pair for correctness and quality:

**QUESTION:**
{query}

**PROVIDED ANSWER:**
{response_answer}

**PROVIDED SOLUTION:**
{response_reasoning}

**GRADING RUBRIC:**
{rubric}

Evaluate on these criteria:

1. **Well-Posed Question** (is the question unambiguous and self-contained?):
   - Does it provide all necessary information to solve the problem?
   - Are there any ambiguities or missing parameters?
   - Is it solvable without external resources or images?

2. **Correct Answer** (is the final answer correct?):
   - Check dimensional analysis
   - Verify numerical/symbolic correctness
   - Check for proper units

3. **Correct Reasoning** (is the solution mathematically and physically correct?):
   - Are the physics principles correctly applied?
   - Are the mathematical steps valid?
   - Does the reasoning logically lead to the answer?

4. **Rubric Appropriateness** (does the rubric properly assess the problem?):
   - Do the criteria cover the key aspects of the solution?
   - Are the point allocations reasonable?

5. **Graduate Level** (is this appropriate for PhD qualifying exams?):
   - Is the difficulty appropriate for graduate students?
   - Does it require synthesis of multiple concepts?

Respond with JSON:
{{
    "well_posed": {{
        "score": 0-100,
        "issues": ["list any issues"],
        "is_acceptable": true/false
    }},
    "answer_correct": {{
        "score": 0-100,
        "issues": ["list any issues"],
        "is_acceptable": true/false
    }},
    "reasoning_correct": {{
        "score": 0-100,
        "issues": ["list any issues"],
        "is_acceptable": true/false
    }},
    "rubric_appropriate": {{
        "score": 0-100,
        "issues": ["list any issues"],
        "is_acceptable": true/false
    }},
    "graduate_level": {{
        "score": 0-100,
        "issues": ["list any issues"],
        "is_acceptable": true/false
    }},
    "overall_correct": true/false,
    "overall_score": 0-100,
    "confidence": 0.0-1.0,
    "summary": "Brief summary of the evaluation"
}}

Be rigorous but fair. A question can have minor issues and still be acceptable.
Mark overall_correct as true only if all core criteria (well_posed, answer_correct, reasoning_correct) are acceptable."""

    def __init__(
        self,
        client: OpenRouterClient,
        judge_model: str = "anthropic/claude-sonnet-4",
        correctness_threshold: float = 0.90,
    ):
        """
        Initialize the correctness judge.

        Args:
            client: OpenRouter API client
            judge_model: Model to use for judging
            correctness_threshold: Required accuracy for dataset (default 90%)
        """
        self.client = client
        self.judge_model = judge_model
        self.correctness_threshold = correctness_threshold

    async def judge(self, qa: PhysicsQADataPoint) -> Tuple[float, Dict[str, Any]]:
        """
        Judge a single QA pair for correctness.

        Args:
            qa: The QA data point to judge

        Returns:
            Tuple of (score 0-1, detailed results)
        """
        rubric_text = json.dumps(qa.rubric.model_dump(), indent=2)

        prompt = self.JUDGE_PROMPT.format(
            query=qa.query,
            response_answer=qa.response_answer,
            response_reasoning=qa.response_reasoning,
            rubric=rubric_text,
        )

        try:
            response = await self.client.chat_completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low for consistent evaluation
                max_tokens=8192,
                response_format={"type": "json_object"},
            )

            content = response["choices"][0]["message"]["content"]
            data = extract_json_from_response(content)

            # Calculate score (normalize to 0-1)
            overall_score = data.get("overall_score", 0) / 100
            is_correct = data.get("overall_correct", False)

            # If marked as correct, ensure score is at least 0.9
            if is_correct and overall_score < 0.9:
                overall_score = 0.9

            return overall_score, data

        except Exception as e:
            logger.error(f"Failed to judge correctness: {e}")
            return 0.0, {
                "error": str(e),
                "overall_correct": False,
                "overall_score": 0,
            }

    async def judge_single(self, qa: PhysicsQADataPoint) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Judge a single QA pair and return pass/fail.

        Args:
            qa: The QA data point to judge

        Returns:
            Tuple of (passed, score, details)
        """
        score, details = await self.judge(qa)
        passed = details.get("overall_correct", False)
        return passed, score, details

    async def validate_dataset(
        self,
        qa_pairs: List[PhysicsQADataPoint],
    ) -> Tuple[float, bool, List[Dict[str, Any]]]:
        """
        Validate an entire dataset for correctness.

        Args:
            qa_pairs: List of QA data points to validate

        Returns:
            Tuple of:
            - accuracy: Proportion of correct QA pairs
            - passed: True if accuracy >= threshold
            - details: List of detailed results for each pair
        """
        logger.info(f"Validating dataset of {len(qa_pairs)} QA pairs for correctness...")

        # Judge all pairs concurrently (with some batching for large datasets)
        batch_size = 10
        all_results = []

        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i + batch_size]
            tasks = [self.judge(qa) for qa in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for qa, result in zip(batch, results):
                if isinstance(result, Exception):
                    all_results.append({
                        "id": qa.id,
                        "error": str(result),
                        "score": 0,
                        "correct": False,
                    })
                else:
                    score, details = result
                    all_results.append({
                        "id": qa.id,
                        "score": score,
                        "correct": details.get("overall_correct", False),
                        "details": details,
                    })

            logger.info(f"Judged {min(i + batch_size, len(qa_pairs))}/{len(qa_pairs)} pairs")

        # Calculate accuracy
        correct_count = sum(1 for r in all_results if r["correct"])
        accuracy = correct_count / len(qa_pairs) if qa_pairs else 0

        passed = accuracy >= self.correctness_threshold

        logger.info(
            f"Dataset correctness: {correct_count}/{len(qa_pairs)} correct "
            f"({accuracy:.1%}). Threshold: {self.correctness_threshold:.0%}. "
            f"Passed: {passed}"
        )

        return accuracy, passed, all_results

    async def get_improvement_suggestions(
        self,
        qa: PhysicsQADataPoint,
        judgment: Dict[str, Any],
    ) -> str:
        """
        Get specific suggestions for improving a QA pair.

        Args:
            qa: The QA data point
            judgment: The judgment details from judge()

        Returns:
            String with improvement suggestions
        """
        # Collect issues from all categories
        all_issues = []
        for category in ["well_posed", "answer_correct", "reasoning_correct", "rubric_appropriate"]:
            if category in judgment:
                issues = judgment[category].get("issues", [])
                if issues:
                    all_issues.extend([f"[{category}] {issue}" for issue in issues])

        if not all_issues:
            return "No specific issues identified."

        return "Improvement suggestions:\n" + "\n".join(f"- {issue}" for issue in all_issues)
