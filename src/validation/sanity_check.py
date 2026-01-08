"""Sanity check validation for physics QA pairs.

This module implements quick sanity checks that catch ~80% of errors:
1. Dimensional Analysis - Does the answer have the right units?
2. Limiting Cases - Does the answer behave correctly at extremes?
3. Sign Check - Is the sign physically reasonable?
4. Order of Magnitude - Is the numerical value plausible?
5. Internal Consistency - Do intermediate steps agree with the final answer?

These checks run EARLY in the pipeline (after schema validation, before Qwen)
to catch bad QAs before wasting expensive API calls on cross-check validation.

MULTI-MODEL MODE: Can run sanity checks with 2-3 different models and require
consensus, catching model-specific blind spots.
"""

import asyncio
import json
import logging
from typing import Tuple, Dict, Any, List, Optional

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint
from ..utils import extract_json_from_response
from ..prompts import JSON_INSTRUCTION

logger = logging.getLogger(__name__)


# Default models for multi-model sanity checking
# Using diverse architectures to avoid correlated blind spots
DEFAULT_SANITY_CHECK_MODELS = [
    "anthropic/claude-opus-4",      # Primary: Claude for physics reasoning
    "openai/gpt-4o",                # Secondary: GPT-4 for different perspective
    "google/gemini-2.5-pro-preview",        # Tertiary: Gemini for additional diversity
]


class SanityCheckValidator:
    """
    Validates physics QA pairs using quick sanity checks.

    These checks catch obvious errors early, before expensive cross-validation:
    - Dimensional analysis (catches ~80% of errors alone)
    - Limiting cases
    - Sign check
    - Order of magnitude
    - Internal consistency between derivation and answer

    Uses a single API call to Claude Opus for fast, accurate checking.
    """

    SANITY_CHECK_PROMPT = """You are an expert physics professor performing SANITY CHECKS on a student's solution.

Your job is to quickly catch OBVIOUS ERRORS using these 5 checks:

QUESTION:
{query}

STUDENT'S ANSWER:
{response_answer}

STUDENT'S DERIVATION:
{response_reasoning}

---

Perform these 5 SANITY CHECKS:

1. **DIMENSIONAL ANALYSIS** (catches ~80% of errors):
   - What are the expected dimensions of the answer? (e.g., [Energy] = [M L² T⁻²])
   - What are the actual dimensions of the student's answer?
   - Do they match?
   - Example failure: Problem asks for length but answer has units of energy

2. **LIMITING CASES**:
   - Pick 2 extreme cases (e.g., parameter → 0, parameter → ∞)
   - Does the answer behave correctly in these limits?
   - Example failure: As perturbation → 0, answer should reduce to unperturbed case

3. **SIGN CHECK**:
   - Is the sign physically reasonable?
   - Energies should be real, probabilities should be 0-1, binding energies negative
   - Example failure: Probability > 1 or negative, or unphysical complex answer

4. **ORDER OF MAGNITUDE**:
   - Is the numerical scale plausible?
   - Example failure: Length smaller than Planck length (10⁻³⁵ m), speed > c

5. **INTERNAL CONSISTENCY**:
   - Does the final answer match the derivation?
   - Are intermediate results consistent with the final answer?
   - Example failure: Derivation says A = 2πE/ω₀ but final answer has 2πE/(mω₀²)

---

{json_instruction}

Respond with JSON:
{{
    "dimensional_analysis": {{
        "expected_dimensions": "e.g., [M L² T⁻²] for energy",
        "actual_dimensions": "dimensions of the given answer",
        "match": true/false,
        "explanation": "why they match or don't match"
    }},
    "limiting_cases": {{
        "cases_checked": [
            {{"case": "parameter → value", "expected_behavior": "what should happen", "actual_behavior": "what the answer gives", "passed": true/false}}
        ],
        "passed": true/false,
        "explanation": "summary of limiting case analysis"
    }},
    "sign_check": {{
        "expected_sign": "positive/negative/either with reason",
        "actual_sign": "positive/negative/complex/undefined",
        "passed": true/false,
        "explanation": "why the sign is or isn't physical"
    }},
    "order_of_magnitude": {{
        "passed": true/false,
        "explanation": "why the scale is or isn't plausible"
    }},
    "internal_consistency": {{
        "passed": true/false,
        "inconsistencies": ["list any inconsistencies found"],
        "explanation": "summary of consistency check"
    }},
    "overall_passed": true/false,
    "critical_failures": ["list of critical failures that MUST be fixed"],
    "confidence": 0.0-1.0,
    "summary": "Brief summary of all checks"
}}"""

    def __init__(
        self,
        client: OpenRouterClient,
        check_model: str = "anthropic/claude-opus-4",
    ):
        """
        Initialize the sanity check validator.

        Args:
            client: OpenRouter API client
            check_model: Model to use for checking (Claude Opus for physics accuracy)
        """
        self.client = client
        self.check_model = check_model

    async def validate(
        self,
        qa: PhysicsQADataPoint,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Run sanity checks on a QA pair.

        Args:
            qa: The QA data point to validate

        Returns:
            Tuple of:
            - passed: True if all critical checks pass
            - details: Dict with detailed check results
            - feedback: String feedback for regeneration if failed
        """
        logger.info(f"Running sanity checks on QA {qa.id}...")

        prompt = self.SANITY_CHECK_PROMPT.format(
            query=qa.query,
            response_answer=qa.response_answer,
            response_reasoning=qa.response_reasoning,
            json_instruction=JSON_INSTRUCTION,
        )

        try:
            response = await self.client.chat_completion(
                model=self.check_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low for consistent checking
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            content = response["choices"][0]["message"]["content"]
            data = extract_json_from_response(content)

            # Extract results
            dim_check = data.get("dimensional_analysis", {})
            limit_check = data.get("limiting_cases", {})
            sign_check = data.get("sign_check", {})
            magnitude_check = data.get("order_of_magnitude", {})
            consistency_check = data.get("internal_consistency", {})

            overall_passed = data.get("overall_passed", False)
            critical_failures = data.get("critical_failures", [])
            confidence = data.get("confidence", 0.5)

            # Log results
            dim_passed = dim_check.get("match", False)
            limit_passed = limit_check.get("passed", False)
            sign_passed = sign_check.get("passed", False)
            mag_passed = magnitude_check.get("passed", False)
            cons_passed = consistency_check.get("passed", False)

            logger.info(
                f"Sanity checks for {qa.id}: "
                f"dim={dim_passed}, limits={limit_passed}, sign={sign_passed}, "
                f"magnitude={mag_passed}, consistency={cons_passed}. "
                f"Overall: {'PASS' if overall_passed else 'FAIL'}"
            )

            # Generate feedback if failed
            feedback = ""
            if not overall_passed:
                feedback = self._generate_feedback(data, critical_failures)

            return overall_passed, data, feedback

        except Exception as e:
            logger.error(f"Sanity check failed with error: {e}")
            return False, {"error": str(e)}, f"Sanity check failed: {e}"

    def _generate_feedback(
        self,
        results: Dict[str, Any],
        critical_failures: List[str],
    ) -> str:
        """
        Generate actionable feedback for regeneration.

        Args:
            results: Full sanity check results
            critical_failures: List of critical failure descriptions

        Returns:
            Feedback string for the generator
        """
        feedback_parts = ["SANITY CHECK FAILED - The answer has obvious errors that must be fixed.\n"]

        # Dimensional analysis failure (most critical)
        dim_check = results.get("dimensional_analysis", {})
        if not dim_check.get("match", True):
            feedback_parts.append(
                f"DIMENSIONAL ANALYSIS FAILED:\n"
                f"  - Expected dimensions: {dim_check.get('expected_dimensions', 'unknown')}\n"
                f"  - Actual dimensions: {dim_check.get('actual_dimensions', 'unknown')}\n"
                f"  - {dim_check.get('explanation', '')}\n"
                f"  YOU MUST fix this - dimensions MUST match.\n"
            )

        # Limiting cases failure
        limit_check = results.get("limiting_cases", {})
        if not limit_check.get("passed", True):
            feedback_parts.append(
                f"LIMITING CASES FAILED:\n"
                f"  - {limit_check.get('explanation', 'Answer does not behave correctly at extremes')}\n"
            )

        # Sign check failure
        sign_check = results.get("sign_check", {})
        if not sign_check.get("passed", True):
            feedback_parts.append(
                f"SIGN CHECK FAILED:\n"
                f"  - Expected: {sign_check.get('expected_sign', 'unknown')}\n"
                f"  - Got: {sign_check.get('actual_sign', 'unknown')}\n"
                f"  - {sign_check.get('explanation', '')}\n"
            )

        # Order of magnitude failure
        mag_check = results.get("order_of_magnitude", {})
        if not mag_check.get("passed", True):
            feedback_parts.append(
                f"ORDER OF MAGNITUDE FAILED:\n"
                f"  - {mag_check.get('explanation', 'Numerical scale is implausible')}\n"
            )

        # Internal consistency failure
        cons_check = results.get("internal_consistency", {})
        if not cons_check.get("passed", True):
            inconsistencies = cons_check.get("inconsistencies", [])
            feedback_parts.append(
                f"INTERNAL CONSISTENCY FAILED:\n"
                f"  - {cons_check.get('explanation', 'Derivation does not match final answer')}\n"
            )
            for inc in inconsistencies[:3]:
                feedback_parts.append(f"  - {inc}\n")

        # Add critical failures
        if critical_failures:
            feedback_parts.append("\nCRITICAL ISSUES:\n")
            for failure in critical_failures[:5]:
                feedback_parts.append(f"  - {failure}\n")

        feedback_parts.append(
            "\nYou MUST:\n"
            "1. Re-derive the answer from first principles\n"
            "2. Track dimensions at EVERY step\n"
            "3. Verify limiting cases (at least 2)\n"
            "4. Check the sign makes physical sense\n"
            "5. Ensure intermediate steps match the final answer\n"
        )

        return "".join(feedback_parts)


class MultiModelSanityValidator:
    """
    Multi-model sanity check validator that requires CONSENSUS.

    Runs the same sanity checks on 2-3 different AI models and requires
    agreement to pass. This catches model-specific blind spots that a
    single model might miss.

    Consensus modes:
    - "all": ALL models must pass (strictest)
    - "majority": Majority of models must pass (default)
    - "any": At least one model must pass (most lenient)
    """

    def __init__(
        self,
        client: OpenRouterClient,
        models: Optional[List[str]] = None,
        consensus_mode: str = "majority",
        min_models_required: int = 2,
    ):
        """
        Initialize multi-model sanity validator.

        Args:
            client: OpenRouter API client
            models: List of models to use (defaults to Claude, GPT-4, Gemini)
            consensus_mode: "all", "majority", or "any"
            min_models_required: Minimum models that must return results
        """
        self.client = client
        self.models = models or DEFAULT_SANITY_CHECK_MODELS[:3]
        self.consensus_mode = consensus_mode
        self.min_models_required = min_models_required

        # Create individual validators for each model
        self.validators = [
            SanityCheckValidator(client, check_model=model)
            for model in self.models
        ]

    async def validate(
        self,
        qa: PhysicsQADataPoint,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Run multi-model sanity checks with consensus requirement.

        Args:
            qa: The QA data point to validate

        Returns:
            Tuple of:
            - passed: True if consensus is reached for passing
            - details: Dict with all model results and consensus info
            - feedback: Combined feedback from failing models
        """
        logger.info(
            f"Running multi-model sanity check on QA {qa.id} "
            f"with {len(self.models)} models ({self.consensus_mode} consensus)..."
        )

        # Run all validators in parallel
        tasks = [
            validator.validate(qa)
            for validator in self.validators
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        model_results = []
        passed_count = 0
        failed_count = 0
        error_count = 0
        all_feedback = []

        for i, result in enumerate(results):
            model_name = self.models[i]

            if isinstance(result, Exception):
                logger.warning(f"Model {model_name} failed with error: {result}")
                model_results.append({
                    "model": model_name,
                    "passed": None,
                    "error": str(result),
                })
                error_count += 1
            else:
                passed, details, feedback = result
                model_results.append({
                    "model": model_name,
                    "passed": passed,
                    "details": details,
                })

                if passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    if feedback:
                        all_feedback.append(f"[{model_name}]: {feedback}")

        # Check if we have enough results
        valid_results = passed_count + failed_count
        if valid_results < self.min_models_required:
            logger.warning(
                f"Only {valid_results} models returned results, "
                f"need at least {self.min_models_required}"
            )
            # Be lenient if not enough models responded
            return True, {
                "model_results": model_results,
                "insufficient_results": True,
                "valid_results": valid_results,
            }, ""

        # Determine consensus
        total_models = len(self.models)
        majority_threshold = (total_models // 2) + 1

        if self.consensus_mode == "all":
            consensus_passed = passed_count == valid_results
        elif self.consensus_mode == "majority":
            consensus_passed = passed_count >= majority_threshold
        else:  # "any"
            consensus_passed = passed_count > 0

        # Log consensus result
        logger.info(
            f"Multi-model sanity check for {qa.id}: "
            f"{passed_count}/{valid_results} passed "
            f"({self.consensus_mode} consensus: {'PASS' if consensus_passed else 'FAIL'})"
        )

        # Compile details
        details = {
            "model_results": model_results,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "error_count": error_count,
            "consensus_mode": self.consensus_mode,
            "consensus_passed": consensus_passed,
            "disagreements": self._find_disagreements(model_results),
        }

        # Generate combined feedback
        combined_feedback = ""
        if not consensus_passed:
            combined_feedback = self._generate_consensus_feedback(
                model_results, all_feedback
            )

        return consensus_passed, details, combined_feedback

    def _find_disagreements(self, model_results: List[Dict]) -> List[Dict]:
        """Find specific checks where models disagreed."""
        disagreements = []

        # Extract check results from each model
        check_names = [
            "dimensional_analysis",
            "limiting_cases",
            "sign_check",
            "order_of_magnitude",
            "internal_consistency",
        ]

        for check_name in check_names:
            check_results = []
            for mr in model_results:
                if mr.get("passed") is None:
                    continue
                details = mr.get("details", {})
                check_data = details.get(check_name, {})

                # Get the pass/fail status
                if check_name == "dimensional_analysis":
                    passed = check_data.get("match", None)
                else:
                    passed = check_data.get("passed", None)

                if passed is not None:
                    check_results.append({
                        "model": mr["model"],
                        "passed": passed,
                    })

            # Check for disagreement
            if check_results:
                passed_models = [r for r in check_results if r["passed"]]
                failed_models = [r for r in check_results if not r["passed"]]

                if passed_models and failed_models:
                    disagreements.append({
                        "check": check_name,
                        "passed_by": [r["model"] for r in passed_models],
                        "failed_by": [r["model"] for r in failed_models],
                    })

        return disagreements

    def _generate_consensus_feedback(
        self,
        model_results: List[Dict],
        all_feedback: List[str],
    ) -> str:
        """Generate feedback highlighting multi-model findings."""
        feedback_parts = [
            "MULTI-MODEL SANITY CHECK FAILED\n"
            "Multiple AI models found issues with this answer.\n\n"
        ]

        # Summarize model votes
        passed_models = [mr["model"] for mr in model_results if mr.get("passed")]
        failed_models = [mr["model"] for mr in model_results if mr.get("passed") is False]

        feedback_parts.append(
            f"Model Votes:\n"
            f"  - PASSED: {', '.join(passed_models) if passed_models else 'None'}\n"
            f"  - FAILED: {', '.join(failed_models) if failed_models else 'None'}\n\n"
        )

        # Add specific feedback from failing models
        if all_feedback:
            feedback_parts.append("Issues Found:\n")
            for fb in all_feedback[:3]:  # Limit to avoid overwhelming
                # Truncate long feedback
                if len(fb) > 500:
                    fb = fb[:500] + "..."
                feedback_parts.append(f"{fb}\n\n")

        feedback_parts.append(
            "The answer was flagged by multiple independent AI models.\n"
            "This strongly suggests a genuine error that needs fixing.\n"
        )

        return "".join(feedback_parts)
