"""Derivation audit validator for step-by-step logical verification.

This module implements a derivation audit that checks for:
1. Logical consistency between steps
2. Internal contradictions
3. Unjustified claims or hand-wavy derivations
4. Dimensional analysis at each step
5. Missing factors or sign errors

Unlike the final answer validator (which checks if answers match),
this auditor verifies the REASONING QUALITY of the derivation.
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


class DerivationAuditor:
    """
    Audits physics derivations for logical consistency and correctness.

    Uses a DIFFERENT model (GPT-4) than the generator (Claude) to avoid
    shared blind spots and circular reasoning.

    The audit focuses on REASONING QUALITY, not just answer matching:
    - Do steps logically follow from each other?
    - Are there internal contradictions?
    - Are claims properly justified?
    - Is dimensional analysis consistent throughout?
    """

    AUDIT_PROMPT = """You are an expert physics professor performing a rigorous audit of a student's solution.

Your task is to find ERRORS and PROBLEMS in this derivation. Be critical and thorough.

QUESTION:
{query}

PROVIDED ANSWER:
{response_answer}

PROVIDED DERIVATION:
{response_reasoning}

RUBRIC (for context):
{rubric}

---

AUDIT INSTRUCTIONS:

You must examine the derivation step-by-step and identify ALL issues. Look for:

1. **LOGICAL CONTRADICTIONS**: Steps that contradict each other
   - Example: "Step 3 says X is zero, but Step 7 uses X as nonzero"

2. **UNJUSTIFIED CLAIMS**: Statements made without derivation or explanation
   - Example: "Using the result from quantum field theory..." without showing it

3. **MISSING STEPS**: Gaps in the derivation where key steps are skipped
   - Example: Going from equation A to equation B without showing the algebra

4. **DIMENSIONAL ERRORS**: Units that don't match at any step
   - Check dimensions at EACH step, not just the final answer

5. **SIGN ERRORS**: Incorrect signs in equations or boundary conditions

6. **FACTOR ERRORS**: Missing or extra factors of 2, pi, etc.

7. **MATHEMATICAL ERRORS**: Wrong algebra, calculus mistakes, etc.

8. **PHYSICS ERRORS**: Wrong physical principles or their application

9. **INCONSISTENT NOTATION**: Variables used inconsistently

10. **QUESTION VALIDITY**: Is the question itself well-posed and solvable?

---

{json_instruction}

Respond with JSON:
{{
    "audit_passed": true/false,
    "confidence": 0.0-1.0,
    "question_well_posed": {{
        "valid": true/false,
        "issues": ["list of issues with the question itself, if any"]
    }},
    "step_analysis": [
        {{
            "step_number": 1,
            "step_description": "Brief description of what this step does",
            "is_valid": true/false,
            "issues": ["list of issues found in this step"],
            "severity": "critical/major/minor/none"
        }}
    ],
    "logical_contradictions": [
        {{
            "description": "Description of the contradiction",
            "step_a": "First contradicting statement",
            "step_b": "Second contradicting statement",
            "severity": "critical/major"
        }}
    ],
    "dimensional_analysis": {{
        "passed": true/false,
        "issues": ["list of dimensional inconsistencies"]
    }},
    "unjustified_claims": [
        {{
            "claim": "The unjustified claim",
            "location": "Where in the derivation",
            "what_is_missing": "What justification is needed"
        }}
    ],
    "mathematical_errors": [
        {{
            "description": "Description of the error",
            "location": "Where in the derivation",
            "severity": "critical/major/minor"
        }}
    ],
    "physics_errors": [
        {{
            "description": "Description of the physics error",
            "correct_physics": "What should have been done",
            "severity": "critical/major/minor"
        }}
    ],
    "final_answer_correct": {{
        "likely_correct": true/false,
        "confidence": 0.0-1.0,
        "issues": ["issues with the final answer, if any"]
    }},
    "overall_verdict": "PASS/FAIL/NEEDS_REVISION",
    "summary": "Brief summary of the audit findings",
    "critical_issues_count": 0,
    "major_issues_count": 0,
    "minor_issues_count": 0
}}"""

    def __init__(
        self,
        client: OpenRouterClient,
        audit_model: str = "openai/gpt-4o",
        required_passes: int = 2,
    ):
        """
        Initialize the derivation auditor.

        Args:
            client: OpenRouter API client
            audit_model: Model to use for auditing (should be DIFFERENT from generator)
            required_passes: Number of audit passes required (default 2 for reliability)
        """
        self.client = client
        self.audit_model = audit_model
        self.required_passes = required_passes

    async def _single_audit(
        self,
        qa: PhysicsQADataPoint,
        attempt_num: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform a single audit of the derivation.

        Args:
            qa: The QA data point to audit
            attempt_num: Which audit attempt this is (for logging)

        Returns:
            Tuple of (passed, audit_details)
        """
        # Format rubric for context
        rubric_str = ""
        if qa.rubric:
            rubric_str = f"""
Total Points: {qa.rubric.total_points}
Final Answer ({qa.rubric.final_answer.points} pts): {qa.rubric.final_answer.value}
Key Steps:
"""
            for step in qa.rubric.key_steps:
                rubric_str += f"  - {step.step} ({step.points} pts)\n"

        prompt = self.AUDIT_PROMPT.format(
            query=qa.query,
            response_answer=qa.response_answer,
            response_reasoning=qa.response_reasoning,
            rubric=rubric_str,
            json_instruction=JSON_INSTRUCTION,
        )

        try:
            response = await self.client.chat_completion(
                model=self.audit_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistent analysis
                max_tokens=8192,
                response_format={"type": "json_object"},
            )

            content = response["choices"][0]["message"]["content"]
            data = extract_json_from_response(content)

            # Determine if audit passed
            audit_passed = data.get("audit_passed", False)
            verdict = data.get("overall_verdict", "FAIL")
            critical_count = data.get("critical_issues_count", 0)
            major_count = data.get("major_issues_count", 0)

            # Override audit_passed based on critical issues
            if critical_count > 0:
                audit_passed = False
                verdict = "FAIL"
            elif major_count >= 2:
                audit_passed = False
                verdict = "NEEDS_REVISION"

            data["audit_passed"] = audit_passed
            data["overall_verdict"] = verdict

            logger.info(
                f"  Audit {attempt_num}: passed={audit_passed}, "
                f"verdict={verdict}, critical={critical_count}, major={major_count}"
            )

            return audit_passed, data

        except Exception as e:
            logger.error(f"Audit {attempt_num} failed: {e}")
            return False, {
                "error": str(e),
                "audit_passed": False,
                "overall_verdict": "ERROR",
            }

    async def audit(
        self,
        qa: PhysicsQADataPoint,
    ) -> Tuple[bool, int, List[Dict[str, Any]]]:
        """
        Audit a QA pair with multiple passes.

        Args:
            qa: The QA data point to audit

        Returns:
            Tuple of:
            - passed: True if all audits passed
            - pass_count: Number of audits that passed
            - audits: List of all audit details
        """
        logger.info(f"Running derivation audit for QA {qa.id} ({self.required_passes}x)...")

        audits = []
        pass_count = 0

        for i in range(self.required_passes):
            passed, details = await self._single_audit(qa, i + 1)
            audits.append(details)

            if passed:
                pass_count += 1
            else:
                logger.warning(
                    f"QA {qa.id} failed audit {i + 1}/{self.required_passes}. "
                    f"Verdict: {details.get('overall_verdict', 'Unknown')}"
                )

        overall_passed = pass_count == self.required_passes

        if overall_passed:
            logger.info(f"QA {qa.id} PASSED derivation audit ({pass_count}/{self.required_passes})")
        else:
            logger.warning(f"QA {qa.id} FAILED derivation audit ({pass_count}/{self.required_passes})")

        return overall_passed, pass_count, audits

    def extract_feedback_from_audits(
        self,
        audits: List[Dict[str, Any]],
    ) -> str:
        """
        Extract actionable feedback from failed audits for regeneration.

        Args:
            audits: List of audit details

        Returns:
            Feedback string for regeneration
        """
        issues = []

        for i, audit in enumerate(audits):
            if audit.get("audit_passed", False):
                continue  # Skip passed audits

            # Question validity issues
            q_validity = audit.get("question_well_posed", {})
            if not q_validity.get("valid", True):
                for issue in q_validity.get("issues", []):
                    issues.append(f"- Question issue: {issue}")

            # Logical contradictions (most severe)
            for contradiction in audit.get("logical_contradictions", []):
                issues.append(
                    f"- LOGICAL CONTRADICTION: {contradiction.get('description', '')} "
                    f"('{contradiction.get('step_a', '')}' vs '{contradiction.get('step_b', '')}')"
                )

            # Physics errors
            for error in audit.get("physics_errors", []):
                if error.get("severity") in ["critical", "major"]:
                    issues.append(
                        f"- PHYSICS ERROR: {error.get('description', '')}. "
                        f"Correct approach: {error.get('correct_physics', '')}"
                    )

            # Mathematical errors
            for error in audit.get("mathematical_errors", []):
                if error.get("severity") in ["critical", "major"]:
                    issues.append(
                        f"- MATH ERROR at {error.get('location', 'unknown')}: "
                        f"{error.get('description', '')}"
                    )

            # Dimensional analysis
            dim_analysis = audit.get("dimensional_analysis", {})
            if not dim_analysis.get("passed", True):
                for issue in dim_analysis.get("issues", [])[:3]:
                    issues.append(f"- DIMENSIONAL ERROR: {issue}")

            # Unjustified claims
            for claim in audit.get("unjustified_claims", [])[:3]:
                issues.append(
                    f"- UNJUSTIFIED CLAIM: '{claim.get('claim', '')}' - "
                    f"Missing: {claim.get('what_is_missing', '')}"
                )

            # Add summary
            summary = audit.get("summary", "")
            if summary:
                issues.append(f"- SUMMARY: {summary}")

        feedback = "DERIVATION AUDIT FAILED - The reasoning contains errors.\n\n"

        if issues:
            feedback += "CRITICAL ISSUES FOUND:\n"
            # Deduplicate and limit issues
            unique_issues = list(dict.fromkeys(issues))[:10]
            feedback += "\n".join(unique_issues)
            feedback += "\n\n"

        feedback += (
            "You MUST fix these issues:\n"
            "1. Ensure EVERY step logically follows from the previous one\n"
            "2. Remove or fix any contradictory statements\n"
            "3. Show ALL intermediate steps - no hand-waving\n"
            "4. Verify dimensional analysis at EACH step\n"
            "5. Double-check all algebra and calculus\n"
            "6. Ensure the physics principles are correctly applied\n"
            "7. If the question is ill-posed, fix the question itself\n"
        )

        return feedback


async def audit_qa_batch(
    client: OpenRouterClient,
    qa_pairs: List[PhysicsQADataPoint],
    audit_model: str = "openai/gpt-4o",
    required_passes: int = 2,
) -> Tuple[List[PhysicsQADataPoint], List[PhysicsQADataPoint], Dict[str, Any]]:
    """
    Audit a batch of QA pairs.

    Args:
        client: OpenRouter API client
        qa_pairs: List of QA pairs to audit
        audit_model: Model to use for auditing
        required_passes: Number of passes required

    Returns:
        Tuple of (passed_pairs, failed_pairs, summary)
    """
    auditor = DerivationAuditor(
        client=client,
        audit_model=audit_model,
        required_passes=required_passes,
    )

    passed_pairs = []
    failed_pairs = []
    all_results = []

    for i, qa in enumerate(qa_pairs):
        logger.info(f"Auditing QA {i + 1}/{len(qa_pairs)} (ID: {qa.id})...")

        passed, pass_count, audits = await auditor.audit(qa)

        result = {
            "id": qa.id,
            "passed": passed,
            "pass_count": pass_count,
            "required": required_passes,
            "topic": qa.topic.value if qa.topic else None,
            "subtopic": qa.subtopic,
            "verdict": audits[-1].get("overall_verdict", "UNKNOWN") if audits else "ERROR",
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
        "total_audited": total,
        "passed": passed_count,
        "failed": len(failed_pairs),
        "pass_rate": f"{pass_rate:.1%}",
        "required_passes": required_passes,
        "audit_model": audit_model,
        "results": all_results,
    }

    logger.info(
        f"Derivation audit complete: {passed_count}/{total} passed ({pass_rate:.1%})"
    )

    return passed_pairs, failed_pairs, summary
