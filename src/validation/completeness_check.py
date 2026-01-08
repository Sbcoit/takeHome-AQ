"""Problem completeness validation for physics QA pairs.

This module ensures physics problems are WELL-POSED:
1. All variables are defined
2. Boundary/initial conditions are specified
3. No missing information needed to solve
4. Physical context is complete
5. Assumptions are stated

This catches the class of errors where the problem itself is ambiguous
or impossible to solve without additional information.
"""

import asyncio
import logging
from typing import Tuple, Dict, Any, List

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint
from ..utils import extract_json_from_response
from ..prompts import JSON_INSTRUCTION

logger = logging.getLogger(__name__)


class CompletenessValidator:
    """
    Validates that physics problems are complete and well-posed.

    Catches issues like:
    - Undefined variables/constants
    - Missing boundary conditions
    - Ambiguous problem statements
    - Underspecified scenarios
    - Missing physical context
    """

    COMPLETENESS_CHECK_PROMPT = """You are an expert physics professor reviewing a problem for COMPLETENESS.

Your job is to determine if this problem can be solved UNAMBIGUOUSLY with the information provided.

QUESTION:
{query}

---

Check for these COMPLETENESS issues:

1. **UNDEFINED VARIABLES**: Are all symbols/variables in the problem defined?
   - Every symbol (m, r, v, E, etc.) must be clearly defined
   - If using standard notation, it should be unambiguous (e.g., c for speed of light)
   - Check: Can a student identify every quantity in the problem?

2. **MISSING INITIAL/BOUNDARY CONDITIONS**: Are all necessary conditions specified?
   - For ODEs: initial position, velocity, etc.
   - For PDEs: boundary conditions (Dirichlet, Neumann, periodic, etc.)
   - For quantum: normalization, asymptotic behavior
   - Check: Is the solution uniquely determined?

3. **AMBIGUOUS PHYSICAL CONTEXT**: Is the physical situation clear?
   - Reference frame specified if relevant
   - Dimensionality clear (1D, 2D, 3D)
   - Approximations stated (non-relativistic, small angle, etc.)
   - Check: Could reasonable physicists interpret this differently?

4. **MISSING CONSTANTS/PARAMETERS**: Are all needed values provided or standard?
   - Physical constants (c, ℏ, G, etc.) - OK to leave unspecified if standard
   - Problem-specific parameters must be given or solvable
   - Check: Can numerical answers be obtained if requested?

5. **IMPLICIT ASSUMPTIONS**: Are necessary assumptions stated?
   - Idealizations (frictionless, massless spring, point particle)
   - Approximation regimes (weak field, low velocity)
   - Check: Would a student need to guess at assumptions?

6. **SOLVABILITY**: Can this problem actually be solved?
   - Is the mathematics tractable?
   - Is there a unique answer?
   - Check: Does the problem have a well-defined solution?

---

{json_instruction}

Respond with JSON:
{{
    "undefined_variables": {{
        "found": ["list of undefined symbols"],
        "passed": true/false,
        "explanation": "which variables need definition"
    }},
    "boundary_conditions": {{
        "required": ["list of conditions needed"],
        "provided": ["list of conditions given"],
        "missing": ["list of missing conditions"],
        "passed": true/false,
        "explanation": "what boundary/initial conditions are missing"
    }},
    "physical_context": {{
        "ambiguities": ["list of ambiguous aspects"],
        "passed": true/false,
        "explanation": "what context is unclear"
    }},
    "missing_parameters": {{
        "found": ["list of missing parameters"],
        "passed": true/false,
        "explanation": "what parameters are needed"
    }},
    "implicit_assumptions": {{
        "unstated": ["list of unstated assumptions"],
        "severity": "none/minor/major",
        "passed": true/false,
        "explanation": "what assumptions should be explicit"
    }},
    "solvability": {{
        "is_solvable": true/false,
        "has_unique_solution": true/false,
        "explanation": "assessment of solvability"
    }},
    "overall_complete": true/false,
    "critical_issues": ["list of critical completeness issues"],
    "suggested_additions": ["list of what should be added to make problem complete"],
    "confidence": 0.0-1.0,
    "summary": "Brief summary of completeness assessment"
}}"""

    def __init__(
        self,
        client: OpenRouterClient,
        check_model: str = "anthropic/claude-opus-4",
    ):
        """
        Initialize the completeness validator.

        Args:
            client: OpenRouter API client
            check_model: Model to use for checking
        """
        self.client = client
        self.check_model = check_model

    async def validate(
        self,
        qa: PhysicsQADataPoint,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Check if a physics problem is complete and well-posed.

        Args:
            qa: The QA data point to validate

        Returns:
            Tuple of:
            - passed: True if problem is complete
            - details: Dict with detailed check results
            - feedback: String feedback for regeneration if incomplete
        """
        logger.info(f"Running completeness check on QA {qa.id}...")

        prompt = self.COMPLETENESS_CHECK_PROMPT.format(
            query=qa.query,
            json_instruction=JSON_INSTRUCTION,
        )

        try:
            response = await self.client.chat_completion(
                model=self.check_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            content = response["choices"][0]["message"]["content"]
            data = extract_json_from_response(content)

            # Extract results
            undefined_vars = data.get("undefined_variables", {})
            boundary_conds = data.get("boundary_conditions", {})
            context = data.get("physical_context", {})
            missing_params = data.get("missing_parameters", {})
            assumptions = data.get("implicit_assumptions", {})
            solvability = data.get("solvability", {})

            overall_complete = data.get("overall_complete", False)
            critical_issues = data.get("critical_issues", [])
            confidence = data.get("confidence", 0.5)

            # Log results
            vars_passed = undefined_vars.get("passed", True)
            bc_passed = boundary_conds.get("passed", True)
            context_passed = context.get("passed", True)
            params_passed = missing_params.get("passed", True)
            assume_passed = assumptions.get("passed", True)
            solvable = solvability.get("is_solvable", True)

            logger.info(
                f"Completeness check for {qa.id}: "
                f"vars={vars_passed}, boundary={bc_passed}, context={context_passed}, "
                f"params={params_passed}, assumptions={assume_passed}, solvable={solvable}. "
                f"Overall: {'COMPLETE' if overall_complete else 'INCOMPLETE'}"
            )

            # Generate feedback if incomplete
            feedback = ""
            if not overall_complete:
                feedback = self._generate_feedback(data, critical_issues)

            return overall_complete, data, feedback

        except Exception as e:
            logger.error(f"Completeness check failed with error: {e}")
            # Be lenient on errors - don't block pipeline
            return True, {"error": str(e)}, ""

    def _generate_feedback(
        self,
        results: Dict[str, Any],
        critical_issues: List[str],
    ) -> str:
        """
        Generate actionable feedback for making the problem complete.

        Args:
            results: Full completeness check results
            critical_issues: List of critical issues found

        Returns:
            Feedback string for the generator
        """
        feedback_parts = ["PROBLEM COMPLETENESS CHECK FAILED - The problem is not well-posed.\n"]

        # Undefined variables
        undefined_vars = results.get("undefined_variables", {})
        if not undefined_vars.get("passed", True):
            found = undefined_vars.get("found", [])
            if found:
                feedback_parts.append(
                    f"UNDEFINED VARIABLES:\n"
                    f"  The following symbols are used but not defined: {', '.join(found)}\n"
                    f"  - {undefined_vars.get('explanation', '')}\n"
                    f"  YOU MUST define every variable in the problem statement.\n\n"
                )

        # Missing boundary conditions
        boundary_conds = results.get("boundary_conditions", {})
        if not boundary_conds.get("passed", True):
            missing = boundary_conds.get("missing", [])
            if missing:
                feedback_parts.append(
                    f"MISSING BOUNDARY/INITIAL CONDITIONS:\n"
                    f"  The following conditions are needed but not provided: {', '.join(missing)}\n"
                    f"  - {boundary_conds.get('explanation', '')}\n"
                    f"  YOU MUST specify all necessary initial/boundary conditions.\n\n"
                )

        # Ambiguous context
        context = results.get("physical_context", {})
        if not context.get("passed", True):
            ambiguities = context.get("ambiguities", [])
            if ambiguities:
                feedback_parts.append(
                    f"AMBIGUOUS PHYSICAL CONTEXT:\n"
                    f"  The following aspects are unclear: {', '.join(ambiguities)}\n"
                    f"  - {context.get('explanation', '')}\n"
                    f"  YOU MUST make the physical situation unambiguous.\n\n"
                )

        # Missing parameters
        missing_params = results.get("missing_parameters", {})
        if not missing_params.get("passed", True):
            found = missing_params.get("found", [])
            if found:
                feedback_parts.append(
                    f"MISSING PARAMETERS:\n"
                    f"  The following values are needed: {', '.join(found)}\n"
                    f"  - {missing_params.get('explanation', '')}\n"
                    f"  YOU MUST provide or clearly define all parameters.\n\n"
                )

        # Unstated assumptions
        assumptions = results.get("implicit_assumptions", {})
        if not assumptions.get("passed", True):
            unstated = assumptions.get("unstated", [])
            severity = assumptions.get("severity", "minor")
            if unstated and severity in ["major"]:
                feedback_parts.append(
                    f"UNSTATED ASSUMPTIONS ({severity.upper()}):\n"
                    f"  The following assumptions should be explicit: {', '.join(unstated)}\n"
                    f"  - {assumptions.get('explanation', '')}\n"
                    f"  YOU MUST state all important assumptions.\n\n"
                )

        # Solvability
        solvability = results.get("solvability", {})
        if not solvability.get("is_solvable", True):
            feedback_parts.append(
                f"SOLVABILITY ISSUE:\n"
                f"  - {solvability.get('explanation', 'Problem may not be solvable')}\n"
                f"  The problem must have a well-defined, unique solution.\n\n"
            )

        # Critical issues
        if critical_issues:
            feedback_parts.append("CRITICAL ISSUES TO FIX:\n")
            for issue in critical_issues[:5]:
                feedback_parts.append(f"  - {issue}\n")

        # Suggested additions
        suggested = results.get("suggested_additions", [])
        if suggested:
            feedback_parts.append("\nTO MAKE THIS PROBLEM COMPLETE, ADD:\n")
            for suggestion in suggested[:5]:
                feedback_parts.append(f"  - {suggestion}\n")

        feedback_parts.append(
            "\nA well-posed physics problem must:\n"
            "1. Define ALL variables and symbols used\n"
            "2. Specify ALL boundary/initial conditions\n"
            "3. Have an unambiguous physical interpretation\n"
            "4. State all non-obvious assumptions\n"
            "5. Have a unique, determinable solution\n"
        )

        return "".join(feedback_parts)


class QuickCompletenessChecker:
    """
    Fast, heuristic-based completeness checker (no API calls).

    Uses pattern matching to catch obvious completeness issues:
    - Undefined standard symbols
    - Missing "given" or "assume" statements
    - Ambiguous pronouns ("it", "this", "the particle")
    """

    # Symbols that MUST be defined if used
    MUST_DEFINE = {
        'm', 'M', 'r', 'R', 'v', 'V', 'a', 'A', 'F', 'E', 'U', 'T', 'L',
        'k', 'K', 'q', 'Q', 'B', 'I', 'P', 'n', 'N', 'omega', 'theta',
        'phi', 'psi', 'lambda', 'rho', 'sigma', 'tau'
    }

    # Standard constants that don't need definition
    STANDARD_CONSTANTS = {
        'c', 'G', 'hbar', 'h', 'e', 'epsilon_0', 'mu_0', 'k_B',
        'm_e', 'm_p', 'pi', 'alpha'
    }

    # Ambiguous phrases that suggest incomplete problems
    AMBIGUOUS_PHRASES = [
        r'\bthe particle\b',
        r'\bthe object\b',
        r'\bthe system\b',
        r'\bit moves\b',
        r'\bthis force\b',
        r'\bthat potential\b',
        r'\bsome value\b',
        r'\ba certain\b',
    ]

    def check(self, query: str) -> Tuple[bool, List[str]]:
        """
        Quick heuristic check for completeness.

        Args:
            query: The problem statement

        Returns:
            Tuple of (likely_complete, warnings)
        """
        import re

        warnings = []
        query_lower = query.lower()

        # Check for ambiguous phrases
        for pattern in self.AMBIGUOUS_PHRASES:
            if re.search(pattern, query_lower):
                warnings.append(f"Potentially ambiguous: '{re.search(pattern, query_lower).group()}'")

        # Check for defined vs undefined symbols (simplified)
        # Look for "let X be" or "where X is" or "X =" patterns
        defined_pattern = r'(?:let|where|given|assume)\s+(\w+)\s+(?:be|is|=|denotes)'
        defined_matches = re.findall(defined_pattern, query_lower)

        # This is a simplified check - the full AI check is more thorough
        if len(defined_matches) == 0 and any(s in query for s in self.MUST_DEFINE):
            warnings.append("Problem uses symbols but may not define them all")

        # Check for boundary conditions keywords
        bc_keywords = ['initial', 'boundary', 'at t=0', 'at r=0', 'at x=0',
                       'as r→∞', 'at infinity', 'normalized']
        has_bc_mention = any(kw in query_lower for kw in bc_keywords)

        # If problem involves differential equations, should have BC
        de_keywords = ['differential equation', 'solve for', 'd/dt', 'd/dx',
                       'schrodinger', 'wave equation', 'laplace', 'poisson']
        needs_bc = any(kw in query_lower for kw in de_keywords)

        if needs_bc and not has_bc_mention:
            warnings.append("Problem may involve DEs but doesn't mention boundary conditions")

        likely_complete = len(warnings) == 0
        return likely_complete, warnings
