"""QA generation using LLMs with two-step generation for accuracy.

Two-step generation process:
1. Generate ONLY the problem (no solution) - focuses on creating good questions
2. Solve the problem independently - gets a clean derivation without bias

This separation prevents the model from creating problems where the derivation
doesn't actually support the stated answer.
"""

import json
import logging
from typing import Optional, Dict, Any

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint, Rubric, FinalAnswer, KeyStep
from ..utils import extract_json_from_response
from .topics import TopicContext

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates physics QA pairs using a two-step process for accuracy.

    Step 1: Generate ONLY the problem statement (no solution)
    Step 2: Solve the problem independently to get a clean derivation

    This separation prevents the model from creating problems where the stated
    answer doesn't actually follow from the derivation.
    """

    # System prompt for problem generation (Step 1)
    PROBLEM_SYSTEM_PROMPT = r"""You are an expert physics professor who writes PhD qualifying exam problems. Your problems appear on exams at top research universities and are calibrated for first-year PhD students.

Your problems are:
- Graduate-level difficulty requiring multi-step derivations
- Self-contained with ALL information needed to solve
- Clear and unambiguous - ONE correct answer
- Solvable through first-principles physics

IMPORTANT: You are ONLY creating the problem statement. Do NOT solve it.

LATEX NOTATION REQUIRED:
All mathematical expressions MUST use LaTeX notation:
- Greek letters: "\\alpha", "\\beta", "\\omega", "\\pi" (NOT Unicode α, β, ω, π)
- Subscripts: "x_1", "T_0", "E_n"
- Superscripts: "x^2", "\\hbar^2"
- Fractions: "\\frac{a}{b}"

NEVER use Unicode symbols - ALWAYS use LaTeX."""

    # System prompt for solving (Step 2)
    SOLVER_SYSTEM_PROMPT = r"""You are an expert physics professor solving a graduate-level physics problem.

Your solution must be:
- RIGOROUS: Show ALL steps, no hand-waving
- DIMENSIONALLY VERIFIED: Track units at EVERY step (this is MANDATORY)
- COMPLETE: Include all intermediate algebra
- CORRECT: Double-check your final answer

MANDATORY DIMENSIONAL ANALYSIS PROTOCOL:
You MUST track dimensions throughout your derivation. For EACH equation:
1. Write the dimensional formula [M^a L^b T^c ...] for BOTH sides
2. Verify they match BEFORE proceeding to the next step
3. If dimensions don't match, you have an ERROR - fix it before continuing

Example of proper dimensional tracking:
"F = ma → [M L T^{-2}] = [M][L T^{-2}] ✓"
"E = ½mv² → [M L² T^{-2}] = [M][L T^{-1}]² = [M L² T^{-2}] ✓"

CRITICAL VERIFICATION CHECKLIST (do this BEFORE finalizing):
1. Does your final answer have correct units/dimensions? (SHOW THE CHECK)
2. Does it behave correctly in limiting cases (e.g., as m→0, as T→∞)?
3. Is the sign correct (check physical intuition)?
4. Did you include all factors (check for missing 2, π, etc.)?

LATEX NOTATION: Use "\\frac{a}{b}", "\\alpha", "\\hbar" etc."""

    # Step 1: Problem generation prompt (NO solution)
    PROBLEM_GENERATION_PROMPT = """Create a PhD qualifying exam question about {topic_context}.

IMPORTANT: Generate ONLY the problem statement. Do NOT solve it or provide the answer.

PROBLEM REQUIREMENTS:
1. Graduate-level difficulty requiring multi-step derivation
2. Self-contained with ALL information needed to solve
3. Clear and unambiguous - should have ONE correct symbolic answer
4. Pure derivation/calculation (use "Derive", "Calculate", "Find the expression for")
5. Single cohesive question - NO parts (a), (b), (c)

GOOD PROBLEM CHARACTERISTICS:
- Requires understanding of core physics principles
- Cannot be solved by plugging into a single formula
- Has intermediate steps where errors can occur
- Tests deep understanding, not just memorization

LATEX: Use "\\\\alpha", "\\\\frac{{{{a}}}}{{{{b}}}}", etc. - NEVER Unicode symbols.

Respond with JSON:
{{{{
    "query": "Complete problem statement with all needed information...",
    "expected_difficulty": "graduate/advanced_graduate",
    "key_concepts": ["concept1", "concept2"],
    "expected_answer_type": "symbolic expression / ratio / energy / etc."
}}}}"""

    # Step 2: Solve prompt (given a problem, derive the solution)
    SOLVE_PROMPT = """Solve this physics problem completely and rigorously.

PROBLEM:
{query}

REQUIREMENTS:
1. Show ALL steps - no hand-waving or "it can be shown that"
2. Track dimensions at EACH step (write [M^a L^b T^c] explicitly)
3. Verify limiting cases where applicable
4. Double-check algebra and signs
5. State your final answer clearly

==== MANDATORY SANITY CHECKS (do ALL of these BEFORE finalizing) ====

1. DIMENSIONAL ANALYSIS (catches ~80% of errors):
   - Write the expected dimensions of the answer (e.g., [Energy] = [M L² T⁻²])
   - Verify your answer has these exact dimensions
   - If dimensions don't match, your answer is WRONG - fix it

2. LIMITING CASES:
   - Check at least 2 limiting cases (e.g., as parameter → 0, as parameter → ∞)
   - Does the answer reduce to known results in these limits?
   - Example: As perturbation → 0, must recover unperturbed case

3. SIGN CHECK:
   - Is the sign physically reasonable?
   - Energies should be real, probabilities must be positive (0-1)
   - Attractive forces should give negative potential energy
   - Repulsive potentials should give positive scattering lengths

4. ORDER OF MAGNITUDE:
   - Is the numerical scale plausible?
   - Nothing physical can be smaller than Planck length (~10⁻³⁵ m)
   - Nothing faster than light, nothing colder than 0 K

5. INTERNAL CONSISTENCY:
   - Do intermediate steps agree with the final answer?
   - Re-derive a key step to verify no algebraic errors

======================================================================

LATEX: Use "\\\\frac{{{{a}}}}{{{{b}}}}", "\\\\alpha", etc.

Respond with JSON:
{{{{
    "response_answer": "**\\\\frac{{{{symbolic}}}}{{{{answer}}}}**",
    "response_reasoning": "## Solution\\n\\n### Setup\\n[Define variables and their dimensions]\\n\\n### Derivation\\n1. [Step with dimensional check]\\n2. [Step with dimensional check]...\\n\\n### Sanity Checks\\n#### Dimensional Analysis\\n[Show expected vs actual dimensions]\\n#### Limiting Cases\\n[Check at least 2 limits]\\n#### Sign Check\\n[Verify sign is physical]\\n#### Internal Consistency\\n[Verify intermediate steps match final]\\n\\n### Final Answer\\n**answer**",
    "dimensional_check": {{{{
        "expected_dimensions": "[M^a L^b T^c ...]",
        "actual_dimensions": "[M^a L^b T^c ...]",
        "match": true/false,
        "explanation": "Detailed dimensional analysis"
    }}}},
    "limiting_cases_checked": [
        {{{{"case": "parameter → 0", "result": "reduces to known X", "physical": true}}}},
        {{{{"case": "parameter → ∞", "result": "approaches Y", "physical": true}}}}
    ],
    "sign_check": {{{{"correct": true/false, "explanation": "why sign is physical"}}}},
    "order_of_magnitude_check": {{{{"plausible": true/false, "explanation": "scale is reasonable because..."}}}},
    "internal_consistency": {{{{"consistent": true/false, "explanation": "intermediate steps verified"}}}},
    "confidence": 0.0-1.0
}}}}"""

    # Combined prompt for rubric generation (after we have Q and A)
    RUBRIC_PROMPT = """Generate a grading rubric for this physics problem and solution.

PROBLEM:
{query}

SOLUTION:
{response_reasoning}

FINAL ANSWER:
{response_answer}

Create a 10-point rubric where:
- Final answer: 3 points
- Key derivation steps: 7 points total

LATEX: Use "\\\\frac{{{{a}}}}{{{{b}}}}", "\\\\alpha", etc.

Respond with JSON:
{{{{
    "rubric": {{{{
        "total_points": 10,
        "final_answer": {{{{
            "value": "{response_answer}",
            "points": 3,
            "tolerance": "equivalent symbolic forms accepted",
            "common_errors": ["error1 (partial credit)", "error2 (0 points)"]
        }}}},
        "key_steps": [
            {{{{"step": "step description", "points": 2}}}},
            {{{{"step": "step description", "points": 2}}}},
            {{{{"step": "step description", "points": 2}}}},
            {{{{"step": "step description", "points": 1}}}}
        ],
        "partial_credit_rules": ["rule1", "rule2"],
        "automatic_zero": ["condition1", "condition2"]
    }}}}
}}}}"""

    # Legacy one-shot prompt (kept for backwards compatibility / fallback)
    GENERATION_PROMPT = PROBLEM_GENERATION_PROMPT  # Simplified default

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        judge_model: str = "anthropic/claude-sonnet-4",
        solver_model: Optional[str] = None,
    ):
        """
        Initialize the QA generator with two-step generation.

        Args:
            client: OpenRouter API client
            model: Model for problem generation (Step 1)
            judge_model: Model to use for judging (used by validators)
            solver_model: Model for solving problems (Step 2). If None, uses same as model.
                         For best results, use a different model to avoid shared blind spots.
        """
        self.client = client
        self.model = model
        self.judge_model = judge_model
        # Use a different model for solving if specified, otherwise use same model
        self.solver_model = solver_model or model

    async def _generate_problem_only(
        self,
        topic_context: TopicContext,
        temperature: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Step 1: Generate ONLY the problem statement (no solution).

        Args:
            topic_context: The topic context to generate for
            temperature: Sampling temperature

        Returns:
            Dict with query and metadata
        """
        prompt = self.PROBLEM_GENERATION_PROMPT.format(
            topic_context=topic_context.to_prompt_string()
        )

        response = await self.client.chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.PROBLEM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=4096,  # Problems are short
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]
        data = extract_json_from_response(content)

        logger.info(f"Generated problem for {topic_context.subtopic}")
        return data

    async def _solve_problem(
        self,
        query: str,
        temperature: float = 0.3,  # Lower temp for more consistent solving
    ) -> Dict[str, Any]:
        """
        Step 2: Solve the problem independently.

        Args:
            query: The problem statement to solve
            temperature: Sampling temperature (lower = more consistent)

        Returns:
            Dict with answer and reasoning
        """
        prompt = self.SOLVE_PROMPT.format(query=query)

        response = await self.client.chat_completion(
            model=self.solver_model,
            messages=[
                {"role": "system", "content": self.SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=16000,  # Solutions can be long
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]
        data = extract_json_from_response(content)

        logger.info(f"Solved problem (confidence: {data.get('confidence', 'N/A')})")
        return data

    async def _generate_rubric(
        self,
        query: str,
        response_answer: str,
        response_reasoning: str,
    ) -> Rubric:
        """
        Step 3: Generate rubric for grading.

        Args:
            query: The problem statement
            response_answer: The final answer
            response_reasoning: The solution derivation

        Returns:
            Rubric object
        """
        prompt = self.RUBRIC_PROMPT.format(
            query=query,
            response_answer=response_answer,
            response_reasoning=response_reasoning,
        )

        response = await self.client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]
        data = extract_json_from_response(content)
        rubric_data = data.get("rubric", {})

        # Parse final_answer
        final_answer_data = rubric_data.get("final_answer", {})
        final_answer = FinalAnswer(
            value=final_answer_data.get("value", response_answer),
            points=final_answer_data.get("points", 3),
            tolerance=final_answer_data.get("tolerance", "equivalent symbolic forms accepted"),
            common_errors=final_answer_data.get("common_errors", []),
        )

        # Parse key_steps
        key_steps_data = rubric_data.get("key_steps", [])
        key_steps = [
            KeyStep(step=s.get("step", ""), points=s.get("points", 1))
            for s in key_steps_data
        ] if key_steps_data else [
            KeyStep(step="Identify correct physical principle", points=1),
            KeyStep(step="Set up governing equations", points=2),
            KeyStep(step="Apply constraints/boundary conditions", points=2),
            KeyStep(step="Perform calculation and simplify", points=2),
        ]

        return Rubric(
            total_points=rubric_data.get("total_points", 10),
            final_answer=final_answer,
            key_steps=key_steps,
            partial_credit_rules=rubric_data.get("partial_credit_rules", []),
            automatic_zero=rubric_data.get("automatic_zero", []),
        )

    async def generate(
        self,
        topic_context: TopicContext,
        temperature: float = 0.8,
        max_retries: int = 3,
    ) -> PhysicsQADataPoint:
        """
        Generate a QA pair using two-step process for accuracy.

        Step 1: Generate problem only (no solution)
        Step 2: Solve problem independently
        Step 3: Generate rubric

        Args:
            topic_context: The topic context to generate for
            temperature: Sampling temperature (higher = more diverse)
            max_retries: Number of retries for failed attempts

        Returns:
            PhysicsQADataPoint with the generated QA pair
        """
        for attempt in range(max_retries):
            try:
                # Step 1: Generate problem only
                logger.info(f"Step 1: Generating problem for {topic_context.subtopic}...")
                problem_data = await self._generate_problem_only(topic_context, temperature)
                query = problem_data["query"]

                # Step 2: Solve the problem independently
                logger.info(f"Step 2: Solving problem independently...")
                solution_data = await self._solve_problem(query, temperature=0.3)

                response_answer = solution_data.get("response_answer", "")
                response_reasoning = solution_data.get("response_reasoning", "")

                # Check solver confidence - if low, retry with new problem
                # Claude should be ~100% confident solving its own problems
                # Low confidence indicates ambiguous/ill-posed problem or likely wrong answer
                confidence = solution_data.get("confidence", 0.5)
                if confidence < 0.95:
                    logger.warning(
                        f"Solver confidence too low ({confidence:.2f} < 0.95), regenerating problem..."
                    )
                    continue

                # Step 3: Generate rubric
                logger.info(f"Step 3: Generating rubric...")
                rubric = await self._generate_rubric(query, response_answer, response_reasoning)

                # Create the data point
                qa = PhysicsQADataPoint(
                    query=query,
                    response_answer=response_answer,
                    response_reasoning=response_reasoning,
                    rubric=rubric,
                    response_images=[],
                    topic=topic_context.topic,
                    subtopic=topic_context.subtopic,
                )

                logger.info(f"Generated QA for {topic_context.subtopic} (2-step process)")
                return qa

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to parse response after {max_retries} attempts")

            except KeyError as e:
                logger.warning(f"Missing field in response on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Missing required field: {e}")

            except Exception as e:
                logger.warning(f"Generation error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError(f"Failed to generate QA after {max_retries} attempts")

    async def generate_batch(
        self,
        topic_contexts: list[TopicContext],
        temperature: float = 0.8,
    ) -> list[PhysicsQADataPoint]:
        """
        Generate multiple QA pairs (sequentially to avoid overwhelming API).

        Args:
            topic_contexts: List of topic contexts to generate for
            temperature: Sampling temperature

        Returns:
            List of generated PhysicsQADataPoint objects
        """
        results = []

        for i, context in enumerate(topic_contexts):
            try:
                qa = await self.generate(context, temperature)
                results.append(qa)
                logger.info(f"Generated {i + 1}/{len(topic_contexts)}")
            except Exception as e:
                logger.error(f"Failed to generate for {context.subtopic}: {e}")
                # Continue with other generations

        return results

    async def regenerate_with_feedback(
        self,
        original: PhysicsQADataPoint,
        feedback: str,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> PhysicsQADataPoint:
        """
        Regenerate a QA pair with specific feedback using two-step process.

        Uses the same two-step approach as generate():
        Step 1: Regenerate problem based on feedback
        Step 2: Solve the new/improved problem independently

        Args:
            original: The original QA pair to improve
            feedback: Specific feedback about what to improve
            temperature: Sampling temperature
            max_retries: Number of retries for failed generation/parsing

        Returns:
            Improved PhysicsQADataPoint
        """
        # Detect if this is a difficulty-related regeneration
        is_difficulty_feedback = "TOO EASY" in feedback.upper() or "HARDER" in feedback.upper()
        is_answer_feedback = "INCORRECT" in feedback.upper() or "WRONG" in feedback.upper() or "ANSWER" in feedback.upper()

        if is_difficulty_feedback:
            # Need a harder problem - regenerate problem only, then solve
            problem_prompt = f"""The following physics question was TOO EASY and needs to be made SIGNIFICANTLY HARDER.

FEEDBACK: {feedback}

ORIGINAL QUESTION (too easy - DO NOT just tweak it):
{original.query}

Generate a COMPLETELY DIFFERENT and MUCH HARDER question on the same topic ({original.subtopic}).

IMPORTANT: Generate ONLY the problem statement. Do NOT solve it.

HOW TO MAKE IT GENUINELY HARDER:
1. Combine multiple physics concepts
2. Add non-trivial constraints or unusual boundary conditions
3. Require multi-step derivations where intermediate results matter
4. Use perturbation theory or approximation methods
5. Require coordinate transformations or rotating frames

CONSTRAINTS:
- Single cohesive question (NO parts a, b, c)
- Self-contained with all needed information
- Clear and unambiguous
- LATEX notation: "\\\\alpha", "\\\\frac{{{{a}}}}{{{{b}}}}"

Respond with JSON:
{{{{
    "query": "A COMPLETELY NEW, MUCH HARDER question...",
    "expected_difficulty": "advanced_graduate",
    "key_concepts": ["concept1", "concept2"],
    "expected_answer_type": "symbolic expression"
}}}}"""
        elif is_answer_feedback:
            # Answer was wrong - keep the problem, just re-solve it
            logger.info("Answer feedback detected - re-solving existing problem...")
            return await self._regenerate_solution_only(original, feedback, temperature, max_retries)
        else:
            # General improvement - regenerate problem based on feedback, then solve
            problem_prompt = f"""The following physics question needs improvement based on this feedback:

FEEDBACK: {feedback}

ORIGINAL QUESTION:
{original.query}

Generate an IMPROVED version of this question that addresses the feedback.

IMPORTANT: Generate ONLY the problem statement. Do NOT solve it.

CONSTRAINTS:
- Single cohesive question (NO parts a, b, c)
- Self-contained with all needed information
- Clear and unambiguous - ONE correct answer
- LATEX notation: "\\\\alpha", "\\\\frac{{{{a}}}}{{{{b}}}}"

Respond with JSON:
{{{{
    "query": "Improved problem statement...",
    "expected_difficulty": "graduate",
    "key_concepts": ["concept1", "concept2"],
    "expected_answer_type": "symbolic expression"
}}}}"""

        # Two-step regeneration
        for attempt in range(max_retries):
            try:
                # Step 1: Regenerate problem
                logger.info(f"Step 1: Regenerating problem based on feedback...")
                response = await self.client.chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.PROBLEM_SYSTEM_PROMPT},
                        {"role": "user", "content": problem_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
                content = response["choices"][0]["message"]["content"]
                problem_data = extract_json_from_response(content)
                query = problem_data["query"]

                # Step 2: Solve the problem independently
                logger.info(f"Step 2: Solving regenerated problem...")
                solution_data = await self._solve_problem(query, temperature=0.3)

                response_answer = solution_data.get("response_answer", "")
                response_reasoning = solution_data.get("response_reasoning", "")

                # Check solver confidence - require high confidence for regenerated problems too
                confidence = solution_data.get("confidence", 0.5)
                if confidence < 0.95:
                    logger.warning(f"Solver confidence too low ({confidence:.2f} < 0.95), retrying...")
                    continue

                # Step 3: Generate rubric
                logger.info(f"Step 3: Generating rubric...")
                rubric = await self._generate_rubric(query, response_answer, response_reasoning)

                qa = PhysicsQADataPoint(
                    query=query,
                    response_answer=response_answer,
                    response_reasoning=response_reasoning,
                    rubric=rubric,
                    response_images=[],
                    topic=original.topic,
                    subtopic=original.subtopic,
                )

                logger.info(f"Regenerated QA for {original.subtopic} (2-step process)")
                return qa

            except Exception as e:
                logger.warning(f"Regeneration error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError(f"Failed to regenerate QA after {max_retries} attempts")

    async def _regenerate_solution_only(
        self,
        original: PhysicsQADataPoint,
        feedback: str,
        temperature: float = 0.5,
        max_retries: int = 3,
    ) -> PhysicsQADataPoint:
        """
        Re-solve an existing problem when only the answer was wrong.

        Args:
            original: The original QA pair (problem is fine, answer is wrong)
            feedback: Feedback about what was wrong
            temperature: Sampling temperature
            max_retries: Number of retries

        Returns:
            QA pair with same problem but new solution
        """
        solve_prompt = f"""Solve this physics problem. The previous solution was INCORRECT.

PROBLEM:
{original.query}

PREVIOUS (WRONG) ANSWER:
{original.response_answer}

FEEDBACK ON WHY IT WAS WRONG:
{feedback}

You must solve this problem from FIRST PRINCIPLES. Do NOT just tweak the old answer.

REQUIREMENTS:
1. Show ALL steps - no hand-waving
2. Track dimensions at EACH step (write [M^a L^b T^c] explicitly)
3. Verify limiting cases
4. Double-check algebra and signs
5. State your final answer clearly

==== MANDATORY SANITY CHECKS (do ALL of these BEFORE finalizing) ====

1. DIMENSIONAL ANALYSIS (catches ~80% of errors):
   - Write the expected dimensions of the answer
   - Verify your answer has these exact dimensions
   - If dimensions don't match, your answer is WRONG

2. LIMITING CASES:
   - Check at least 2 limiting cases
   - Does the answer reduce to known results in these limits?

3. SIGN CHECK:
   - Is the sign physically reasonable?
   - Energies should be real, probabilities must be positive (0-1)

4. ORDER OF MAGNITUDE:
   - Is the numerical scale plausible?
   - Nothing smaller than Planck length, nothing faster than light

5. INTERNAL CONSISTENCY:
   - Do intermediate steps agree with the final answer?

======================================================================

LATEX: Use "\\\\frac{{{{a}}}}{{{{b}}}}", "\\\\alpha", etc.

Respond with JSON:
{{{{
    "response_answer": "**\\\\frac{{{{symbolic}}}}{{{{answer}}}}**",
    "response_reasoning": "## Solution\\n\\n### Setup\\n[Define variables and dimensions]\\n\\n### Derivation\\n[Show ALL steps with dimensional checks]\\n\\n### Sanity Checks\\n[All 5 checks]\\n\\n### Final Answer\\n**answer**",
    "dimensional_check": {{{{
        "expected_dimensions": "[M^a L^b T^c ...]",
        "actual_dimensions": "[M^a L^b T^c ...]",
        "match": true/false,
        "explanation": "Detailed dimensional analysis"
    }}}},
    "limiting_cases_checked": [
        {{{{"case": "parameter → 0", "result": "reduces to X", "physical": true}}}},
        {{{{"case": "parameter → ∞", "result": "approaches Y", "physical": true}}}}
    ],
    "sign_check": {{{{"correct": true/false, "explanation": "why sign is physical"}}}},
    "order_of_magnitude_check": {{{{"plausible": true/false, "explanation": "scale is reasonable"}}}},
    "internal_consistency": {{{{"consistent": true/false, "explanation": "steps verified"}}}},
    "confidence": 0.0-1.0
}}}}"""

        for attempt in range(max_retries):
            try:
                response = await self.client.chat_completion(
                    model=self.solver_model,
                    messages=[
                        {"role": "system", "content": self.SOLVER_SYSTEM_PROMPT},
                        {"role": "user", "content": solve_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=16000,
                    response_format={"type": "json_object"},
                )

                content = response["choices"][0]["message"]["content"]
                solution_data = extract_json_from_response(content)

                response_answer = solution_data.get("response_answer", "")
                response_reasoning = solution_data.get("response_reasoning", "")

                # Generate new rubric for updated solution
                rubric = await self._generate_rubric(original.query, response_answer, response_reasoning)

                qa = PhysicsQADataPoint(
                    query=original.query,  # Keep original problem
                    response_answer=response_answer,
                    response_reasoning=response_reasoning,
                    rubric=rubric,
                    response_images=[],
                    topic=original.topic,
                    subtopic=original.subtopic,
                )

                logger.info(f"Re-solved problem for {original.subtopic}")
                return qa

            except Exception as e:
                logger.warning(f"Re-solve error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError(f"Failed to re-solve problem after {max_retries} attempts")
