"""QA generation using LLMs."""

import json
import logging
from typing import Optional, Dict, Any

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint, Rubric, FinalAnswer, KeyStep
from ..utils import extract_json_from_response
from .topics import TopicContext

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates physics QA pairs using LLMs."""

    SYSTEM_PROMPT = r"""You are an expert physics professor who writes PhD qualifying exam problems. Your problems appear on exams at top research universities and are calibrated for first-year PhD students.

YOUR SPECIAL SKILL: Writing problems that DISCRIMINATE between careful and careless solvers.
- Weaker AI models (Qwen, smaller LLMs) should get these WRONG due to common pitfalls
- Strong AI models (GPT-4, Claude Opus, Gemini Pro) should get them RIGHT with careful reasoning
- The problems test UNDERSTANDING, not just formula recall

Your problems feature:
- TRAPS that catch pattern-matching and formula-plugging
- Subtle sign conventions where mistakes propagate
- Boundary conditions that are easy to forget
- Coupled systems requiring careful algebra
- Factors of 2, π, or combinatorial factors that careless solvers miss
- Non-standard setups where textbook formulas don't directly apply

Your problems are FAIR:
- All information needed is explicitly provided
- The correct answer follows uniquely from careful analysis
- A human PhD student who thinks carefully CAN solve it
- The solution is verifiable (correct dimensions, limiting cases work)

CRITICAL - LATEX NOTATION REQUIRED:
All mathematical expressions MUST use LaTeX notation with escaped backslashes for JSON:
- Variables: "m", "\\omega", "\\hbar", "\\vec{r}"
- Fractions: "\\frac{a}{b}"
- Powers: "x^2", "e^{-x}"
- Roots: "\\sqrt{x}", "\\sqrt[3]{x}"
- Greek letters: "\\alpha", "\\beta", "\\omega", "\\pi" (NOT Unicode α, β, ω, π)
- Subscripts: "x_1", "T_0", "E_n" (NOT Unicode ₁, ₀, ₙ)
- Superscripts: "x^2", "\\hbar^2" (NOT Unicode ²)
- Operators: "\\partial", "\\nabla", "\\int", "\\sum"

NEVER use Unicode symbols (α, β, ₁, ², etc.) - ALWAYS use LaTeX (\\alpha, \\beta, _1, ^2, etc.)"""

    GENERATION_PROMPT = """Create a PhD qualifying exam question about {topic_context}.

CRITICAL DIFFICULTY REQUIREMENT:
This question must be HARD ENOUGH that a mid-tier AI model (like Qwen) will frequently get it WRONG,
but SOLVABLE ENOUGH that top frontier models (GPT-4, Claude, Gemini) can solve it correctly.

The sweet spot: Questions that PUNISH common mistakes and shortcuts, but reward careful reasoning.

WHAT TRIPS UP WEAKER MODELS (use these techniques):
1. **Subtle sign conventions** - Problems where getting the sign wrong changes everything
2. **Non-obvious coordinate choices** - Where the "obvious" choice leads to harder math
3. **Coupled equations** - Systems where you can't solve variables independently
4. **Boundary condition traps** - Where forgetting one condition gives a plausible but wrong answer
5. **Integration by parts pitfalls** - Where naive integration misses surface terms
6. **Degeneracy and symmetry** - Where you must carefully count states or handle special cases
7. **Non-commuting limits** - Where the order of taking limits matters
8. **Dimensional analysis traps** - Answers that look dimensionally correct but are wrong by factors of 2, \\pi, etc.

QUESTION DESIGN PRINCIPLES:
- The WRONG approach should give a plausible-looking but incorrect answer
- The RIGHT approach requires recognizing a subtlety that shortcuts miss
- Multiple derivation paths exist, but careless ones have traps
- The problem should look standard at first glance but have a twist

QUESTION TYPE: Pure derivation/calculation problem
- Solvable through mathematical derivation from first principles
- NO ambiguous "estimate" or "approximate" questions
- YES to: "Derive", "Calculate", "Show that", "Find the expression for"

QUESTION FORMAT:
- Single cohesive question (NOT labeled parts (a), (b), (c))
- Multiple reasoning steps, ONE final answer

ANSWER FORMAT:
- SYMBOLIC answer (not decimal): \\frac{{{{4}}}}{{{{3}}}}, 2\\pi, n(n+1)\\hbar^2
- Use LaTeX: \\frac{{{{a}}}}{{{{b}}}}, \\alpha, \\hbar (NEVER Unicode α, ℏ)
- Answer must follow UNIQUELY from the setup

EXAMPLE DIFFICULTY PATTERNS (pick one style):

Style A - Sign/Convention Trap:
"A charged particle moves in crossed E and B fields..." (where naive velocity addition gives wrong sign)

Style B - Coupled System:
"Two coupled oscillators with masses m_1, m_2 and springs k_1, k_2, k_{{12}}..." (must diagonalize properly)

Style C - Boundary Condition Trap:
"A quantum particle in a finite well..." (must match both value AND derivative at boundaries)

Style D - Integration Trap:
"Calculate the magnetic field of a current loop at an arbitrary point..." (where symmetry arguments fail off-axis)

Style E - Counting/Degeneracy:
"Find the partition function for N indistinguishable particles..." (where naive counting overcounts)

MANDATORY SELF-VERIFICATION:
1. Solve the problem yourself step-by-step
2. Identify where a careless solver would make mistakes
3. Verify your answer has correct dimensions
4. Check limiting cases
5. Confirm the answer is UNIQUE (no ambiguity)

LATEX IN JSON: Use double backslashes: "\\\\frac{{{{a}}}}{{{{b}}}}", "\\\\alpha", "\\\\hbar"

RUBRIC FORMAT (10-point scale):
- Final answer: 3 points (can't pass by guessing alone)
- Key steps: 7 points total (conceptual + calculational steps)
- Pass requires: 7+ points AND correct final answer

Respond with JSON:
{{{{
    "query": "Problem statement with all necessary information, using LaTeX for math...",
    "response_answer": "**\\\\frac{{{{symbolic}}}}{{{{answer}}}}**",
    "response_reasoning": "## Solution\\n\\n### Key Insight\\n[What makes this problem tricky]\\n\\n### Setup\\n[Define variables and equations]\\n\\n### Derivation\\n1. [Step with potential pitfall noted]\\n2. [Careful handling of subtlety]...\\n\\n### Common Mistakes\\n- [What a careless solver would do wrong]\\n\\n### Final Answer\\n**answer**",
    "rubric": {{{{
        "total_points": 10,
        "final_answer": {{{{
            "value": "\\\\frac{{{{symbolic}}}}{{{{answer}}}}",
            "points": 3,
            "tolerance": "equivalent symbolic forms accepted (e.g., 0.5x = x/2 = \\\\frac{{{{x}}}}{{{{2}}}})",
            "common_errors": [
                "Missing factor of 2 from [reason] (1 point partial credit)",
                "Wrong sign due to [reason] (0 points)"
            ]
        }}}},
        "key_steps": [
            {{{{"step": "Identify the correct physical principle: [specific law/theorem]", "points": 1}}}},
            {{{{"step": "Set up the governing equation: [equation]", "points": 2}}}},
            {{{{"step": "Apply boundary condition / constraint: [specific condition]", "points": 1}}}},
            {{{{"step": "Perform the key calculation: [integration/algebra/etc]", "points": 2}}}},
            {{{{"step": "Simplify to final form, checking dimensions", "points": 1}}}}
        ],
        "partial_credit_rules": [
            "Correct method but arithmetic error: deduct 1-2 points",
            "Correct setup but wrong algebra: max 5/10",
            "Only final answer, no work shown: max 3/10"
        ],
        "automatic_zero": [
            "Uses completely wrong method (e.g., classical approach for quantum problem)",
            "Answer is dimensionally incorrect",
            "Misidentifies the fundamental physics (e.g., treats as equilibrium when dynamic)"
        ]
    }}}},
    "response_images": []
}}}}"""

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        judge_model: str = "anthropic/claude-sonnet-4",
    ):
        """
        Initialize the QA generator.

        Args:
            client: OpenRouter API client
            model: Model identifier for generation
            judge_model: Model to use for judging (used by validators, not generator)
        """
        self.client = client
        self.model = model
        self.judge_model = judge_model

    async def generate(
        self,
        topic_context: TopicContext,
        temperature: float = 0.8,
        max_retries: int = 3,
    ) -> PhysicsQADataPoint:
        """
        Generate a single QA pair.

        Args:
            topic_context: The topic context to generate for
            temperature: Sampling temperature (higher = more diverse)
            max_retries: Number of retries for failed JSON parsing

        Returns:
            PhysicsQADataPoint with the generated QA pair
        """
        prompt = self.GENERATION_PROMPT.format(topic_context=topic_context.to_prompt_string())

        for attempt in range(max_retries):
            try:
                response = await self.client.chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=32768,
                    response_format={"type": "json_object"},
                )

                content = response["choices"][0]["message"]["content"]
                data = extract_json_from_response(content)

                # Build the rubric with new point-based format
                rubric_data = data.get("rubric", {})

                # Parse final_answer
                final_answer_data = rubric_data.get("final_answer", {})
                final_answer = FinalAnswer(
                    value=final_answer_data.get("value", data.get("response_answer", "")),
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

                rubric = Rubric(
                    total_points=rubric_data.get("total_points", 10),
                    final_answer=final_answer,
                    key_steps=key_steps,
                    partial_credit_rules=rubric_data.get("partial_credit_rules", []),
                    automatic_zero=rubric_data.get("automatic_zero", []),
                )

                # Create the data point
                qa = PhysicsQADataPoint(
                    query=data["query"],
                    response_answer=data["response_answer"],
                    response_reasoning=data["response_reasoning"],
                    rubric=rubric,
                    response_images=[],  # Always empty
                    topic=topic_context.topic,
                    subtopic=topic_context.subtopic,
                )

                logger.info(f"Generated QA for {topic_context.subtopic}")
                return qa

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to parse generation response after {max_retries} attempts")

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
        Regenerate a QA pair with specific feedback.

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

        if is_difficulty_feedback:
            improvement_prompt = f"""The following physics question was TOO EASY and needs to be made SIGNIFICANTLY HARDER.

FEEDBACK: {feedback}

ORIGINAL QUESTION (too easy - DO NOT just tweak it):
{original.query}

CRITICAL: You must generate a COMPLETELY DIFFERENT and MUCH HARDER question on the same topic.

HOW TO MAKE IT GENUINELY HARDER (you MUST do at least 3 of these):
1. **Combine multiple physics concepts** - e.g., thermodynamics + statistical mechanics, or QM + E&M
2. **Add non-trivial constraints** - unusual boundary conditions, coupled systems, non-standard geometries
3. **Require multi-step derivations** - where intermediate results feed into subsequent calculations
4. **Use perturbation theory or approximation methods** - where students must identify the small parameter
5. **Include subtle limiting cases** - where naive approaches give wrong answers
6. **Require coordinate transformations** - non-Cartesian coordinates, rotating frames, etc.
7. **Add time-dependence or dynamics** - instead of static/equilibrium problems
8. **Use less common but important physics** - Lagrangian/Hamiltonian methods, Green's functions, etc.

WHAT MAKES A QUESTION PhD-LEVEL HARD:
- Cannot be solved by plugging into a standard formula
- Requires recognizing which approximations are valid
- Has multiple steps where errors compound
- Tests deep understanding, not just memorization
- Would challenge a first-year physics PhD student

CRITICAL CONSTRAINTS:
1. Single cohesive question - NO labeled parts (a), (b), (c)
2. SYMBOLIC answer with LaTeX: "**\\\\frac{{{{m\\\\omega^2 R^2}}}}{{{{2}}}}**"
3. Self-contained with all needed information
4. NO image references

LATEX: Use "\\\\frac{{{{a}}}}{{{{b}}}}", "\\\\alpha", "\\\\hbar" - NEVER Unicode symbols.

Respond with JSON:
{{{{
    "query": "A COMPLETELY NEW, MUCH HARDER question on {original.subtopic}...",
    "response_answer": "**\\\\frac{{{{symbolic}}}}{{{{answer}}}}**",
    "response_reasoning": "## Solution\\n\\n### Approach\\n[Why this approach]\\n\\n### Derivation\\n1. [Complex step 1]\\n2. [Complex step 2]...\\n\\n### Final Answer\\n**answer**",
    "rubric": {{{{
        "total_points": 10,
        "final_answer": {{{{
            "value": "\\\\frac{{{{symbolic}}}}{{{{answer}}}}",
            "points": 3,
            "tolerance": "equivalent symbolic forms accepted",
            "common_errors": ["[common wrong answer] due to [reason] (partial credit)"]
        }}}},
        "key_steps": [
            {{{{"step": "Key conceptual step 1", "points": 1}}}},
            {{{{"step": "Key calculational step 2", "points": 2}}}},
            {{{{"step": "Key step 3", "points": 2}}}},
            {{{{"step": "Key step 4", "points": 2}}}}
        ],
        "partial_credit_rules": ["Correct method but error: deduct 1-2 points"],
        "automatic_zero": ["Wrong method entirely", "Dimensionally incorrect"]
    }}}},
    "response_images": []
}}}}"""
        else:
            improvement_prompt = f"""The following physics question needs improvement based on this feedback:

FEEDBACK: {feedback}

ORIGINAL QUESTION:
{original.query}

ORIGINAL ANSWER:
{original.response_answer}

ORIGINAL REASONING:
{original.response_reasoning}

CRITICAL CONSTRAINTS (must follow):
1. The question must be a SINGLE cohesive question - NO labeled parts like (a), (b), (c)
2. The answer must be SYMBOLIC with markdown bold using LaTeX: "**\\frac{{{{m\\omega^2 R^2}}}}{{{{2}}}}**", "**\\frac{{{{4}}}}{{{{3}}}}**"
3. The question must be self-contained with all information needed to solve it
4. NO references to figures, diagrams, or images
5. Use MARKDOWN formatting throughout for readability

LATEX NOTATION REQUIRED (CRITICAL):
- In JSON, use double backslashes: "\\\\frac{{{{a}}}}{{{{b}}}}", "\\\\alpha", "\\\\hbar"
- Greek letters: "\\\\alpha", "\\\\beta", "\\\\omega", "\\\\pi" - NEVER Unicode (α, β, ω, π)
- Subscripts: "x_1", "T_0", "E_n" - NEVER Unicode (x₁, T₀, Eₙ)
- Superscripts: "x^2", "\\\\hbar^2" - NEVER Unicode (x², ℏ²)
- Fractions: "\\\\frac{{{{numerator}}}}{{{{denominator}}}}"

MANDATORY SELF-VERIFICATION (Do this BEFORE outputting):
1. Write the COMPLETE governing equation(s) - don't simplify prematurely
2. Include ALL terms and factors - check standard references if unsure
3. Verify UNITS/DIMENSIONS of your final answer
4. Check limiting cases (does the answer make physical sense?)
5. If the original answer was wrong, derive from scratch - don't just tweak it

COMMON PHYSICS ERRORS TO FIX:
- Missing concentration/mole fraction factors in transport equations
- Dropped factors of 2, \\pi, or dimensionless constants
- Sign errors in gradients (\\nabla T vs dT/dx)
- Using approximate formulas where exact ones are needed

Please generate an improved version that addresses the feedback while following all constraints above. Respond with JSON only.

{{{{
    "query": "Improved question using LaTeX notation for all math...",
    "response_answer": "**\\\\frac{{{{symbolic}}}}{{{{answer}}}}** (with equivalent forms noted)",
    "response_reasoning": "## Solution\\n\\n### Given\\n- m: mass\\n- \\\\omega: angular frequency\\n\\n### Approach\\nMethod description.\\n\\n### Derivation\\n1. Step one: **\\\\frac{{{{p^2}}}}{{{{2m}}}}**\\n2. Step two...\\n\\n### Verification\\n- Dimensional analysis: [units]\\n- Limiting case: [check]\\n\\n### Final Answer\\n**\\\\frac{{{{symbolic}}}}{{{{answer}}}}**",
    "rubric": {{{{
        "total_points": 10,
        "final_answer": {{{{
            "value": "\\\\frac{{{{symbolic}}}}{{{{answer}}}}",
            "points": 3,
            "tolerance": "equivalent symbolic forms accepted",
            "common_errors": ["[common wrong answer] due to [reason] (partial credit)"]
        }}}},
        "key_steps": [
            {{{{"step": "Key conceptual step 1", "points": 1}}}},
            {{{{"step": "Key calculational step 2", "points": 2}}}},
            {{{{"step": "Key step 3", "points": 2}}}},
            {{{{"step": "Key step 4", "points": 2}}}}
        ],
        "partial_credit_rules": ["Correct method but error: deduct 1-2 points"],
        "automatic_zero": ["Wrong method entirely", "Dimensionally incorrect"]
    }}}},
    "response_images": []
}}}}"""

        for attempt in range(max_retries):
            try:
                response = await self.client.chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": improvement_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=32768,
                    response_format={"type": "json_object"},
                )

                content = response["choices"][0]["message"]["content"]
                data = extract_json_from_response(content)

                # Build the rubric with new point-based format
                rubric_data = data.get("rubric", {})

                # Parse final_answer
                final_answer_data = rubric_data.get("final_answer", {})
                final_answer = FinalAnswer(
                    value=final_answer_data.get("value", data.get("response_answer", "")),
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

                rubric = Rubric(
                    total_points=rubric_data.get("total_points", 10),
                    final_answer=final_answer,
                    key_steps=key_steps,
                    partial_credit_rules=rubric_data.get("partial_credit_rules", []),
                    automatic_zero=rubric_data.get("automatic_zero", []),
                )

                qa = PhysicsQADataPoint(
                    query=data["query"],
                    response_answer=data["response_answer"],
                    response_reasoning=data["response_reasoning"],
                    rubric=rubric,
                    response_images=[],
                    topic=original.topic,
                    subtopic=original.subtopic,
                )

                logger.info(f"Regenerated QA for {original.subtopic}")
                return qa

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on regeneration attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to parse regeneration response after {max_retries} attempts")

            except KeyError as e:
                logger.warning(f"Missing field in regeneration response on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Missing required field: {e}")

            except Exception as e:
                logger.warning(f"Regeneration error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError("Unexpected end of retry loop")
