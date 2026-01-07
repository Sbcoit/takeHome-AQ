"""QA generation using LLMs."""

import json
import logging
import re
from typing import Optional, Dict, Any

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint, Rubric, PhysicsTopic
from .topics import TopicContext


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

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates physics QA pairs using LLMs."""

    SYSTEM_PROMPT = """You are an expert physics professor who writes PhD qualifying exam problems. Your problems appear on exams at top research universities and are calibrated for first-year PhD students who have completed graduate coursework in classical mechanics, electromagnetism, quantum mechanics, and statistical mechanics.

Your problems:
- Require genuine mastery of graduate-level physics (not undergraduate review)
- Test whether students can apply theory to concrete physical situations
- Have clear, unambiguous setups with all necessary information provided
- Reward correct physical reasoning and mathematical execution
- Are CHALLENGING - they should NOT be solvable by pattern-matching or simple formula application

All mathematical expressions use plain text (e.g., "x^2", "sqrt(x)", "integral from 0 to infinity", "partial derivative of f with respect to x")."""

    GENERATION_PROMPT = """Create a PhD qualifying exam question about {topic_context}.

QUESTION TYPE: Pure derivation/calculation problem
- The question must be solvable through mathematical derivation from first principles
- NO questions about physical effects, corrections, or perturbations where the answer depends on interpretation
- NO questions about "what happens when X" or "estimate the effect of Y"
- YES to: "Derive an expression for X", "Calculate Y given Z", "Show that X equals Y"

QUESTION FORMAT:
- A single cohesive question written as flowing prose (NOT labeled parts like (a), (b), (c))
- The question can involve multiple reasoning steps, but asks for ONE final answer

ANSWER FORMAT (CRITICAL):
- The final answer MUST be symbolic, a ratio, or a simple integer/fraction
- Express answers in terms of given variables (m, ω, ℏ, k, etc.) NOT as decimal numbers
- Good: "4/3", "2π", "mω²R²/2", "n(n+1)ℏ²", "E₀/4"
- Bad: "1.333", "6.28", "0.0023 J", "82 nanobarns"
- The answer must follow UNIQUELY from the problem setup - no judgment calls

DIFFICULTY CALIBRATION:
Make it hard through mathematical complexity, NOT through subtle physics effects.

What makes it hard:
- Requires careful setup of boundary conditions or constraints
- Involves non-trivial integration, series expansion, or algebraic manipulation
- Multiple steps where errors compound
- Requires recognizing the right mathematical technique to apply

What keeps it solvable:
- All information needed is explicitly stated in the problem
- Standard physics (mechanics, E&M, QM, stat mech) - not cutting-edge topics
- The derivation follows a clear logical path
- You can verify your answer is correct with 100% confidence

GOOD QUESTION PATTERNS:
- "Derive the expression for X in terms of Y and Z"
- "A system has Hamiltonian H = ... Find the eigenvalues"
- "Given potential V(r) = ..., calculate the bound state energies"
- "Show that the ratio X/Y equals ..."
- "For a particle in ... find the probability that ..."

AVOID:
- Questions about thermal corrections, perturbative effects, or small parameters
- Questions where the answer depends on which terms you keep/drop
- "Estimate" or "approximate" questions
- Questions about real physical systems (use idealized setups)
- Questions that reference a figure, diagram, or image
- Standard textbook problems (even with different numbers)

Respond with JSON:
{{
    "query": "The complete problem statement with all necessary information.",
    "response_answer": "The symbolic/ratio answer (e.g., '4/3', 'mω²R²/2', 'n+1'). NO decimals.",
    "response_reasoning": "Complete worked solution showing the physics reasoning, key equations, mathematical steps, and final calculation.",
    "rubric": {{
        "correct_physics": "What physical principles/laws must be correctly identified and applied (e.g., 'Must use conservation of energy and angular momentum')",
        "correct_answer": "The expected answer and any equivalent forms (e.g., 'mω²R²/2 or equivalent expression for kinetic energy')",
        "sound_reasoning": "What mathematical/logical steps are required (e.g., 'Must set up the Lagrangian, derive equations of motion, and solve for the period')"
    }},
    "response_images": []
}}"""

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

                # Build the rubric
                rubric_data = data.get("rubric", {})
                rubric = Rubric(
                    correct_physics=rubric_data.get("correct_physics", ""),
                    correct_answer=rubric_data.get("correct_answer", ""),
                    sound_reasoning=rubric_data.get("sound_reasoning", ""),
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
2. The answer must be SYMBOLIC (e.g., "mω²R²/2", "4/3", "exp(-x²)") - NO decimal numbers
3. The question must be self-contained with all information needed to solve it
4. NO references to figures, diagrams, or images

Please generate an improved version that addresses the feedback while following all constraints above. Respond with JSON only.

{{
    "query": "Improved question...",
    "response_answer": "Improved answer...",
    "response_reasoning": "Improved reasoning...",
    "rubric": {{
        "total_points": 100,
        "criteria": [
            {{"criterion": "...", "max_points": 25, "description": "..."}},
            {{"criterion": "...", "max_points": 25, "description": "..."}},
            {{"criterion": "...", "max_points": 25, "description": "..."}},
            {{"criterion": "...", "max_points": 15, "description": "..."}},
            {{"criterion": "...", "max_points": 10, "description": "..."}}
        ],
        "passing_threshold": 70
    }},
    "response_images": []
}}"""

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
                data = json.loads(content)

                rubric_data = data.get("rubric", {})
                criteria = [RubricCriterion(**c) for c in rubric_data.get("criteria", [])]

                # Validate we have criteria before proceeding
                if not criteria:
                    raise ValueError("Empty criteria list in rubric")

                total_from_criteria = sum(c.max_points for c in criteria)

                # Validate total_points is reasonable
                if total_from_criteria < 10:
                    raise ValueError(f"Total points {total_from_criteria} is too low (minimum 10)")

                rubric = Rubric(
                    total_points=total_from_criteria,
                    criteria=criteria,
                    passing_threshold=rubric_data.get("passing_threshold", 60),
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
