"""QA generation using LLMs."""

import json
import logging
from typing import Optional, Dict, Any

from ..api.client import OpenRouterClient
from ..models.schemas import PhysicsQADataPoint, Rubric, RubricCriterion, PhysicsTopic
from .topics import TopicContext

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates physics QA pairs using LLMs."""

    SYSTEM_PROMPT = """You are an expert physics professor who writes PhD qualifying exam problems. Your problems appear on exams at top research universities and are calibrated for first-year PhD students who have completed graduate coursework in classical mechanics, electromagnetism, quantum mechanics, and statistical mechanics.

Your problems:
- Require genuine mastery of graduate-level physics (not undergraduate review)
- Test whether students can apply theory to concrete physical situations
- Have clear, unambiguous setups with all necessary information provided
- Reward correct physical reasoning and mathematical execution

All mathematical expressions use plain text (e.g., "x^2", "sqrt(x)", "integral from 0 to infinity", "partial derivative of f with respect to x")."""

    GENERATION_PROMPT = """Create a PhD qualifying exam question about {topic_context}.

The problem should:
- Be appropriate for a first-year physics PhD student
- Present a specific physical scenario (not a generic textbook exercise)
- Require graduate-level analysis and problem-solving
- Have a definite numerical or analytical answer
- Ensure that it is humanly possible to answer the question

Respond with JSON:
{{
    "query": "The complete problem statement with all necessary information.",
    "response_answer": "The final answer with numerical value/expression and units.",
    "response_reasoning": "Complete worked solution showing the physics reasoning, key equations, mathematical steps, and final calculation.",
    "rubric": {{
        "total_points": 100,
        "criteria": [
            {{
                "criterion": "Physical concepts and approach",
                "max_points": 25,
                "description": "Correctly identifies the relevant physics and chooses an appropriate solution method."
            }},
            {{
                "criterion": "Mathematical formulation",
                "max_points": 25,
                "description": "Sets up the correct equations with appropriate approximations where needed."
            }},
            {{
                "criterion": "Solution execution",
                "max_points": 25,
                "description": "Carries out the mathematical derivation correctly."
            }},
            {{
                "criterion": "Final result",
                "max_points": 15,
                "description": "Arrives at the correct answer with proper units."
            }},
            {{
                "criterion": "Physical insight",
                "max_points": 10,
                "description": "Demonstrates understanding of the physics behind the mathematics."
            }}
        ],
        "passing_threshold": 70
    }},
    "response_images": []
}}"""

    def __init__(self, client: OpenRouterClient, model: str):
        """
        Initialize the QA generator.

        Args:
            client: OpenRouter API client
            model: Model identifier for generation
        """
        self.client = client
        self.model = model

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
            max_retries: Number of retries for failed generation/parsing

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
                    max_tokens=16384,
                    response_format={"type": "json_object"},
                )

                content = response["choices"][0]["message"]["content"]
                data = json.loads(content)

                # Build the rubric
                rubric_data = data.get("rubric", {})
                criteria = [
                    RubricCriterion(**c) for c in rubric_data.get("criteria", [])
                ]

                # Ensure criteria sum to total_points
                total_from_criteria = sum(c.max_points for c in criteria)
                rubric = Rubric(
                    total_points=total_from_criteria,  # Use actual sum
                    criteria=criteria,
                    passing_threshold=rubric_data.get("passing_threshold", 60),
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

        raise RuntimeError("Unexpected end of retry loop")

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

Please generate an improved version that addresses the feedback while maintaining graduate-level difficulty. Respond with the same JSON format as before.

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
                    max_tokens=16384,
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
