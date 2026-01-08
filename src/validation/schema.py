"""Schema validation for QA pairs."""

import logging
from typing import Tuple, List, Dict, Any

from pydantic import ValidationError

from ..models.schemas import PhysicsQADataPoint, Rubric

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates QA pairs against the required schema."""

    # Minimum requirements for graduate-level content
    MIN_QUERY_WORDS = 15
    MIN_REASONING_WORDS = 50
    MIN_ANSWER_LENGTH = 1  # Physics answers can be single symbols like "π", "0", "∞"

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a QA data point against the schema.

        Args:
            data: Dictionary with QA data fields

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields exist
        required_fields = ["query", "response_answer", "response_reasoning", "rubric", "response_images"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        # Validate response_images is empty (hard constraint)
        if data.get("response_images") != []:
            errors.append("response_images must be an empty array")

        # Validate query
        query = data.get("query", "")
        if not isinstance(query, str):
            errors.append("query must be a string")
        elif len(query.strip()) < 50:
            errors.append(f"query too short (min 50 chars, got {len(query.strip())})")
        elif len(query.split()) < self.MIN_QUERY_WORDS:
            errors.append(f"query should have at least {self.MIN_QUERY_WORDS} words")

        # Validate response_answer
        answer = data.get("response_answer", "")
        if not isinstance(answer, str):
            errors.append("response_answer must be a string")
        elif len(answer.strip()) < self.MIN_ANSWER_LENGTH:
            errors.append(f"response_answer too short (min {self.MIN_ANSWER_LENGTH} chars)")

        # Validate response_reasoning
        reasoning = data.get("response_reasoning", "")
        if not isinstance(reasoning, str):
            errors.append("response_reasoning must be a string")
        elif len(reasoning.strip()) < 100:
            errors.append("response_reasoning too short (min 100 chars)")
        elif len(reasoning.split()) < self.MIN_REASONING_WORDS:
            errors.append(f"response_reasoning should have at least {self.MIN_REASONING_WORDS} words")

        # Validate rubric structure
        rubric = data.get("rubric", {})
        rubric_errors = self._validate_rubric(rubric)
        errors.extend(rubric_errors)

        # Check for image references in text (they shouldn't reference images)
        image_keywords = ["figure", "diagram", "see image", "as shown", "refer to", "illustrated"]
        combined_text = f"{query} {reasoning}".lower()
        for keyword in image_keywords:
            if keyword in combined_text:
                # Only flag if it seems to reference an external image
                if any(phrase in combined_text for phrase in [
                    f"see {keyword}",
                    f"refer to {keyword}",
                    f"in the {keyword}",
                    f"shown in {keyword}",
                ]):
                    errors.append(f"Text may reference an image (found '{keyword}'). Questions must be self-contained without images.")
                    break

        return len(errors) == 0, errors

    def _validate_rubric(self, rubric: Any) -> List[str]:
        """
        Validate the rubric structure.

        Rubric format (10-point scale):
        {
            "total_points": 10,
            "final_answer": {
                "value": "The correct answer in LaTeX",
                "points": 3,
                "tolerance": "equivalent symbolic forms accepted",
                "common_errors": ["list of common errors"]
            },
            "key_steps": [
                {"step": "Description of key step", "points": 1-4}
            ],
            "partial_credit_rules": ["list of rules"],
            "automatic_zero": ["conditions for automatic zero"]
        }
        """
        errors = []

        if not isinstance(rubric, dict):
            return ["rubric must be an object"]

        # Validate final_answer
        final_answer = rubric.get("final_answer")
        if not final_answer:
            errors.append("rubric.final_answer is required")
        elif not isinstance(final_answer, dict):
            errors.append("rubric.final_answer must be an object")
        else:
            if not final_answer.get("value"):
                errors.append("rubric.final_answer.value is required")
            elif not isinstance(final_answer.get("value"), str):
                errors.append("rubric.final_answer.value must be a string")

        # Validate key_steps
        key_steps = rubric.get("key_steps")
        if not key_steps:
            errors.append("rubric.key_steps is required")
        elif not isinstance(key_steps, list):
            errors.append("rubric.key_steps must be an array")
        elif len(key_steps) < 3:
            errors.append("rubric.key_steps must have at least 3 steps")
        elif len(key_steps) > 7:
            errors.append("rubric.key_steps must have at most 7 steps")
        else:
            total_points = 0
            for i, step in enumerate(key_steps):
                if not isinstance(step, dict):
                    errors.append(f"rubric.key_steps[{i}] must be an object")
                else:
                    if not step.get("step"):
                        errors.append(f"rubric.key_steps[{i}].step is required")
                    points = step.get("points")
                    if points is None:
                        errors.append(f"rubric.key_steps[{i}].points is required")
                    elif not isinstance(points, int) or points < 1 or points > 4:
                        errors.append(f"rubric.key_steps[{i}].points must be an integer between 1 and 4")
                    else:
                        total_points += points
            if total_points < 5 or total_points > 9:
                errors.append(f"rubric.key_steps points must sum to 5-9, got {total_points}")

        return errors

    def validate_pydantic(self, data: Dict[str, Any]) -> Tuple[bool, List[str], PhysicsQADataPoint | None]:
        """
        Validate using Pydantic model for stricter validation.

        Args:
            data: Dictionary with QA data fields

        Returns:
            Tuple of (is_valid, list_of_errors, parsed_model_or_none)
        """
        try:
            qa = PhysicsQADataPoint(**data)
            return True, [], qa
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            return False, errors, None

    def validate_complete(self, data: Dict[str, Any]) -> Tuple[bool, List[str], PhysicsQADataPoint | None]:
        """
        Run both basic and Pydantic validation.

        Args:
            data: Dictionary with QA data fields

        Returns:
            Tuple of (is_valid, list_of_errors, parsed_model_or_none)
        """
        # Basic validation first
        basic_valid, basic_errors = self.validate(data)
        if not basic_valid:
            return False, basic_errors, None

        # Then Pydantic validation
        return self.validate_pydantic(data)
