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
    MIN_ANSWER_LENGTH = 5
    MIN_RUBRIC_CRITERIA = 3
    MAX_RUBRIC_CRITERIA = 10

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
        """Validate the rubric structure."""
        errors = []

        if not isinstance(rubric, dict):
            return ["rubric must be an object"]

        # Check total_points
        total_points = rubric.get("total_points")
        if not isinstance(total_points, int):
            errors.append("rubric.total_points must be an integer")
        elif total_points < 10 or total_points > 100:
            errors.append("rubric.total_points must be between 10 and 100")

        # Check passing_threshold
        threshold = rubric.get("passing_threshold")
        if not isinstance(threshold, int):
            errors.append("rubric.passing_threshold must be an integer")
        elif total_points and (threshold < 1 or threshold > total_points):
            errors.append("rubric.passing_threshold must be between 1 and total_points")

        # Check criteria
        criteria = rubric.get("criteria", [])
        if not isinstance(criteria, list):
            errors.append("rubric.criteria must be an array")
        elif len(criteria) < self.MIN_RUBRIC_CRITERIA:
            errors.append(f"rubric must have at least {self.MIN_RUBRIC_CRITERIA} criteria")
        elif len(criteria) > self.MAX_RUBRIC_CRITERIA:
            errors.append(f"rubric must have at most {self.MAX_RUBRIC_CRITERIA} criteria")
        else:
            # Validate each criterion
            points_sum = 0
            for i, criterion in enumerate(criteria):
                if not isinstance(criterion, dict):
                    errors.append(f"rubric.criteria[{i}] must be an object")
                    continue

                if not criterion.get("criterion"):
                    errors.append(f"rubric.criteria[{i}].criterion is required")

                max_pts = criterion.get("max_points")
                if not isinstance(max_pts, int) or max_pts < 1:
                    errors.append(f"rubric.criteria[{i}].max_points must be a positive integer")
                else:
                    points_sum += max_pts

                if not criterion.get("description"):
                    errors.append(f"rubric.criteria[{i}].description is required")

            # Check points sum
            if isinstance(total_points, int) and points_sum != total_points:
                errors.append(
                    f"Sum of criteria max_points ({points_sum}) must equal total_points ({total_points})"
                )

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
