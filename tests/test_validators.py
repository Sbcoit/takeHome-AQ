"""Tests for validation components."""

import pytest

from src.validation.schema import SchemaValidator


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    @pytest.fixture
    def validator(self):
        """Fixture for schema validator."""
        return SchemaValidator()

    @pytest.fixture
    def valid_data(self):
        """Fixture for valid QA data."""
        return {
            "query": "Consider a quantum harmonic oscillator with angular frequency omega. " * 5,
            "response_answer": "E_n = hbar * omega * (n + 1/2)",
            "response_reasoning": "We start with the time-independent Schrodinger equation. " * 20,
            "rubric": {
                "final_answer": {
                    "value": "E_n = hbar * omega * (n + 1/2)",
                    "points": 3,
                    "tolerance": "equivalent symbolic forms accepted",
                    "common_errors": ["Missing 1/2 term", "Wrong coefficient"]
                },
                "key_steps": [
                    {"step": "Set up the Schrodinger equation for harmonic potential", "points": 2},
                    {"step": "Apply appropriate boundary conditions", "points": 2},
                    {"step": "Solve for energy eigenvalues", "points": 3}
                ],
                "partial_credit_rules": ["Correct method but arithmetic error: deduct 1-2 points"],
                "automatic_zero": ["Uses completely wrong method for the problem type"]
            },
            "response_images": [],
        }

    def test_valid_data_passes(self, validator, valid_data):
        """Test that valid data passes validation."""
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_query_fails(self, validator, valid_data):
        """Test that missing query fails."""
        del valid_data["query"]
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is False
        assert any("query" in e.lower() for e in errors)

    def test_non_empty_images_fails(self, validator, valid_data):
        """Test that non-empty images fails."""
        valid_data["response_images"] = ["image.png"]
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is False
        assert any("images" in e.lower() for e in errors)

    def test_short_query_fails(self, validator, valid_data):
        """Test that short query fails."""
        valid_data["query"] = "Short question?"
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is False

    def test_short_reasoning_fails(self, validator, valid_data):
        """Test that short reasoning fails."""
        valid_data["response_reasoning"] = "Short."
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is False

    def test_invalid_rubric_structure_fails(self, validator, valid_data):
        """Test that invalid rubric structure fails."""
        valid_data["rubric"] = "not a dict"
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is False

    def test_missing_rubric_field_fails(self, validator, valid_data):
        """Test that missing rubric field fails."""
        del valid_data["rubric"]["final_answer"]
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is False

    def test_empty_rubric_field_fails(self, validator, valid_data):
        """Test that empty rubric field fails."""
        valid_data["rubric"]["final_answer"]["value"] = ""
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is False

    def test_pydantic_validation(self, validator, valid_data):
        """Test Pydantic validation."""
        is_valid, errors, qa = validator.validate_pydantic(valid_data)
        assert is_valid is True
        assert qa is not None
        assert qa.response_images == []

    def test_complete_validation(self, validator, valid_data):
        """Test complete validation pipeline."""
        is_valid, errors, qa = validator.validate_complete(valid_data)
        assert is_valid is True
        assert qa is not None
