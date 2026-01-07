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
                "total_points": 100,
                "criteria": [
                    {"criterion": "Physical setup", "max_points": 25, "description": "D1"},
                    {"criterion": "Mathematical formulation", "max_points": 30, "description": "D2"},
                    {"criterion": "Final answer", "max_points": 25, "description": "D3"},
                    {"criterion": "Units check", "max_points": 20, "description": "D4"},
                ],
                "passing_threshold": 60,
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

    def test_rubric_criteria_sum_mismatch_fails(self, validator, valid_data):
        """Test that rubric criteria sum mismatch fails."""
        valid_data["rubric"]["criteria"][0]["max_points"] = 10  # Changes sum
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is False
        assert any("sum" in e.lower() or "equal" in e.lower() for e in errors)

    def test_too_few_criteria_fails(self, validator, valid_data):
        """Test that too few criteria fails."""
        valid_data["rubric"]["total_points"] = 50
        valid_data["rubric"]["criteria"] = [
            {"criterion": "A", "max_points": 25, "description": "D1"},
            {"criterion": "B", "max_points": 25, "description": "D2"},
        ]
        is_valid, errors = validator.validate(valid_data)
        assert is_valid is False
        assert any("criteria" in e.lower() for e in errors)

    def test_missing_criterion_field_fails(self, validator, valid_data):
        """Test that missing criterion field fails."""
        del valid_data["rubric"]["criteria"][0]["criterion"]
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
