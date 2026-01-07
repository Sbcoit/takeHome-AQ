"""Tests for data models and schemas."""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    PhysicsQADataPoint,
    Rubric,
    RubricCriterion,
    ValidationResult,
    GenerationStats,
    PhysicsTopic,
)


class TestRubricCriterion:
    """Tests for RubricCriterion model."""

    def test_valid_criterion(self):
        """Test creating a valid criterion."""
        criterion = RubricCriterion(
            criterion="Test criterion",
            max_points=20,
            description="Test description",
        )
        assert criterion.criterion == "Test criterion"
        assert criterion.max_points == 20
        assert criterion.description == "Test description"

    def test_empty_criterion_fails(self):
        """Test that empty criterion name fails."""
        with pytest.raises(ValidationError):
            RubricCriterion(
                criterion="",
                max_points=20,
                description="Test description",
            )

    def test_negative_points_fails(self):
        """Test that negative points fail."""
        with pytest.raises(ValidationError):
            RubricCriterion(
                criterion="Test",
                max_points=-5,
                description="Test",
            )


class TestRubric:
    """Tests for Rubric model."""

    def test_valid_rubric(self):
        """Test creating a valid rubric."""
        rubric = Rubric(
            total_points=100,
            criteria=[
                RubricCriterion(criterion="A", max_points=25, description="D1"),
                RubricCriterion(criterion="B", max_points=25, description="D2"),
                RubricCriterion(criterion="C", max_points=25, description="D3"),
                RubricCriterion(criterion="D", max_points=25, description="D4"),
            ],
            passing_threshold=60,
        )
        assert rubric.total_points == 100
        assert len(rubric.criteria) == 4
        assert rubric.passing_threshold == 60

    def test_criteria_must_sum_to_total(self):
        """Test that criteria points must sum to total_points."""
        with pytest.raises(ValidationError) as exc_info:
            Rubric(
                total_points=100,
                criteria=[
                    RubricCriterion(criterion="A", max_points=30, description="D1"),
                    RubricCriterion(criterion="B", max_points=30, description="D2"),
                    RubricCriterion(criterion="C", max_points=30, description="D3"),
                ],
                passing_threshold=60,
            )
        assert "must equal total_points" in str(exc_info.value)

    def test_min_criteria_required(self):
        """Test that minimum criteria are required."""
        with pytest.raises(ValidationError):
            Rubric(
                total_points=50,
                criteria=[
                    RubricCriterion(criterion="A", max_points=25, description="D1"),
                    RubricCriterion(criterion="B", max_points=25, description="D2"),
                ],
                passing_threshold=30,
            )

    def test_passing_threshold_validation(self):
        """Test that passing threshold must be valid."""
        with pytest.raises(ValidationError):
            Rubric(
                total_points=100,
                criteria=[
                    RubricCriterion(criterion="A", max_points=25, description="D1"),
                    RubricCriterion(criterion="B", max_points=25, description="D2"),
                    RubricCriterion(criterion="C", max_points=25, description="D3"),
                    RubricCriterion(criterion="D", max_points=25, description="D4"),
                ],
                passing_threshold=150,  # Exceeds total
            )


class TestPhysicsQADataPoint:
    """Tests for PhysicsQADataPoint model."""

    @pytest.fixture
    def valid_qa_data(self):
        """Fixture for valid QA data."""
        return {
            "query": "Consider a quantum harmonic oscillator with frequency omega. " * 5,
            "response_answer": "E_n = hbar * omega * (n + 1/2)",
            "response_reasoning": "Starting with the Schrodinger equation... " * 20,
            "rubric": {
                "total_points": 100,
                "criteria": [
                    {"criterion": "Setup", "max_points": 25, "description": "D1"},
                    {"criterion": "Math", "max_points": 30, "description": "D2"},
                    {"criterion": "Answer", "max_points": 25, "description": "D3"},
                    {"criterion": "Units", "max_points": 20, "description": "D4"},
                ],
                "passing_threshold": 60,
            },
            "response_images": [],
        }

    def test_valid_qa_point(self, valid_qa_data):
        """Test creating a valid QA data point."""
        qa = PhysicsQADataPoint(**valid_qa_data)
        assert qa.response_images == []
        assert qa.topic is None

    def test_images_must_be_empty(self, valid_qa_data):
        """Test that response_images must be empty."""
        valid_qa_data["response_images"] = ["some_image.png"]
        with pytest.raises(ValidationError) as exc_info:
            PhysicsQADataPoint(**valid_qa_data)
        assert "must be empty" in str(exc_info.value).lower()

    def test_query_minimum_length(self, valid_qa_data):
        """Test that query must meet minimum length."""
        valid_qa_data["query"] = "Short query"
        with pytest.raises(ValidationError):
            PhysicsQADataPoint(**valid_qa_data)

    def test_reasoning_minimum_length(self, valid_qa_data):
        """Test that reasoning must meet minimum length."""
        valid_qa_data["response_reasoning"] = "Short"
        with pytest.raises(ValidationError):
            PhysicsQADataPoint(**valid_qa_data)

    def test_to_output_dict(self, valid_qa_data):
        """Test output dict contains only required fields."""
        qa = PhysicsQADataPoint(**valid_qa_data)
        output = qa.to_output_dict()

        assert set(output.keys()) == {
            "query",
            "response_answer",
            "response_reasoning",
            "rubric",
            "response_images",
        }
        assert output["response_images"] == []

    def test_to_json_line(self, valid_qa_data):
        """Test JSON line output."""
        qa = PhysicsQADataPoint(**valid_qa_data)
        json_line = qa.to_json_line()

        import json
        parsed = json.loads(json_line)
        assert "query" in parsed
        assert "rubric" in parsed
        assert parsed["response_images"] == []


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_compute_overall_all_pass(self):
        """Test compute_overall when all checks pass."""
        result = ValidationResult(
            data_point_id="test-123",
            schema_valid=True,
            images_check_passed=True,
            qwen_constraint_passed=True,
            crosscheck_passed=True,
            correctness_passed=True,
        )
        assert result.compute_overall() is True
        assert result.overall_passed is True
        assert len(result.failure_reasons) == 0

    def test_compute_overall_with_failures(self):
        """Test compute_overall with failures."""
        result = ValidationResult(
            data_point_id="test-123",
            schema_valid=True,
            images_check_passed=True,
            qwen_constraint_passed=False,
            qwen_pass_rate=0.10,
            crosscheck_passed=True,
            correctness_passed=True,
        )
        assert result.compute_overall() is False
        assert result.overall_passed is False
        assert len(result.failure_reasons) == 1
        assert "Qwen constraint" in result.failure_reasons[0]


class TestGenerationStats:
    """Tests for GenerationStats model."""

    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        stats = GenerationStats(
            total_generated=100,
            passed_all=25,
        )
        assert stats.pass_rate == 0.25

    def test_pass_rate_zero_generated(self):
        """Test pass rate with zero generated."""
        stats = GenerationStats()
        assert stats.pass_rate == 0.0

    def test_summary_dict(self):
        """Test summary dict generation."""
        stats = GenerationStats(
            total_generated=50,
            passed_schema=45,
            passed_qwen=30,
            passed_crosscheck=25,
            passed_correctness=20,
            passed_all=20,
            failed=30,
        )
        summary = stats.to_summary_dict()

        assert summary["total_generated"] == 50
        assert summary["passed_all"] == 20
        assert summary["failed"] == 30
        assert "by_stage" in summary
        assert summary["by_stage"]["schema"] == 45
