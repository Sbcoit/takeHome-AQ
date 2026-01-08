"""Tests for data models and schemas."""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    PhysicsQADataPoint,
    Rubric,
    KeyStep,
    FinalAnswer,
    ValidationResult,
    GenerationStats,
    PhysicsTopic,
)


class TestRubric:
    """Tests for Rubric model."""

    @pytest.fixture
    def valid_rubric_data(self):
        """Fixture for valid rubric data."""
        return {
            "final_answer": {
                "value": "E = mc^2",
                "points": 3,
                "tolerance": "equivalent symbolic forms accepted",
                "common_errors": ["Missing factor of 2", "Wrong sign"]
            },
            "key_steps": [
                {"step": "Apply conservation of energy", "points": 2},
                {"step": "Apply conservation of momentum", "points": 2},
                {"step": "Solve algebraically for final answer", "points": 3}
            ],
            "partial_credit_rules": ["Correct method but arithmetic error: deduct 1-2 points"],
            "automatic_zero": ["Uses completely wrong method for the problem type"]
        }

    def test_valid_rubric(self, valid_rubric_data):
        """Test creating a valid rubric."""
        rubric = Rubric(**valid_rubric_data)
        assert rubric.total_points == 10
        assert rubric.final_answer.value == "E = mc^2"
        assert rubric.final_answer.points == 3
        assert len(rubric.key_steps) == 3
        assert sum(step.points for step in rubric.key_steps) == 7

    def test_key_steps_points_validation(self, valid_rubric_data):
        """Test that key steps must sum to 5-9 points."""
        # Too few points
        valid_rubric_data["key_steps"] = [
            {"step": "Step 1", "points": 1},
            {"step": "Step 2", "points": 1},
            {"step": "Step 3", "points": 1}
        ]
        with pytest.raises(ValidationError):
            Rubric(**valid_rubric_data)

    def test_key_steps_minimum_count(self, valid_rubric_data):
        """Test that at least 3 key steps are required."""
        valid_rubric_data["key_steps"] = [
            {"step": "Step 1", "points": 3},
            {"step": "Step 2", "points": 4}
        ]
        with pytest.raises(ValidationError):
            Rubric(**valid_rubric_data)

    def test_key_steps_maximum_count(self, valid_rubric_data):
        """Test that at most 7 key steps are allowed."""
        valid_rubric_data["key_steps"] = [
            {"step": f"Step {i}", "points": 1} for i in range(8)
        ]
        with pytest.raises(ValidationError):
            Rubric(**valid_rubric_data)

    def test_final_answer_required(self):
        """Test that final_answer is required."""
        with pytest.raises(ValidationError):
            Rubric(
                key_steps=[
                    {"step": "Step 1", "points": 2},
                    {"step": "Step 2", "points": 2},
                    {"step": "Step 3", "points": 3}
                ]
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
