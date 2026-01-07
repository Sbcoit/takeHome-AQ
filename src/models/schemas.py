"""Data models for the Physics QA system."""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class PhysicsTopic(str, Enum):
    """Physics topics for question generation."""

    CLASSICAL_MECHANICS = "Classical Mechanics"
    ELECTROMAGNETISM = "Electromagnetism"
    QUANTUM_MECHANICS = "Quantum Mechanics"
    STATISTICAL_MECHANICS = "Statistical Mechanics"
    THERMODYNAMICS = "Thermodynamics"
    SPECIAL_RELATIVITY = "Special Relativity"
    GENERAL_RELATIVITY = "General Relativity"
    CONDENSED_MATTER = "Condensed Matter Physics"
    NUCLEAR_PHYSICS = "Nuclear Physics"
    PARTICLE_PHYSICS = "Particle Physics"
    OPTICS = "Optics"
    FLUID_MECHANICS = "Fluid Mechanics"


class RubricCriterion(BaseModel):
    """A single criterion in a grading rubric."""

    criterion: str = Field(..., description="What is being evaluated")
    max_points: int = Field(..., ge=1, le=30, description="Maximum points for this criterion")
    description: str = Field(..., description="How to award points for this criterion")

    @field_validator("criterion")
    @classmethod
    def criterion_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Criterion cannot be empty")
        return v.strip()


class Rubric(BaseModel):
    """A complete grading rubric for a physics question."""

    total_points: int = Field(..., ge=10, le=100, description="Total points possible")
    criteria: List[RubricCriterion] = Field(
        ..., min_length=3, max_length=10, description="Grading criteria"
    )
    passing_threshold: int = Field(..., description="Minimum points to pass")

    @model_validator(mode="after")
    def validate_points(self) -> "Rubric":
        """Ensure criteria points sum to total and threshold is valid."""
        criteria_sum = sum(c.max_points for c in self.criteria)
        if criteria_sum != self.total_points:
            raise ValueError(
                f"Criteria points ({criteria_sum}) must equal total_points ({self.total_points})"
            )
        if self.passing_threshold > self.total_points:
            raise ValueError("Passing threshold cannot exceed total points")
        if self.passing_threshold < 1:
            raise ValueError("Passing threshold must be at least 1")
        return self


class PhysicsQADataPoint(BaseModel):
    """A single physics QA data point - the core output format."""

    query: str = Field(..., min_length=50, description="The graduate-level physics question")
    response_answer: str = Field(..., min_length=5, description="The final answer (concise)")
    response_reasoning: str = Field(
        ..., min_length=100, description="Step-by-step derivation/reasoning"
    )
    rubric: Rubric = Field(..., description="Grading rubric for the question")
    response_images: List[Any] = Field(
        default_factory=list, description="Must be empty (no images)"
    )

    # Metadata (not included in final output)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    topic: Optional[PhysicsTopic] = Field(default=None, description="Physics topic")
    subtopic: Optional[str] = Field(default=None, description="Specific subtopic")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    validation_passed: bool = Field(default=False)

    @field_validator("response_images")
    @classmethod
    def images_must_be_empty(cls, v: List[Any]) -> List[Any]:
        """Enforce the no-images constraint."""
        if v:
            raise ValueError("response_images must be empty - no images allowed")
        return []

    @field_validator("query")
    @classmethod
    def query_is_substantial(cls, v: str) -> str:
        """Ensure query is substantial for graduate-level."""
        v = v.strip()
        if len(v.split()) < 15:
            raise ValueError("Query should have at least 15 words for graduate-level complexity")
        return v

    @field_validator("response_reasoning")
    @classmethod
    def reasoning_is_substantial(cls, v: str) -> str:
        """Ensure reasoning is detailed enough."""
        v = v.strip()
        if len(v.split()) < 50:
            raise ValueError("Reasoning should have at least 50 words for proper derivation")
        return v

    def to_output_dict(self) -> Dict[str, Any]:
        """Return only the required output fields (no metadata)."""
        return {
            "query": self.query,
            "response_answer": self.response_answer,
            "response_reasoning": self.response_reasoning,
            "rubric": self.rubric.model_dump(),
            "response_images": [],
        }

    def to_json_line(self) -> str:
        """Return JSON string for JSONL output."""
        import json

        return json.dumps(self.to_output_dict())


class ValidationStatus(str, Enum):
    """Status of validation checks."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ModelTestResult(BaseModel):
    """Result of testing a single model on a question."""

    model: str
    samples: int
    correct_count: int
    responses: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    @property
    def accuracy(self) -> float:
        """Calculate accuracy as correct/samples."""
        return self.correct_count / self.samples if self.samples > 0 else 0.0


class ValidationResult(BaseModel):
    """Complete validation result for a QA pair."""

    data_point_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Schema validation
    schema_valid: bool = False
    schema_errors: List[str] = Field(default_factory=list)

    # No-images check
    images_check_passed: bool = False

    # Qwen constraint
    qwen_samples: int = 0
    qwen_high_pass_count: int = 0  # Count where Qwen passed 4+/5 criteria
    qwen_pass_rate: float = 0.0
    qwen_constraint_passed: bool = False

    # Cross-check validation
    crosscheck_results: Dict[str, ModelTestResult] = Field(default_factory=dict)
    crosscheck_models_with_correct: int = 0
    crosscheck_total_correct: int = 0
    crosscheck_passed: bool = False

    # Correctness judge
    correctness_score: float = 0.0
    correctness_details: Dict[str, Any] = Field(default_factory=dict)
    correctness_passed: bool = False

    # Overall
    overall_passed: bool = False
    failure_reasons: List[str] = Field(default_factory=list)

    def compute_overall(self) -> bool:
        """Compute overall pass status based on all checks."""
        self.failure_reasons = []

        if not self.schema_valid:
            self.failure_reasons.append(f"Schema validation failed: {self.schema_errors}")

        if not self.images_check_passed:
            self.failure_reasons.append("Images check failed (response_images not empty)")

        if not self.qwen_constraint_passed:
            self.failure_reasons.append(
                f"Qwen constraint failed (pass rate {self.qwen_pass_rate:.2%} > 5%)"
            )

        if not self.crosscheck_passed:
            self.failure_reasons.append(
                f"Cross-check failed (models with correct: {self.crosscheck_models_with_correct}, "
                f"total correct: {self.crosscheck_total_correct})"
            )

        if not self.correctness_passed:
            self.failure_reasons.append(
                f"Correctness check failed (score: {self.correctness_score:.2%})"
            )

        self.overall_passed = len(self.failure_reasons) == 0
        return self.overall_passed


class GenerationStats(BaseModel):
    """Statistics for the generation pipeline."""

    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Counts
    total_generated: int = 0
    passed_schema: int = 0
    passed_qwen: int = 0
    passed_crosscheck: int = 0
    passed_correctness: int = 0
    passed_all: int = 0
    failed: int = 0

    # By topic
    by_topic: Dict[str, int] = Field(default_factory=dict)

    # Timing
    total_api_calls: int = 0
    total_api_time_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        return self.passed_all / self.total_generated if self.total_generated > 0 else 0.0

    @property
    def avg_api_time(self) -> float:
        """Average API call time in seconds."""
        return (
            self.total_api_time_seconds / self.total_api_calls
            if self.total_api_calls > 0
            else 0.0
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a summary dictionary for display."""
        return {
            "total_generated": self.total_generated,
            "passed_all": self.passed_all,
            "failed": self.failed,
            "pass_rate": f"{self.pass_rate:.1%}",
            "by_stage": {
                "schema": self.passed_schema,
                "qwen_constraint": self.passed_qwen,
                "crosscheck": self.passed_crosscheck,
                "correctness": self.passed_correctness,
            },
            "by_topic": self.by_topic,
            "api_stats": {
                "total_calls": self.total_api_calls,
                "avg_time_ms": f"{self.avg_api_time * 1000:.0f}",
            },
        }


class GradingResult(BaseModel):
    """Result of grading a student answer against a rubric."""

    criteria_scores: Dict[str, int] = Field(
        default_factory=dict, description="Points awarded per criterion"
    )
    total_points: int = Field(default=0)
    max_points: int = Field(default=100)
    criteria_passed: int = Field(default=0, description="Number of criteria with >= 80% points")
    total_criteria: int = Field(default=0)
    passed: bool = Field(default=False)
    explanation: str = Field(default="")

    @property
    def score_percentage(self) -> float:
        """Calculate score as percentage."""
        return self.total_points / self.max_points if self.max_points > 0 else 0.0

    @property
    def criteria_pass_rate(self) -> float:
        """Calculate criteria pass rate."""
        return self.criteria_passed / self.total_criteria if self.total_criteria > 0 else 0.0
