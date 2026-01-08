"""Prompt templates and response formats."""

from .response_formats import (
    JSON_INSTRUCTION,
    LATEX_FORMAT_GUIDE,
    ANSWER_FORMAT_GUIDE,
    FINAL_ANSWER_SCHEMA,
    EQUIVALENCE_SCHEMA,
    GRADING_SCHEMA,
    QA_GENERATION_SCHEMA,
    SOLUTION_SCHEMA,
    with_json_format,
    with_answer_format,
)

__all__ = [
    "JSON_INSTRUCTION",
    "LATEX_FORMAT_GUIDE",
    "ANSWER_FORMAT_GUIDE",
    "FINAL_ANSWER_SCHEMA",
    "EQUIVALENCE_SCHEMA",
    "GRADING_SCHEMA",
    "QA_GENERATION_SCHEMA",
    "SOLUTION_SCHEMA",
    "with_json_format",
    "with_answer_format",
]
