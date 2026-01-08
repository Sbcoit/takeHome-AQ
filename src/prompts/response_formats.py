"""Standardized response formats for all LLM interactions.

This module provides consistent JSON response format instructions
that can be appended to any prompt to ensure uniform model responses.
"""

# =============================================================================
# UNIVERSAL JSON INSTRUCTION - append to any prompt requiring JSON output
# =============================================================================
JSON_INSTRUCTION = """

RESPONSE FORMAT: You MUST respond with ONLY a valid JSON object.
- No text before or after the JSON
- No markdown code blocks (no ```)
- No explanations outside the JSON
- Start your response with {{ and end with }}"""


# =============================================================================
# LATEX MATH FORMAT - use LaTeX for all mathematical expressions
# =============================================================================
LATEX_FORMAT_GUIDE = r"""
MATH NOTATION: Use LaTeX for ALL mathematical expressions.
- Inline math: $expression$ (e.g., $E = mc^2$)
- Display math: $$expression$$ for important equations
- Variables: $m$, $\omega$, $\hbar$, $\vec{{r}}$
- Fractions: $\frac{{a}}{{b}}$
- Powers: $x^2$, $e^{{-x}}$
- Roots: $\sqrt{{x}}$, $\sqrt[3]{{x}}$
- Greek: $\alpha$, $\beta$, $\omega$, $\pi$
- Operators: $\partial$, $\nabla$, $\int$, $\sum$

IMPORTANT FOR JSON: In JSON strings, escape backslashes as \\
- Write "\\frac{{a}}{{b}}" in JSON (renders as $\frac{{a}}{{b}}$)
- Write "\\sqrt{{x}}" in JSON (renders as $\sqrt{{x}}$)
"""


# =============================================================================
# PHYSICS ANSWER FORMAT - how models should express their final answers
# =============================================================================
ANSWER_FORMAT_GUIDE = r"""
ANSWER FORMAT RULES:
1. Express answers SYMBOLICALLY using LaTeX notation
2. In JSON, escape backslashes: "\\frac{{1}}{{2}}", "\\sqrt{{3}}", "\\pi"
3. Simplify ratios completely: 4/6 -> 2/3
4. NO decimals: use \\pi not 3.14159, use \\sqrt{{2}} not 1.414
5. Include units only if problem gives numerical values

IN JSON WRITE:
- "\\frac{{m \\omega^2 R^2}}{{2}}" for fractions
- "\\sqrt{{3}}/2" for roots
- "n(n+1)\\hbar^2" for products

BAD: "1.333", "6.28", "approximately 4"
"""


# =============================================================================
# JSON SCHEMAS - specific response structures
# =============================================================================
FINAL_ANSWER_SCHEMA = """{
    "final_answer": "symbolic answer (e.g., mω²R²/2, 4/3, 2πℏ)"
}"""

EQUIVALENCE_SCHEMA = """{
    "equivalent": true or false,
    "explanation": "one sentence why"
}"""

GRADING_SCHEMA = """{
    "correct_physics": {"passed": true/false, "explanation": "one sentence"},
    "correct_answer": {"passed": true/false, "explanation": "one sentence"},
    "sound_reasoning": {"passed": true/false, "explanation": "one sentence"},
    "is_correct": true/false,
    "explanation": "one sentence summary"
}"""

QA_GENERATION_SCHEMA = """{
    "query": "problem statement",
    "response_answer": "symbolic answer",
    "response_reasoning": "solution steps",
    "rubric": {
        "correct_physics": "required physics principles",
        "correct_answer": "expected answer and equivalents",
        "sound_reasoning": "required mathematical steps"
    },
    "response_images": []
}"""

SOLUTION_SCHEMA = """{
    "solution": "step-by-step solution",
    "final_answer": "symbolic answer"
}"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def with_json_format(prompt: str, schema: str = "") -> str:
    """
    Append JSON format instructions to a prompt.

    Args:
        prompt: The base prompt
        schema: Optional specific JSON schema to include

    Returns:
        Prompt with JSON format instructions appended
    """
    result = prompt + JSON_INSTRUCTION
    if schema:
        result += f"\n\nExpected format:\n{schema}"
    return result


def with_answer_format(prompt: str) -> str:
    """
    Append answer format guide to a prompt.

    Args:
        prompt: The base prompt

    Returns:
        Prompt with answer format rules appended
    """
    return prompt + ANSWER_FORMAT_GUIDE
