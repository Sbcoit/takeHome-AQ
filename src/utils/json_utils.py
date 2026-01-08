"""JSON parsing utilities for LLM responses."""

import json
import re


def extract_json_from_response(content: str) -> dict:
    """
    Extract JSON from a response that may contain markdown or other text.

    Handles cases like:
    - Pure JSON: {"key": "value"}
    - Markdown wrapped: ```json\n{"key": "value"}\n```
    - Text before/after JSON: "Here is the result: {"key": "value"}"

    Args:
        content: The raw response content from an LLM

    Returns:
        Parsed JSON as a dictionary

    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted
    """
    content = content.strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code blocks
    json_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(json_block_pattern, content)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to find JSON object in the text (find first { and last })
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(content[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Nothing worked, raise error with content preview
    raise json.JSONDecodeError(
        f"Could not extract JSON from response: {content[:200]}...",
        content, 0
    )
