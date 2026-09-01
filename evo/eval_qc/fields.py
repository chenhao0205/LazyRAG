from __future__ import annotations


def missing_required_fields(
    *,
    query: str,
    gt_answer: str,
    gt_text: list[str],
    key_points: list[str],
) -> list[str]:
    """Return the names of required case fields that are empty (order stable)."""
    missing: list[str] = []
    if not query.strip():
        missing.append('query')
    if not gt_answer.strip():
        missing.append('gt_answer')
    if not gt_text:
        missing.append('gt_text')
    if not key_points:
        missing.append('key_points')
    return missing
