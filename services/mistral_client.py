"""
Mistral LLM client — thin wrapper for generating narrative commentary.
"""

from __future__ import annotations

import os


def get_mistral_commentary(
    prompt: str, model: str = "mistral-small-latest"
) -> str | None:
    """Call Mistral API and return a narrative commentary string.

    Uses a lazy import so the app degrades gracefully if mistralai is not installed.
    Returns None on any error (missing key, network failure, rate-limit, etc.).
    """
    try:
        from mistralai import Mistral  # lazy import
    except ImportError:
        return None

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return None

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content  # type: ignore
    except Exception:
        return None
