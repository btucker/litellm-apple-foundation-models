"""
Common utilities for Apple Foundation Models provider.

Handles availability checks for v0.2.0+ SDK.
"""

from typing import Any

from applefoundationmodels import AsyncSession, Session, apple_intelligence_available

from litellm._logging import verbose_logger


def _ensure_available() -> None:
    """
    Verify Apple Intelligence is available on this system.

    Raises:
        RuntimeError: If Apple Intelligence is not available
    """
    verbose_logger.debug("Checking Apple Intelligence availability")
    try:
        available = apple_intelligence_available()
    except Exception as exc:  # pragma: no cover - SDK specific
        raise RuntimeError(
            f"Failed to determine Apple Intelligence availability: {exc}"
        ) from exc

    if not available:
        raise RuntimeError(
            "Apple Intelligence is not available on this system. "
            "Requirements: macOS 26.0+ (Sequoia) with Apple Intelligence enabled."
        )


def get_apple_session_class() -> Any:
    """
    Get Apple Foundation Models Session class.

    Returns:
        Apple Foundation Models Session class
    """
    _ensure_available()
    return Session


def get_apple_async_session_class() -> Any:
    """
    Get Apple Foundation Models AsyncSession class.

    Returns:
        Apple Foundation Models AsyncSession class
    """
    _ensure_available()
    return AsyncSession
