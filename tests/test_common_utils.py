"""
Unit tests for Apple Foundation Models common utilities (v0.2.0+ SDK).
"""

import sys
from pathlib import Path

import pytest
from applefoundationmodels import AsyncSession, Session

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))

from litellm_apple_foundation_models.common_utils import (  # noqa: E402
    get_apple_async_session_class,
    get_apple_session_class,
)


class TestAppleFoundationModelsCommonUtils:
    """Test suite for Apple Foundation Models common utilities (v0.2.0+ SDK)."""

    def test_get_apple_session_class_availability_failure(self, monkeypatch):
        """Test that RuntimeError is raised when Apple Intelligence not available."""
        monkeypatch.setattr(
            "litellm_apple_foundation_models.common_utils.apple_intelligence_available",
            lambda: False,
        )
        with pytest.raises(RuntimeError) as exc_info:
            get_apple_session_class()

        assert "Apple Intelligence" in str(exc_info.value)

    def test_get_apple_session_class_success(self, monkeypatch):
        """Test successful Session class retrieval."""
        monkeypatch.setattr(
            "litellm_apple_foundation_models.common_utils.apple_intelligence_available",
            lambda: True,
        )
        result = get_apple_session_class()
        assert result is Session

    def test_get_apple_async_session_class_availability_failure(self, monkeypatch):
        """Test that RuntimeError is raised when Apple Intelligence not available (async)."""
        monkeypatch.setattr(
            "litellm_apple_foundation_models.common_utils.apple_intelligence_available",
            lambda: False,
        )
        with pytest.raises(RuntimeError) as exc_info:
            get_apple_async_session_class()

        assert "Apple Intelligence" in str(exc_info.value)

    def test_get_apple_async_session_class_success(self, monkeypatch):
        """Test successful AsyncSession class retrieval."""
        monkeypatch.setattr(
            "litellm_apple_foundation_models.common_utils.apple_intelligence_available",
            lambda: True,
        )
        result = get_apple_async_session_class()
        assert result is AsyncSession
