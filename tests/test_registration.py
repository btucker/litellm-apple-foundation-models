import litellm

from litellm_apple_foundation_models import (
    AppleFoundationModelsCustomLLM,
    register_apple_foundation_models_provider,
)


def test_register_provider_adds_custom_handler():
    prev_map = list(getattr(litellm, "custom_provider_map", []))
    prev_custom_providers = list(getattr(litellm, "_custom_providers", []))
    try:
        handler = register_apple_foundation_models_provider()
        assert isinstance(handler, AppleFoundationModelsCustomLLM)
        assert any(
            entry.get("provider") == "apple_foundation_models"
            for entry in litellm.custom_provider_map
        )
        assert "apple_foundation_models" in litellm._custom_providers
    finally:
        litellm.custom_provider_map = prev_map
        litellm._custom_providers = prev_custom_providers
