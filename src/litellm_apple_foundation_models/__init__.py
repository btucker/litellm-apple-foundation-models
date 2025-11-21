"""
LiteLLM custom provider for Apple's on-device Foundation Models.

Use ``register_provider()`` to attach the provider to ``litellm.custom_provider_map``.
"""

from .provider import (
    AppleFoundationModelsCustomLLM,
    register_apple_foundation_models_provider,
)

__all__ = [
    "AppleFoundationModelsCustomLLM",
    "register_apple_foundation_models_provider",
]
