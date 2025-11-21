from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Union

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.utils import custom_llm_setup

from .chat.transformation import AppleFoundationModelsLLM


class AppleFoundationModelsCustomLLM(CustomLLM):
    """
    CustomLLM wrapper that reuses the core Apple Foundation Models implementation.

    This allows registering the provider via ``litellm.custom_provider_map`` without
    adding it to LiteLLM core.
    """

    def __init__(self) -> None:
        super().__init__()
        self._backend = AppleFoundationModelsLLM()

    def _dispatch(
        self,
        *,
        model: str,
        messages: list,
        model_response: ModelResponse,
        logging_obj,
        optional_params: Optional[dict],
        stream: bool,
        async_mode: bool,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Call into the Apple backend and normalize streaming output for CustomLLM usage.
        """
        response = self._backend.dispatch_completion(
            model=model,
            messages=messages,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params or {},
            stream=stream,
            async_mode=async_mode,
        )

        if stream and isinstance(response, CustomStreamWrapper):
            return response.completion_stream
        return response

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[Dict[str, Any]] = None,
        timeout=None,
        client: Optional[HTTPHandler] = None,
    ) -> ModelResponse:
        return self._dispatch(
            model=model,
            messages=messages,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            stream=False,
            async_mode=False,
        )  # type: ignore[return-value]

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[Dict[str, Any]] = None,
        timeout=None,
        client: Optional[HTTPHandler] = None,
    ):
        return self._dispatch(
            model=model,
            messages=messages,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            stream=True,
            async_mode=False,
        )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[Dict[str, Any]] = None,
        timeout=None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ModelResponse:
        response = self._dispatch(
            model=model,
            messages=messages,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            stream=False,
            async_mode=True,
        )
        if inspect.isawaitable(response):
            response = await response
        return response  # type: ignore[return-value]

    def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[Dict[str, Any]] = None,
        timeout=None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[GenericStreamingChunk, CustomStreamWrapper, Any]:
        return self._dispatch(
            model=model,
            messages=messages,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            stream=True,
            async_mode=True,
        )


def register_provider(provider_name: str = "apple_foundation_models") -> AppleFoundationModelsCustomLLM:
    """
    Register the custom provider with LiteLLM and return the handler instance.
    """
    handler = AppleFoundationModelsCustomLLM()
    provider_map = list(getattr(litellm, "custom_provider_map", []))

    for entry in provider_map:
        if entry.get("provider") == provider_name:
            entry["custom_handler"] = handler
            break
    else:
        provider_map.append({"provider": provider_name, "custom_handler": handler})

    litellm.custom_provider_map = provider_map
    custom_llm_setup()
    return handler
