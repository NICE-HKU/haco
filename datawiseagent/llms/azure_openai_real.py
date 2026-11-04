from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, List, Optional, Union
from uuid import uuid4

from pydantic import Field, PrivateAttr
from tenacity import retry, stop_after_attempt, wait_exponential

from datawiseagent.common.log import logger
from datawiseagent.common.types.llm import LLMResult
from datawiseagent.common.utils.json_utils import JsonRepair
from datawiseagent.common.utils.misc import log_retry

from . import llm_registry
from .base import BaseChatModel, BaseModelArgs

try:
    from openai import AsyncAzureOpenAI, AzureOpenAI
    from openai._types import NOT_GIVEN, NotGiven
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )
except ImportError:
    AsyncAzureOpenAI = None
    AzureOpenAI = None
    NOT_GIVEN = None
    NotGiven = None
    ChatCompletion = None
    ChatCompletionMessageToolCall = None
    Function = None
    logger.warn("openai package is not installed, Azure real chat model is unavailable.")


class AzureOpenAIRealArgs(BaseModelArgs):
    model: str = Field(default="gpt-4o")
    base_url: str = Field(default="")
    key: str = Field(default="")
    version: str = Field(default="2024-12-01-preview")
    max_tokens: Optional[int] = Field(default=None)
    max_completion_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=1.0)
    top_p: Optional[float] = Field(default=1.0)
    n: Optional[int] = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=0)
    frequency_penalty: Optional[float] = Field(default=0)


@llm_registry.register("azure-openai-real-chat")
class AzureOpenAIRealChat(BaseChatModel):
    args: AzureOpenAIRealArgs = Field(default_factory=AzureOpenAIRealArgs)

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    _async_client: Any = PrivateAttr(default=None)
    _sync_client: Any = PrivateAttr(default=None)

    def __init__(self, max_retry: int = 3, **kwargs):
        if AsyncAzureOpenAI is None or AzureOpenAI is None:
            raise ImportError("openai package with Azure support is required for azure-openai-real-chat.")

        args = AzureOpenAIRealArgs().model_dump()
        for k, v in list(args.items()):
            args[k] = kwargs.pop(k, v)

        # Accept aliases for compatibility with external configs.
        base_url = args.get("base_url") or kwargs.pop("azure_endpoint", "")
        key = args.get("key") or kwargs.pop("api_key", "")
        version = args.get("version") or kwargs.pop("api_version", "2024-12-01-preview")

        args["base_url"] = base_url
        args["key"] = key
        args["version"] = version

        if kwargs:
            logger.warn(f"Unused arguments: {kwargs}")

        super().__init__(args=AzureOpenAIRealArgs(**args), max_retry=max_retry)
        self._async_client = AsyncAzureOpenAI(
            api_version=version,
            azure_endpoint=base_url,
            api_key=key,
        )
        self._sync_client = AzureOpenAI(
            api_version=version,
            azure_endpoint=base_url,
            api_key=key,
        )

    @staticmethod
    def _clean_model_args(model_args: dict) -> dict:
        # Azure OpenAI rejects unknown/None kwargs in some deployments.
        cleaned = {
            k: v
            for k, v in model_args.items()
            if v is not None and k not in {"base_url", "key", "version"}
        }

        # GPT-5 Azure chat deployments may reject explicit sampling controls
        # (for example temperature=0.0), requiring provider defaults.
        model_name = str(cleaned.get("model", "") or "").lower()
        if model_name.startswith("gpt-5"):
            for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
                cleaned.pop(key, None)

        return cleaned

    @staticmethod
    def _safe_usage(usage, field: str) -> int:
        if usage is None:
            return 0
        return int(getattr(usage, field, 0) or 0)

    @staticmethod
    def _convert_parallel_tool_calls(tool_calls):
        if not tool_calls:
            return tool_calls
        first = tool_calls[0]
        try:
            if first.function.name != "multi_tool_use.parallel":
                return tool_calls
            all_tool_calls = json.loads(first.function.arguments).get("tool_uses", [])
            converted = [
                ChatCompletionMessageToolCall(
                    id=f"call_{uuid4()}",
                    function=Function(
                        arguments=json.dumps(call["parameters"]),
                        name=call["recipient_name"].lstrip("functions."),
                    ),
                    type="function",
                )
                for call in all_tool_calls
            ]
            return converted
        except Exception:
            return tool_calls

    @staticmethod
    def _validate_tool_calls(tool_calls, tools):
        if tool_calls is None:
            return
        if tools is NOT_GIVEN:
            return
        all_tool_names = set(tool["function"]["name"] for tool in tools)
        for tool in tool_calls:
            if tool.function.name not in all_tool_names:
                raise ValueError(
                    f"The returned function name {tool.function.name} is not in the list of valid functions."
                )

    def construct_messages(
        self, prepend_prompt: list[str], history: list[dict], append_prompt: list[str]
    ):
        messages = []
        for prompt in prepend_prompt:
            if prompt != "":
                messages.append({"role": "system", "content": prompt})
        if history:
            messages += history
        for prompt in append_prompt:
            if prompt != "":
                messages.append({"role": "system", "content": prompt})
        return messages

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
        before_sleep=log_retry,
    )
    async def agenerate_response(
        self,
        prepend_prompt: str | list[str] = "",
        history: List[dict] = [],
        append_prompt: str | list[str] = "",
        tools: List[dict] | NotGiven = NOT_GIVEN,
        response_format: dict | NotGiven = NOT_GIVEN,
        **kwargs,
    ) -> LLMResult:
        if isinstance(prepend_prompt, str):
            prepend_prompt = [prepend_prompt]
        if isinstance(append_prompt, str):
            append_prompt = [append_prompt]

        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)

        model_args = deepcopy(self.args.model_dump())
        model_args.update(kwargs)
        model_args = self._clean_model_args(model_args)

        response: ChatCompletion = await self._async_client.chat.completions.create(
            messages=messages,
            tools=tools,
            response_format=response_format,
            **model_args,
        )

        if response.choices[0].finish_reason == "content_filter":
            raise ValueError("The response is filtered by Azure OpenAI content filter.")

        if response_format is not NOT_GIVEN and response_format.get("type") == "json_object":
            raw_content = response.choices[0].message.content
            if raw_content is not None:
                try:
                    content = json.loads(raw_content)
                except json.JSONDecodeError:
                    content = json.loads(JsonRepair(raw_content).repair())
            else:
                content = None
        else:
            content = response.choices[0].message.content

        tool_calls = self._convert_parallel_tool_calls(response.choices[0].message.tool_calls)
        self._validate_tool_calls(tool_calls, tools)

        usage = getattr(response, "usage", None)
        prompt_tokens = self._safe_usage(usage, "prompt_tokens")
        completion_tokens = self._safe_usage(usage, "completion_tokens")
        total_tokens = self._safe_usage(usage, "total_tokens")

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return LLMResult(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            send_tokens=prompt_tokens,
            recv_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
        before_sleep=log_retry,
    )
    def generate_response(
        self,
        prepend_prompt: str | list[str] = "",
        history: List[dict] = [],
        append_prompt: str | list[str] = "",
        tools: List[dict] | NotGiven = NOT_GIVEN,
        response_format: dict | NotGiven = NOT_GIVEN,
        **kwargs,
    ) -> LLMResult:
        if isinstance(prepend_prompt, str):
            prepend_prompt = [prepend_prompt]
        if isinstance(append_prompt, str):
            append_prompt = [append_prompt]

        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)

        model_args = deepcopy(self.args.model_dump())
        model_args.update(kwargs)
        model_args = self._clean_model_args(model_args)

        response: ChatCompletion = self._sync_client.chat.completions.create(
            messages=messages,
            tools=tools,
            response_format=response_format,
            **model_args,
        )

        if response.choices[0].finish_reason == "content_filter":
            raise ValueError("The response is filtered by Azure OpenAI content filter.")

        if response_format is not NOT_GIVEN and response_format.get("type") == "json_object":
            raw_content = response.choices[0].message.content
            if raw_content is not None:
                try:
                    content = json.loads(raw_content)
                except json.JSONDecodeError:
                    content = json.loads(JsonRepair(raw_content).repair())
            else:
                content = None
        else:
            content = response.choices[0].message.content

        tool_calls = self._convert_parallel_tool_calls(response.choices[0].message.tool_calls)
        self._validate_tool_calls(tool_calls, tools)

        usage = getattr(response, "usage", None)
        prompt_tokens = self._safe_usage(usage, "prompt_tokens")
        completion_tokens = self._safe_usage(usage, "completion_tokens")
        total_tokens = self._safe_usage(usage, "total_tokens")

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return LLMResult(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            send_tokens=prompt_tokens,
            recv_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def get_spend(self) -> float:
        # Pricing is deployment-specific in Azure.
        return 0.0
