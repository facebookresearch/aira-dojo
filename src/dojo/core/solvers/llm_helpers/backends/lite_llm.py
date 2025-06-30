# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jsonschema
import litellm
import time
import httpx

from dataclasses_json import DataClassJsonMixin
from litellm import completion as completion_fn

litellm.api_version = "2024-12-01-preview"
litellm.set_verbose = False

NUM_RETRIES = 10
TIMEOUT = 1500


# Configure logging
logger = logging.getLogger("Backend")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: Dict[str, Any]  # JSON schema
    description: str

    def __post_init__(self):
        # Validate the JSON schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI's function format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_tool_choice_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_anthropic_tool_dict(self):
        """Convert to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,  # Anthropic uses input_schema instead of parameters
        }

    @property
    def anthropic_tool_choice_dict(self):
        """Convert to Anthropic's tool choice format."""
        return {
            "type": "tool",  # Anthropic uses "tool" instead of "function"
            "name": self.name,
        }


class LiteLLMClient:
    PromptType = Union[str, Dict[str, Any], List[Any]]
    FunctionCallType = Dict[str, Any]
    OutputType = Union[str, FunctionCallType]

    def __init__(self, client_cfg):
        """
        Initialize the OpenAI client with any desired default arguments or configuration.
        """
        self.model = client_cfg.model_id
        self.base_url = client_cfg.base_url
        api_key = os.getenv("PRIMARY_KEY_" + self.model.replace("-", "_").upper(), "")
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("PRIMARY_KEY", "")
        self.use_azure_client = client_cfg.use_azure_client
        self.provider = client_cfg.provider
        if self.use_azure_client:
            self.model_prefix = "azure/"
        else:
            self.model_prefix = "openai/"

        self.model = self.model_prefix + self.model

        logging.getLogger("httpx").setLevel(logging.WARNING)

    @property
    def client_content_key(self):
        return "content"

    def _calculate_cost(self, prompt_tokens, completion_tokens):
        """Calculate the API cost for a request based on token usage and provider-specific pricing."""
        cost = 0.0
        # Example cost calculation for different providers/models
        if self.provider.lower() == "openai":
            # Define cost per 1K tokens for some known OpenAI models (in USD)
            if "gpt-3.5" in self.model.lower():
                prompt_cost_per_1k = 0.0
                completion_cost_per_1k = 0.0
            elif "gpt-4" in self.model.lower():
                prompt_cost_per_1k = 0.0
                completion_cost_per_1k = 0.0
            else:
                # Default rates for other OpenAI models (if any)
                prompt_cost_per_1k = 0.0
                completion_cost_per_1k = 0.0
            # Calculate cost proportionally to the number of tokens (token counts are divided by 1000 for per-1K pricing)
            cost = (prompt_tokens / 1000.0) * prompt_cost_per_1k + (
                completion_tokens / 1000.0
            ) * completion_cost_per_1k
        elif self.provider.lower() == "anthropic":
            prompt_cost_per_1k = 0.0  # example cost per 1K tokens for prompts on Anthropic
            completion_cost_per_1k = 0.0  # example cost per 1K tokens for completions on Anthropic
            cost = (prompt_tokens / 1000.0) * prompt_cost_per_1k + (
                completion_tokens / 1000.0
            ) * completion_cost_per_1k
        else:
            # Other providers or default case
            # If costs are not known, leave as 0 or implement accordingly
            cost = 0.0
        return round(cost, 6)  # rounding to a reasonable number of decimal places for currency

    def count_tokens(self, text):
        """Utility method to count tokens in a given text string."""
        # In a real scenario, this should use the model's tokenizer for accuracy.
        # Here, we'll use a simple whitespace split as a placeholder.
        if text is None:
            return 0
        return len(text.split())

    def _query_client(
        self,
        messages: List[Dict[str, str]],
        model_kwargs: Dict[str, Any] = {},
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Tuple[OutputType, Dict[str, Any]]:
        # Prepare function specifications if provided
        func_spec = None
        if json_schema and function_name and function_description:
            func_spec = FunctionSpec(function_name, json.loads(json_schema), function_description)

        # Always include necessary model parameters
        model_kwargs["model"] = self.model
        model_kwargs["base_url"] = self.base_url
        model_kwargs["api_key"] = self.api_key
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        # Attach function specifications if provided
        if func_spec is not None:
            filtered_kwargs["functions"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["function_call"] = "auto"

        filtered_kwargs["max_retries"] = NUM_RETRIES
        filtered_kwargs["num_retries"] = NUM_RETRIES
        filtered_kwargs["request_timeout"] = httpx.Timeout(timeout=TIMEOUT)

        # Record start time for latency measurement
        start_time = time.monotonic()

        # Execute the LLM call, with fallback for function calling errors
        completion = None
        try:
            completion = completion_fn(messages=messages, **filtered_kwargs)
        except litellm.BadRequestError as e:
            if "function calling" in str(e).lower() or "functions" in str(e).lower():
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )
                # Remove function calling parameters and retry
                filtered_kwargs.pop("functions", None)
                filtered_kwargs.pop("function_call", None)
                completion = completion_fn(messages=messages, **filtered_kwargs)
            else:
                raise

        # Calculate latency
        latency = time.monotonic() - start_time

        # Extract usage stats from the LLM response (if available)
        choice = completion.choices[0]
        if completion is not None:
            usage_stats = completion.to_dict().get("usage", {})
        else:
            usage_stats = {}

        # Add latency and success status to the stats
        usage_stats["latency"] = latency
        usage_stats["success"] = True

        # If token counts are not available from the response, estimate them.
        if "prompt_tokens" not in usage_stats:
            prompt_text = " ".join([m.get("content", "") for m in messages])
            usage_stats["prompt_tokens"] = self.count_tokens(prompt_text)
        if "completion_tokens" not in usage_stats:
            usage_stats["completion_tokens"] = self.count_tokens(choice.message.content)
        usage_stats["total_tokens"] = usage_stats["prompt_tokens"] + usage_stats["completion_tokens"]

        # Calculate cost using a helper (this method can adjust for different backends)
        usage_stats["cost"] = self._calculate_cost(usage_stats["prompt_tokens"], usage_stats["completion_tokens"])

        # Parse the response as before
        if func_spec is None or "functions" not in filtered_kwargs:
            output = choice.message.content
        else:
            try:
                function_call = choice.message.function_call
            except:
                function_call = None
            if not function_call:
                logger.warning(
                    "No function call was used despite function spec. Fallback to text.\n"
                    f"Message content: {choice.message.content}"
                )
                output = choice.message.content
            else:
                if not str(function_call.name).strip() == str(func_spec.name).strip():
                    logger.warning(
                        f"Function name mismatch: expected {func_spec.name}, "
                        f"got {function_call.name}. Fallback to text."
                    )
                    output = choice.message.content
                else:
                    try:
                        output = json.loads(function_call.arguments)
                    except json.JSONDecodeError as ex:
                        logger.error(f"Error decoding function arguments:\n{function_call.arguments}")
                        raise ex

        return output, usage_stats

    def query(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
        **model_kwargs,
    ) -> OutputType:
        """
        General LLM query for various backends with a single system and user message.
        Supports function calling for some backends.

        Args:
            system_message (PromptType | None): Uncompiled system message.
            user_message (PromptType | None): Uncompiled user message.
            model (str): Identifier for the model to use (e.g., "gpt-4-turbo").
            temperature (float | None, optional): Sampling temperature.
            max_tokens (int | None, optional): Maximum number of tokens to generate.
            func_spec (FunctionSpec | None, optional): Optional FunctionSpec for function calling.
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            OutputType: A string completion or a dict with function call details.
        """

        if self.model == "azure/o1-preview" or self.model == "azure/o3-mini":
            messages = [{"role": "user", self.client_content_key: m[self.client_content_key]} for m in messages]
            if "temperature" in model_kwargs:
                model_kwargs.pop("temperature")

        output, usage_stats = self._query_client(
            messages=messages,
            model_kwargs=model_kwargs,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )

        return output, usage_stats
