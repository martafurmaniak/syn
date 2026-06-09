"""Azure OpenAI client wrapper with integrated token tracking.

Reads configuration from environment variables (load via python-dotenv):
    AZURE_OPENAI_ENDPOINT    e.g. https://my-resource.openai.azure.com/
    AZURE_OPENAI_KEY         API key
    AZURE_OPENAI_API_VERSION e.g. 2024-08-01-preview
    AZURE_OPENAI_DEPLOYMENT  deployment name (model alias in Azure portal)

Usage:
    from sow_synth.llm import LlmClient
    from sow_synth.telemetry import Telemetry

    tel = Telemetry()
    llm = LlmClient(tel)
    result = llm.complete("step_name", system_prompt, user_prompt, MyPydanticModel)
"""
from __future__ import annotations

import os
import time
from typing import TypeVar

from pydantic import BaseModel

from sow_synth.telemetry import Telemetry

T = TypeVar("T", bound=BaseModel)


class LlmClient:
    """Thin wrapper around AzureOpenAI that records token usage in Telemetry."""

    def __init__(self, telemetry: Telemetry) -> None:
        from openai import AzureOpenAI  # deferred so the library is optional

        self._client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
        self._deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self._tel = telemetry

    def complete(
        self,
        step: str,
        system: str,
        user: str,
        response_model: type[T],
    ) -> T:
        """Call the model, parse the response into response_model, record telemetry.

        Uses structured outputs (beta.chat.completions.parse) so the model always
        returns a schema-valid response — numbers in the prompt are never generated
        by the model, only the prose fields.
        """
        t0 = time.perf_counter()
        response = self._client.beta.chat.completions.parse(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format=response_model,
        )
        duration = time.perf_counter() - t0
        usage = response.usage
        self._tel.record_llm_call(
            name=step,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            duration_s=duration,
        )
        return response.choices[0].message.parsed
