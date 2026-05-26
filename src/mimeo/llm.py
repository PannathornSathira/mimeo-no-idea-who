"""OpenRouter LLM client with a structured-output helper.

We use the ``openai`` Python SDK pointed at OpenRouter's OpenAI-compatible
endpoint. Structured output goes through ``response_format`` in JSON mode,
then we validate against a pydantic model. We prefer this over
``client.beta.chat.completions.parse`` because not every OpenRouter-served
model supports the stricter schema mode.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    DEFAULT_MODEL,
    OPENROUTER_BASE_URL,
    PROMPTS_DIR,
    openrouter_default_headers,
    require_openrouter_key,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Thin wrapper around ``AsyncOpenAI`` pointed at the selected provider."""

    def __init__(self, model: str = DEFAULT_MODEL, provider: str = "auto") -> None:
        import os
        from .config import MissingCredentialError

        provider_lower = provider.lower() if provider else "auto"
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if provider_lower == "auto":
            if gemini_key:
                provider_lower = "gemini"
            elif openai_key:
                provider_lower = "openai"
            elif openrouter_key:
                provider_lower = "openrouter"
            elif model.startswith("gemini-"):
                provider_lower = "gemini"
            elif model.startswith("gpt-"):
                provider_lower = "openai"
            else:
                provider_lower = "openrouter"

        self.provider = provider_lower

        if provider_lower == "gemini":
            api_key = gemini_key or openrouter_key
            if not api_key:
                raise MissingCredentialError(
                    "GEMINI_API_KEY is not set. Get a free key from Google AI Studio."
                )
            base_url = os.environ.get("MIMEO_GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
            headers = {}
            if model == DEFAULT_MODEL:
                model = "gemini-2.5-flash"
        elif provider_lower == "openai":
            api_key = openai_key
            if not api_key:
                raise MissingCredentialError(
                    "OPENAI_API_KEY is not set. Please set it in your .env file."
                )
            base_url = os.environ.get("MIMEO_OPENAI_BASE_URL") or "https://api.openai.com/v1"
            headers = {}
            if model == DEFAULT_MODEL:
                model = "gpt-4o-mini"
        elif provider_lower == "ollama":
            api_key = "ollama"
            base_url = os.environ.get("MIMEO_OLLAMA_BASE_URL") or "http://localhost:11434/v1"
            headers = {}
            if model == DEFAULT_MODEL:
                model = "llama3"
        else:
            # Default to OpenRouter
            api_key = require_openrouter_key()
            base_url = OPENROUTER_BASE_URL
            headers = openrouter_default_headers() or {}

        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers or None,
        )

    async def complete(
        self,
        *,
        system: str | None,
        user: str,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Plain text completion."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        async for attempt in _network_retryer():
            with attempt:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
        raise RuntimeError("unreachable")  # pragma: no cover - tenacity reraises

    async def structured(
        self,
        *,
        system: str | None,
        user: str,
        schema: type[T],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> T:
        """Request JSON that validates against ``schema``.

        We give the model the schema inline (so it knows the shape) and ask
        for ``response_format={"type": "json_object"}``. If the response
        fails validation we retry up to :data:`_SCHEMA_REPAIR_ATTEMPTS` times,
        each retry sending the model its previous reply plus the error so it
        can self-correct.
        """
        schema_hint = _format_schema_hint(schema)
        augmented_user = (
            f"{user}\n\n"
            f"Return ONLY a JSON object matching this Pydantic schema:\n"
            f"{schema_hint}\n\n"
            f"Do not include any commentary, code fences, or prose - JSON only."
        )

        base_messages: list[dict[str, str]] = []
        if system:
            base_messages.append({"role": "system", "content": system})
        base_messages.append({"role": "user", "content": augmented_user})

        last_raw: str | None = None
        last_error: str | None = None

        for repair_attempt in range(_SCHEMA_REPAIR_ATTEMPTS):
            messages = list(base_messages)
            if last_raw is not None and last_error is not None:
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed schema validation:\n"
                            f"{last_error}\n"
                            "Return corrected JSON matching the schema exactly, "
                            "with no commentary or code fences."
                        ),
                    }
                )

            # Inner retry handles transient network/5xx errors.
            raw = ""
            async for attempt in _network_retryer():
                with attempt:
                    resp = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    raw = (resp.choices[0].message.content or "").strip()

            try:
                data = json.loads(_strip_code_fence(raw))
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_raw = raw
                last_error = str(exc)[:1500]
                remaining = _SCHEMA_REPAIR_ATTEMPTS - repair_attempt - 1
                logger.warning(
                    "Schema validation failed (%d repair attempts left): %s",
                    remaining,
                    exc,
                )
                if remaining == 0:
                    raise

        raise RuntimeError("unreachable")  # pragma: no cover - loop always returns or raises


# Status codes where a retry has a reasonable chance of succeeding. 4xx
# errors like 401 (bad key) or 400 (bad request) will keep failing, so we
# surface them immediately rather than burning the budget.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# How many times we'll ask the model to repair a response that parsed but
# didn't match the schema. Network-level retries are separate and happen
# inside each attempt.
_SCHEMA_REPAIR_ATTEMPTS = 3


def _is_network_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return False


def _network_retryer() -> AsyncRetrying:
    """Retry only on transient network / 5xx errors, not schema failures."""
    return AsyncRetrying(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception(_is_network_retryable),
        reraise=True,
    )


def _strip_code_fence(text: str) -> str:
    """Some models still wrap JSON in ```json ... ``` even when told not to."""
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence (```json or ```)
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _format_schema_hint(schema: type[BaseModel]) -> str:
    """Compact JSON-schema description the model can follow."""
    return json.dumps(schema.model_json_schema(), indent=2)


def load_prompt(name: str) -> str:
    """Load a prompt template from the ``prompts/`` directory.

    ``name`` may or may not include the ``.md`` extension.
    """
    filename = name if name.endswith(".md") else f"{name}.md"
    path: Path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **values: str) -> str:
    """Substitute ``{name}`` placeholders in ``template`` with ``values``.

    Unlike ``str.format``, this only touches exact ``{key}`` tokens for keys
    we pass in, so the template body can contain literal ``{...}`` skeleton
    braces (e.g. ``{2-3 paragraph overview}``) without escaping.
    """
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", val)
    return out
