"""Async API client with support for OpenRouter and OpenAI endpoints."""

import aiohttp
import asyncio
import json
import random
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class APICallMetrics:
    """Metrics for an API call."""

    model: str
    start_time: float
    end_time: float
    success: bool
    provider: str = "openrouter"
    error: Optional[str] = None
    tokens_used: int = 0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.tokens = requests_per_minute
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            # Refill tokens based on time elapsed
            time_passed = now - self.last_refill
            self.tokens = min(self.rpm, self.tokens + time_passed * (self.rpm / 60))
            self.last_refill = now

            if self.tokens < 1:
                # Wait for a token to be available
                wait_time = (1 - self.tokens) * (60 / self.rpm)
                await asyncio.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1


class OpenRouterClient:
    """Async client for OpenRouter and OpenAI APIs with rate limiting and retries."""

    # Models that should be routed to OpenAI directly
    OPENAI_MODELS = [
        "openai/o4-mini-2025-04-16",
        "openai/o4-mini",
        "openai/gpt-4",
        "openai/gpt-4-turbo",
        "openai/gpt-4o",
        "openai/gpt-3.5-turbo",
    ]

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        openai_api_key: Optional[str] = None,
        openai_base_url: str = "https://api.openai.com/v1",
        max_concurrent: int = 10,
        requests_per_minute: int = 60,
        max_retries: int = 5,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = RateLimiter(requests_per_minute)
        self._openrouter_session: Optional[aiohttp.ClientSession] = None
        self._openai_session: Optional[aiohttp.ClientSession] = None
        self._metrics: List[APICallMetrics] = []

    def _is_openai_model(self, model: str) -> bool:
        """Check if a model should be routed to OpenAI directly."""
        return any(model.startswith(prefix.replace("openai/", "")) or model == prefix
                   for prefix in self.OPENAI_MODELS)

    def _get_openai_model_name(self, model: str) -> str:
        """Convert OpenRouter model ID to OpenAI model ID."""
        # Strip the "openai/" prefix if present
        if model.startswith("openai/"):
            return model[7:]  # Remove "openai/" prefix
        return model

    async def __aenter__(self) -> "OpenRouterClient":
        """Enter async context and create sessions."""
        # OpenRouter session
        self._openrouter_session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://physics-qa-generator.local",
                "X-Title": "Physics QA Generator",
            },
            timeout=aiohttp.ClientTimeout(total=600),  # 10 minute timeout
        )

        # OpenAI session (if API key provided)
        if self.openai_api_key:
            self._openai_session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=300),
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and close sessions."""
        if self._openrouter_session:
            await self._openrouter_session.close()
            self._openrouter_session = None
        if self._openai_session:
            await self._openai_session.close()
            self._openai_session = None

    @property
    def metrics(self) -> List[APICallMetrics]:
        """Get all API call metrics."""
        return self._metrics

    def clear_metrics(self):
        """Clear collected metrics."""
        self._metrics = []

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 16384,
        response_format: Optional[Dict[str, str]] = None,
        reasoning: bool = False,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a chat completion request with rate limiting and retries.
        Automatically routes to OpenAI or OpenRouter based on model.

        Args:
            model: The model identifier (e.g., "anthropic/claude-sonnet-4" or "openai/o4-mini-2025-04-16")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional format spec (e.g., {"type": "json_object"})
            reasoning: Enable reasoning/thinking mode if supported
            stop: Optional stop sequences

        Returns:
            The API response as a dictionary
        """
        # Determine which provider to use
        use_openai = self._is_openai_model(model)

        if use_openai:
            if not self._openai_session or not self.openai_api_key:
                raise RuntimeError(
                    f"Model {model} requires OpenAI API, but OPENAI_API_KEY is not set. "
                    "Please set the OPENAI_API_KEY environment variable."
                )
            session = self._openai_session
            base_url = self.openai_base_url
            actual_model = self._get_openai_model_name(model)
            provider = "openai"
        else:
            if not self._openrouter_session:
                raise RuntimeError("Client not initialized. Use 'async with' context manager.")
            session = self._openrouter_session
            base_url = self.base_url
            actual_model = model
            provider = "openrouter"

        # Check if this is an OpenAI o-series reasoning model
        is_o_series = use_openai and any(
            x in actual_model for x in ["o4-mini", "o1", "o3", "o4"]
        )

        payload: Dict[str, Any] = {
            "model": actual_model,
            "messages": messages,
        }

        # OpenRouter provider routing options for better reliability
        if not use_openai:
            payload["provider"] = {
                "allow_fallbacks": True,  # Use backup providers if primary fails
                "sort": "throughput",  # Prioritize faster providers
            }

        # Model-specific settings for unreliable models or rate-limited models
        is_grok = "grok" in model.lower()
        is_qwen = "qwen" in model.lower()
        # Longer backoff for Grok (unreliable) and Qwen (rate-limited)
        if is_grok:
            model_retry_multiplier = 2.0
        elif is_qwen:
            model_retry_multiplier = 1.5  # Qwen has stricter rate limits
        else:
            model_retry_multiplier = 1.0

        # OpenAI o-series models only support temperature=1 (default), so omit it
        # For other models, include the temperature parameter
        if not is_o_series:
            payload["temperature"] = temperature

        # OpenAI o-series models use max_completion_tokens, others use max_tokens
        # o-series models need much higher limits because reasoning tokens count against the limit
        if is_o_series:
            payload["max_completion_tokens"] = max(max_tokens, 65536)
        else:
            payload["max_tokens"] = max_tokens

        if response_format:
            payload["response_format"] = response_format

        # Reasoning parameter (OpenRouter-specific, not for OpenAI)
        if reasoning and not use_openai:
            payload["reasoning"] = {"enabled": True}

        if stop:
            payload["stop"] = stop

        start_time = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    await self._rate_limiter.acquire()

                    async with session.post(
                        f"{base_url}/chat/completions",
                        json=payload,
                    ) as response:
                        # Handle rate limiting with exponential backoff
                        if response.status == 429:
                            retry_after = int(response.headers.get("Retry-After", 10))
                            # Exponential backoff: base wait increases with each attempt
                            exp_backoff = min(2 ** attempt, 60)  # Cap at 60 seconds
                            base_wait = max(retry_after, exp_backoff)
                            wait_time = (base_wait + random.uniform(1, 5)) * model_retry_multiplier
                            logger.warning(
                                f"Rate limited on {model} ({provider}), waiting {wait_time:.1f}s "
                                f"(attempt {attempt + 1}/{self.max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue

                        # Handle server errors with retry
                        if response.status >= 500:
                            # Cap at 30 seconds max wait time
                            base_wait = min((2**attempt) + random.uniform(0, 1), 30)
                            wait_time = base_wait * model_retry_multiplier
                            logger.warning(
                                f"Server error {response.status} on {model} ({provider}), "
                                f"retrying in {wait_time:.1f}s"
                            )
                            await asyncio.sleep(wait_time)
                            continue

                        # Handle client errors
                        if response.status >= 400:
                            error_body = await response.text()
                            logger.error(f"API error {response.status} on {model}: {error_body}")
                            response.raise_for_status()

                        result = await response.json()

                        # Debug logging for API responses
                        logger.debug(
                            f"API Response from {model} ({provider}): "
                            f"status={response.status}, "
                            f"keys={list(result.keys())}"
                        )

                        # Validate response structure
                        choices = result.get("choices", [])
                        if not choices:
                            wait_time = (1 + attempt) * model_retry_multiplier
                            logger.warning(
                                f"Empty choices in response from {model} ({provider}), "
                                f"full response: {json.dumps(result)[:500]}, "
                                f"retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue

                        content = choices[0].get("message", {}).get("content")
                        finish_reason = choices[0].get("finish_reason", "unknown")

                        # Check for mid-stream errors from OpenRouter
                        if finish_reason == "error" or result.get("error"):
                            error_msg = result.get("error", {}).get("message", "Unknown provider error")
                            wait_time = (2 + attempt) * model_retry_multiplier
                            logger.warning(
                                f"Provider error from {model} ({provider}): {error_msg}, "
                                f"retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue

                        if content is None or content == "":
                            # Log full response for debugging empty content
                            wait_time = (1 + attempt) * model_retry_multiplier
                            logger.warning(
                                f"Empty content from {model} ({provider}), "
                                f"finish_reason={finish_reason}, "
                                f"message={choices[0].get('message', {})}, "
                                f"retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue

                        # Log successful response summary
                        logger.debug(
                            f"Success from {model}: finish_reason={finish_reason}, "
                            f"content_length={len(content)}, "
                            f"tokens={result.get('usage', {}).get('total_tokens', 'N/A')}"
                        )

                        # Record successful metric
                        end_time = time.monotonic()
                        tokens = result.get("usage", {}).get("total_tokens", 0)
                        self._metrics.append(
                            APICallMetrics(
                                model=model,
                                start_time=start_time,
                                end_time=end_time,
                                success=True,
                                provider=provider,
                                tokens_used=tokens,
                            )
                        )

                        return result

            except aiohttp.ClientError as e:
                last_error = e
                # Longer backoff for connection errors (likely server overload)
                # Cap at 60 seconds max wait time
                base_wait = min((3**attempt) + random.uniform(1, 5), 60)
                wait_time = base_wait * model_retry_multiplier
                logger.warning(
                    f"Client error on {model} ({provider}): {e}, retrying in {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(wait_time)

            except asyncio.TimeoutError as e:
                last_error = e
                # Timeout - retry with backoff
                base_wait = min((2**attempt) + random.uniform(0, 2), 30)
                wait_time = base_wait * model_retry_multiplier
                logger.warning(
                    f"Timeout on {model} ({provider}), retrying in {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(wait_time)

            except Exception as e:
                last_error = e
                # For Grok and other flaky models, retry on unexpected errors too
                if is_grok or "timeout" in str(e).lower() or "connection" in str(e).lower():
                    base_wait = min((2**attempt) + random.uniform(1, 3), 30)
                    wait_time = base_wait * model_retry_multiplier
                    logger.warning(
                        f"Error on {model} ({provider}): {e}, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error on {model} ({provider}): {e}")
                    break

        # Record failed metric
        end_time = time.monotonic()
        self._metrics.append(
            APICallMetrics(
                model=model,
                start_time=start_time,
                end_time=end_time,
                success=False,
                provider=provider,
                error=str(last_error),
            )
        )

        raise Exception(f"Max retries exceeded for {model} ({provider}): {last_error}")

    async def chat_completion_simple(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Simplified chat completion that returns just the response text.

        Args:
            model: The model identifier
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments passed to chat_completion

        Returns:
            The response text content
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.chat_completion(model=model, messages=messages, **kwargs)

        return response["choices"][0]["message"]["content"]

    async def chat_completion_json(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chat completion that returns parsed JSON.

        Args:
            model: The model identifier
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments passed to chat_completion

        Returns:
            Parsed JSON response
        """
        kwargs["response_format"] = {"type": "json_object"}

        content = await self.chat_completion_simple(
            model=model, prompt=prompt, system_prompt=system_prompt, **kwargs
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {content[:500]}...")
            raise ValueError(f"Invalid JSON response from {model}: {e}")

    async def batch_completions(
        self,
        model: str,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Run multiple completions in parallel.

        Args:
            model: The model identifier
            prompts: List of user prompts
            system_prompt: Optional system prompt for all requests
            **kwargs: Additional arguments passed to chat_completion

        Returns:
            List of response texts (or error messages for failed requests)
        """
        tasks = [
            self.chat_completion_simple(
                model=model, prompt=prompt, system_prompt=system_prompt, **kwargs
            )
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            r if isinstance(r, str) else f"ERROR: {r}"
            for r in results
        ]

    async def multi_model_completion(
        self,
        models: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        samples_per_model: int = 1,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """
        Run the same prompt across multiple models.

        Args:
            models: List of model identifiers
            prompt: The user prompt
            system_prompt: Optional system prompt
            samples_per_model: Number of samples per model
            **kwargs: Additional arguments passed to chat_completion

        Returns:
            Dict mapping model -> list of responses
        """
        results: Dict[str, List[str]] = {}

        for model in models:
            tasks = [
                self.chat_completion_simple(
                    model=model, prompt=prompt, system_prompt=system_prompt, **kwargs
                )
                for _ in range(samples_per_model)
            ]

            model_results = await asyncio.gather(*tasks, return_exceptions=True)
            results[model] = [
                r if isinstance(r, str) else f"ERROR: {r}"
                for r in model_results
            ]

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about API calls."""
        if not self._metrics:
            return {"total_calls": 0}

        successful = [m for m in self._metrics if m.success]
        failed = [m for m in self._metrics if not m.success]

        total_time = sum(m.duration_seconds for m in self._metrics)
        total_tokens = sum(m.tokens_used for m in successful)

        # Stats by provider
        openrouter_calls = [m for m in self._metrics if m.provider == "openrouter"]
        openai_calls = [m for m in self._metrics if m.provider == "openai"]

        return {
            "total_calls": len(self._metrics),
            "successful": len(successful),
            "failed": len(failed),
            "total_time_seconds": total_time,
            "avg_time_seconds": total_time / len(self._metrics) if self._metrics else 0,
            "total_tokens": total_tokens,
            "by_provider": {
                "openrouter": len(openrouter_calls),
                "openai": len(openai_calls),
            },
            "by_model": self._get_stats_by_model(),
        }

    def _get_stats_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics grouped by model."""
        by_model: Dict[str, List[APICallMetrics]] = {}

        for m in self._metrics:
            if m.model not in by_model:
                by_model[m.model] = []
            by_model[m.model].append(m)

        return {
            model: {
                "calls": len(metrics),
                "provider": metrics[0].provider if metrics else "unknown",
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
                "avg_time": sum(m.duration_seconds for m in metrics) / len(metrics),
                "total_tokens": sum(m.tokens_used for m in metrics if m.success),
            }
            for model, metrics in by_model.items()
        }
