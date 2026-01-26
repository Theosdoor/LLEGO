import asyncio
import logging
import re
from typing import List

import httpx
from openai import AsyncOpenAI, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class LLM_API:
    """
    Helper class to call the API and parse the responses.
    Compatible with OpenAI SDK v1.x
    """

    def __init__(
        self,
        model: str,
        api_type: str,
        api_base: str,
        api_version: str,
        api_key: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_tokens: List,
        system_message: str,
        with_logprobs: bool,
    ) -> None:

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_tokens = list(stop_tokens) if stop_tokens else None
        self.system_message = system_message
        self.with_logprobs = with_logprobs

        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        self.MAX_RETRIES = 4
        self.retry_backoff = [10, 30, 60]
        self.REQUEST_TIMEOUT = 30

        # Create the async client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=httpx.Timeout(self.REQUEST_TIMEOUT, connect=10.0),
        )

    def _extract_retry_time(self, exception: str, attempt_num: int) -> int:
        """Calculate exact retry time from RateLimitError exception message."""
        match = re.search(r"retry after (\d+) seconds", exception)
        if match:
            return int(match.group(1)) + 1
        else:
            return self.retry_backoff[min(attempt_num, len(self.retry_backoff) - 1)]

    async def _async_generate_without_logprobs(
        self, user_message: str, n_generations_per_prompt: int
    ):
        """Generate a response from the LLM async."""

        messages = [
            {
                "role": "system",
                "content": self.system_message,
            },
            {"role": "user", "content": user_message}
        ]

        resp = None
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n=n_generations_per_prompt,
                    stop=self.stop_tokens if self.stop_tokens else None,
                )
                break
            except RateLimitError as e:
                if attempt < self.MAX_RETRIES - 1:
                    retry_time = self._extract_retry_time(str(e), attempt)
                    logger.info(
                        f"[LLM API] Rate Limit Error. Retrying in {retry_time} seconds"
                    )
                    await asyncio.sleep(retry_time)
            except (APITimeoutError, asyncio.exceptions.TimeoutError) as e:
                if attempt < self.MAX_RETRIES - 1:
                    backoff_idx = min(attempt, len(self.retry_backoff) - 1)
                    logger.info(
                        f"[LLM API] OpenAI API timeout. Sleeping for {self.retry_backoff[backoff_idx]} seconds"
                    )
                    await asyncio.sleep(self.retry_backoff[backoff_idx])
            except Exception as e:
                logger.info(f"Error: {e}")
                raise e

        return resp

    async def _async_generate_with_logprobs(
        self, user_message: str, n_generations_per_prompt: int
    ):
        """Generate a response from the LLM async."""

        resp = None

        for attempt in range(self.MAX_RETRIES):
            try:
                # Use chat completions for all models in v1.x
                messages = [
                    {
                        "role": "system",
                        "content": self.system_message,
                    },
                    {"role": "user", "content": user_message}
                ]

                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n=n_generations_per_prompt,
                    stop=self.stop_tokens if self.stop_tokens else None,
                    logprobs=True,
                )
                break
            except RateLimitError as e:
                if attempt < self.MAX_RETRIES - 1:
                    retry_time = self._extract_retry_time(str(e), attempt)
                    logger.info(
                        f"[LLM API] Rate Limit Error. Retrying in {retry_time} seconds"
                    )
                    await asyncio.sleep(retry_time)
            except (APITimeoutError, asyncio.exceptions.TimeoutError) as e:
                if attempt < self.MAX_RETRIES - 1:
                    backoff_idx = min(attempt, len(self.retry_backoff) - 1)
                    logger.info(
                        f"[LLM API] OpenAI API timeout. Sleeping for {self.retry_backoff[backoff_idx]} seconds"
                    )
                    await asyncio.sleep(self.retry_backoff[backoff_idx])
            except Exception as e:
                raise e

        return resp

    async def _async_generate_concurrently(
        self, list_prompts, n_generations_per_prompt: int
    ):
        """
        Perform concurrent generation of responses from the LLM async.
        Returns a list of responses.
        """

        coroutines = []
        for prompt in list_prompts:
            if self.with_logprobs:
                logger.info("Generating with logprobs")
                coroutines.append(
                    self._async_generate_with_logprobs(prompt, n_generations_per_prompt)
                )

            else:
                logger.info("Generating without logprobs")
                coroutines.append(
                    self._async_generate_without_logprobs(
                        prompt, n_generations_per_prompt
                    )
                )

        tasks = [asyncio.create_task(c) for c in coroutines]

        results = [None] * len(coroutines)

        llm_response = await asyncio.gather(*tasks)

        for idx, response in enumerate(llm_response):
            if response is not None:
                resp = response
                results[idx] = resp

        return results
