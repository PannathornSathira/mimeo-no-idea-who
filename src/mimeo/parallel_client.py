"""Thin wrapper around the official ``parallel-web`` Python SDK.

Exposes three high-level operations:

* :meth:`ParallelClient.search` — single Search API call.
* :meth:`ParallelClient.extract` — LLM-optimized full content for known URLs.
* :meth:`ParallelClient.deep_research` — create + poll a Task API run.

All calls are async and retried on transient errors via ``tenacity``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from parallel import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncParallel,
    RateLimitError,
)
from parallel.types import ExtractResponse, SearchResult, TaskRun, TaskRunResult
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
)

from .config import require_parallel_key

logger = logging.getLogger(__name__)


# Status codes that indicate a transient server-side problem and are worth
# retrying. 4xx class errors (except 408/425/429) generally won't succeed on
# retry and should surface immediately.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return False


def _retryer(attempts: int = 4) -> AsyncRetrying:
    from tenacity import retry_if_exception

    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )


class ParallelClient:
    """Facade that keeps a single ``AsyncParallel`` or uses Tavily/DDG fallbacks."""

    def __init__(self) -> None:
        import os
        provider = os.environ.get("MIMEO_SEARCH_PROVIDER", "").lower()
        parallel_key = os.environ.get("PARALLEL_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")

        if not provider:
            if parallel_key:
                provider = "parallel"
            elif tavily_key:
                provider = "tavily"
            else:
                provider = "duckduckgo"

        self.provider = provider

        if provider == "parallel":
            self._client = AsyncParallel(api_key=require_parallel_key())
        else:
            self._client = None

    async def search(
        self,
        *,
        objective: str,
        search_queries: list[str] | None = None,
        max_chars_total: int = 30_000,
        mode: str = "advanced",
    ) -> SearchResult:
        """Run one Search API call or fallback.

        ``objective`` is a natural-language description of what we want.
        ``search_queries`` are optional targeted keyword queries.
        """
        queries = search_queries or [objective]
        if self.provider == "parallel" and self._client:
            async for attempt in _retryer():
                with attempt:
                    return await self._client.search(
                        objective=objective,
                        search_queries=queries,
                        max_chars_total=max_chars_total,
                        mode=mode,  # type: ignore[arg-type]
                    )
            raise RuntimeError("unreachable")  # pragma: no cover - tenacity reraises
        elif self.provider == "tavily":
            return await self._tavily_search(queries)
        else:
            return await self._ddg_search(queries)

    async def extract(
        self,
        *,
        urls: list[str],
        objective: str | None = None,
        max_chars_total: int = 20_000,
    ) -> ExtractResponse:
        """Get LLM-optimized full content or fallback."""
        if self.provider == "parallel" and self._client:
            async for attempt in _retryer():
                with attempt:
                    return await self._client.extract(
                        urls=urls,
                        objective=objective,
                        max_chars_total=max_chars_total,
                    )
            raise RuntimeError("unreachable")  # pragma: no cover - tenacity reraises
        else:
            # Fall back to Trafilatura/Jina by returning empty results list
            return ExtractResponse(
                results=[],
                errors=[],
                extract_id="fallback_extract",
                session_id="fallback_session",
                usage=None,
                warnings=None,
            )

    async def deep_research(
        self,
        *,
        input_text: str,
        processor: str = "pro-fast",
        metadata: dict[str, Any] | None = None,
        poll_interval_s: float = 10.0,
        max_wait_s: float = 60 * 25,
    ) -> TaskRunResult:
        """Create a Task API run and poll until it completes."""
        if self.provider != "parallel" or not self._client:
            from .config import MissingCredentialError
            raise MissingCredentialError(
                "Deep research is a paid feature that requires a PARALLEL_API_KEY. "
                "Please run without the --deep-research flag to use the free search mode."
            )

        # Cast metadata values to str|int|float|bool only (the SDK restricts).
        safe_metadata: dict[str, str | float | bool] | None = None
        if metadata:
            safe_metadata = {
                k: v
                for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool))
            }

        run: TaskRun = await self._client.task_run.create(
            input=input_text,
            processor=processor,
            metadata=safe_metadata,  # type: ignore[arg-type]
        )
        run_id = run.run_id
        logger.info("Started Parallel deep-research run %s (processor=%s)", run_id, processor)

        deadline = asyncio.get_event_loop().time() + max_wait_s
        while True:
            try:
                # Use the server-side long-poll via api_timeout.
                return await self._client.task_run.result(
                    run_id, api_timeout=int(min(poll_interval_s * 3, 540))
                )
            except APITimeoutError:
                pass
            except APIStatusError as exc:
                # 408/425-style "still running" surfaces as a status error on
                # some deployments; retry until our overall deadline hits.
                if exc.status_code in (408, 425, 504):
                    logger.debug("Task %s not ready yet (%s), continuing", run_id, exc.status_code)
                else:
                    raise

            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(
                    f"Parallel deep-research run {run_id} exceeded {max_wait_s:.0f}s deadline"
                )
            await asyncio.sleep(poll_interval_s)

    async def _tavily_search(self, queries: list[str]) -> SearchResult:
        import httpx
        import os
        from parallel.types import SearchResult, WebSearchResult
        
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            from .config import MissingCredentialError
            raise MissingCredentialError("TAVILY_API_KEY is not set. Please set it in your .env file.")
            
        seen_urls = set()
        results = []
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            tasks = [
                client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": 10
                    }
                )
                for query in queries[:3]
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for resp in responses:
                if isinstance(resp, Exception):
                    logger.warning("Tavily query failed: %s", resp)
                    continue
                if resp.status_code != 200:
                    logger.warning("Tavily query failed with status %d: %s", resp.status_code, resp.text)
                    continue
                try:
                    data = resp.json()
                    for r in data.get("results") or []:
                        url = r.get("url")
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        excerpts = [r.get("content")] if r.get("content") else []
                        results.append(
                            WebSearchResult(
                                url=url,
                                title=r.get("title", ""),
                                publish_date=None,
                                excerpts=excerpts
                            )
                        )
                except Exception as e:
                    logger.warning("Failed to parse Tavily response: %s", e)
                    
        return SearchResult(
            results=results,
            search_id="tavily_search",
            session_id="tavily_session",
            usage=None,
            warnings=None
        )

    async def _ddg_search(self, queries: list[str]) -> SearchResult:
        from parallel.types import SearchResult, WebSearchResult
        
        tasks = [
            asyncio.to_thread(self._run_ddg, q)
            for q in queries[:3]
        ]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        seen_urls = set()
        results = []
        
        for res_list in results_lists:
            if isinstance(res_list, Exception):
                logger.warning("DuckDuckGo search failed: %s", res_list)
                continue
            for r in res_list:
                url = r.get("href")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                
                body = r.get("body", "")
                excerpts = [body] if body else []
                results.append(
                    WebSearchResult(
                        url=url,
                        title=r.get("title", ""),
                        publish_date=None,
                        excerpts=excerpts
                    )
                )
                
        return SearchResult(
            results=results,
            search_id="ddg_search",
            session_id="ddg_session",
            usage=None,
            warnings=None
        )

    def _run_ddg(self, query: str) -> list[dict[str, str]]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=10)
                return list(results) if results else []
        except Exception as e:
            logger.warning("DuckDuckGo search exception for query '%s': %s", query, e)
            return []
