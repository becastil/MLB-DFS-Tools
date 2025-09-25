"""Utility helpers for the Firecrawl search feature."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional


try:
    from .data_sources.firecrawl_client import get_firecrawl_client
except Exception:  # pragma: no cover - firecrawl package may be absent
    get_firecrawl_client = None  # type: ignore

from .sample_data import data_path


LOGGER = logging.getLogger(__name__)


def run_firecrawl_search(
    query: str,
    limit: int = 5,
    include_content: bool = False,
    use_sample: bool = False,
    output_path: Optional[Path | str] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """Execute a Firecrawl search query and optionally persist the results.

    Args:
        query: Search string to submit to Firecrawl Search.
        limit: Maximum number of hits to return.
        include_content: Whether Firecrawl should fetch the body content for each
            hit. Useful when you plan to feed the results directly into an LLM but
            consumes more credits.
        use_sample: Force the helper to return a bundled static response. Handy for
            local development when Firecrawl credentials are unavailable.
        output_path: Optional JSON destination for the raw search payload.

    Returns:
        Dictionary containing the search results.
    """

    payload: Dict[str, List[Dict[str, object]]]

    if not use_sample and get_firecrawl_client is not None:
        try:
            client = get_firecrawl_client()
            payload = client.search_web(
                query=query,
                limit=limit,
                include_content=include_content,
            )
        except Exception as exc:  # pragma: no cover - remote I/O
            LOGGER.warning("Firecrawl search failed (%s); falling back to sample data", exc)
            use_sample = True
        else:
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
            return payload

    with data_path("firecrawl_search_results.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return payload


__all__ = ["run_firecrawl_search"]

