"""Small demo showing how to use Firecrawl search from the CLI toolkit."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pipeline.firecrawl_search import run_firecrawl_search


def main() -> None:
    payload = run_firecrawl_search(
        query="latest MLB implied run totals",
        limit=3,
        include_content=False,
        use_sample=True,
        output_path=Path("output/firecrawl_search_sample.json"),
    )

    for idx, result in enumerate(payload.get("results", []), start=1):
        print(f"[{idx}] {result.get('title')}\n    {result.get('url')}")


if __name__ == "__main__":
    main()

