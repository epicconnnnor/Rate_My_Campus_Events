"""
Embed every event that needs it.

    python -m app.rag.backfill

Safe to run repeatedly: an event whose document has not changed since it was
last embedded is skipped, so a second run costs nothing.
"""

import argparse
import logging
from typing import List, Optional

from app.rag.indexer import index_events
from app.rag.providers import get_embedding_provider

log = logging.getLogger("backfill")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Embed every event whose document has changed."
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="override RAG_PROVIDER for this run, e.g. 'fake'",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stats = index_events(provider=get_embedding_provider(args.provider))
    log.info(
        "done: %d embedded, %d already current",
        stats["embedded"],
        stats["skipped"],
    )


if __name__ == "__main__":
    main()
