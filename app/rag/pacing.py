"""
Staying under a per-minute quota by waiting, rather than by being refused.

Google publishes two numbers per model: requests per minute and requests per
day. The daily one there is nothing to do about but spend it carefully. The
per-minute one is only a matter of not arriving too fast, so this waits.

Waiting has to happen before the request rather than after the refusal. The
SDK's own retry exhausted itself against a sustained 429 rather than riding it
out, which is how the first eval run died on document 101 of 102.

Both the embedding backfill and the chat calls use this, against different
limits and different windows. Quotas are per project per model, so a pacer
belongs to one model and counts only what is sent to it.
"""

import logging
import time

log = logging.getLogger("pacing")

# Slack, so a slow clock or another job on the same project cannot land us
# exactly on the line.
HEADROOM_SECONDS = 2

WINDOW_SECONDS = 60


class Pacer:
    """A rolling minute, and a sleep when the next call would not fit in it.

    Deliberately not clever. Each of these runs inside one job, so sleeping out
    the rest of the minute costs less than anything smarter would, and it is
    obvious what it does when it appears in a CI log.
    """

    def __init__(self, limit: int, unit: str = "requests") -> None:
        self.limit = limit
        self.unit = unit
        self._window_started = time.monotonic()
        self._spent = 0

    def reserve(self, count: int) -> None:
        """Block until `count` more can be sent without breaking the quota."""
        elapsed = time.monotonic() - self._window_started
        if elapsed >= WINDOW_SECONDS:
            self._window_started = time.monotonic()
            self._spent = 0
            elapsed = 0

        if self._spent + count > self.limit:
            wait = max(0.0, WINDOW_SECONDS - elapsed) + HEADROOM_SECONDS
            log.info(
                "%d/%d %s used this minute; waiting %.0fs for the quota "
                "window to roll", self._spent, self.limit, self.unit, wait,
            )
            time.sleep(wait)
            self._window_started = time.monotonic()
            self._spent = 0

        self._spent += count
