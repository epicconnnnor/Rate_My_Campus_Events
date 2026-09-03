"""
The seam between this app and whoever is doing the embedding and the talking.

Nothing outside this module names a model or an SDK. Swapping Gemini out means
writing another pair of classes here and pointing RAG_PROVIDER at them.

    from app.rag.providers import get_embedding_provider

    provider = get_embedding_provider()
    vectors = provider.embed_documents(["..."])
"""

import hashlib
import logging
import math
import re
import time
from typing import Callable, List, Optional, Protocol

log = logging.getLogger("providers")

from app.core.config import (
    CHAT_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    RAG_PROVIDER,
)


class EmbeddingProvider(Protocol):
    """Turns text into vectors of exactly EMBEDDING_DIMENSIONS floats."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed text that is being stored and searched over."""

    def embed_query(self, text: str) -> List[float]:
        """Embed a question. Kept separate because providers tune the two
        differently, even when the model is the same."""


class ChatProvider(Protocol):
    def complete(self, prompt: str) -> str:
        """One prompt in, one answer out. No history, no tools."""


# =============================================================================
# WHAT WE ARE TALKING TO
# =============================================================================

_ANNOUNCED = False


def announce_models() -> None:
    """Say which model ids we resolved, once per process.

    A retired model is invisible until a key is fresh: gemini-2.5-flash kept
    working for existing keys and answered a rotated one with 404 NOT_FOUND,
    surfacing three layers down inside a retrieval call rather than as anything
    about models. Naming them up front means the next one identifies itself.
    """
    global _ANNOUNCED
    if _ANNOUNCED:
        return
    _ANNOUNCED = True

    if RAG_PROVIDER == "fake":
        log.info("rag provider=fake -- no model is being called")
        return

    log.info(
        "rag provider=%s | embedding=%s (%d dims) | chat=%s",
        RAG_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, CHAT_MODEL,
    )


# =============================================================================
# QUOTA BACKSTOP
# =============================================================================

# Pacing in the indexer is what keeps us inside the quota. This is only for the
# cases pacing cannot see -- another job on the same project, a per-model limit
# we are not tracking -- so the numbers are small on purpose. The SDK's own
# retry gives up against a sustained quota refusal, hence our own.
QUOTA_RETRY_ATTEMPTS = 4
QUOTA_FALLBACK_DELAY_SECONDS = 20

# e.g. "retryDelay": "14.044580725s"
_RETRY_DELAY = re.compile(r"retry[_-]?delay['\"]?\s*[:=]\s*['\"]?([0-9.]+)s",
                          re.IGNORECASE)


def _is_quota_error(error) -> bool:
    if getattr(error, "code", None) == 429:
        return True
    return "RESOURCE_EXHAUSTED" in str(error)


def _retry_after(error) -> Optional[float]:
    """However long the server asked us to wait, if it said.

    A quota refusal carries RetryInfo. Guessing when we have been told is
    pointless, and guessing short gets us refused again.
    """
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        details = details.get("error", {}).get("details")
    if isinstance(details, list):
        for entry in details:
            if not isinstance(entry, dict):
                continue
            if entry.get("@type", "").endswith("RetryInfo"):
                match = _RETRY_DELAY.search(f"retryDelay: {entry.get('retryDelay')}")
                if match:
                    return float(match.group(1))

    match = _RETRY_DELAY.search(str(error))
    return float(match.group(1)) if match else None


def _with_quota_retry(call: Callable):
    """Run `call`, riding out a 429 rather than failing the whole backfill."""
    from google.genai import errors

    for attempt in range(1, QUOTA_RETRY_ATTEMPTS + 1):
        try:
            return call()
        except errors.APIError as error:
            if not _is_quota_error(error) or attempt == QUOTA_RETRY_ATTEMPTS:
                raise
            delay = _retry_after(error)
            if delay is None:
                delay = QUOTA_FALLBACK_DELAY_SECONDS * attempt
            delay += 1  # a second of slack rather than racing the window
            log.warning(
                "quota refused the request; waiting %.1fs then retrying "
                "(attempt %d of %d)", delay, attempt, QUOTA_RETRY_ATTEMPTS,
            )
            time.sleep(delay)


# =============================================================================
# GEMINI
# =============================================================================


def _normalize(vector: List[float]) -> List[float]:
    """Scale to unit length.

    Gemini's embeddings only come back normalized at their full width; asking
    for a narrower one means normalizing it again, which matters because
    cosine distance assumes it.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class GeminiEmbeddingProvider:
    def __init__(self) -> None:
        # Imported lazily so the app runs without the SDK until something
        # actually embeds.
        from google import genai

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; export it or set RAG_PROVIDER=fake"
            )
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._model = EMBEDDING_MODEL

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        from google.genai import types

        response = _with_quota_retry(lambda: self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        ))
        return [_normalize(item.values) for item in response.embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiChatProvider:
    def __init__(self) -> None:
        from google import genai

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; export it or set RAG_PROVIDER=fake"
            )
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._model = CHAT_MODEL

    def complete(self, prompt: str) -> str:
        response = _with_quota_retry(lambda: self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        ))
        return response.text


# =============================================================================
# FAKE
# =============================================================================


class FakeEmbeddingProvider:
    """Deterministic vectors derived from the text itself.

    Not an approximation of anything -- similar texts do not get similar
    vectors. It exists so the indexer's bookkeeping (what changed, what got
    skipped) can be exercised without a network or a key.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [
            digest[index % len(digest)] / 255.0
            for index in range(EMBEDDING_DIMENSIONS)
        ]
        return _normalize(raw)


class FakeChatProvider:
    def complete(self, prompt: str) -> str:
        return "fake completion"


# =============================================================================
# SELECTION
# =============================================================================

_EMBEDDING_PROVIDERS = {
    "gemini": GeminiEmbeddingProvider,
    "fake": FakeEmbeddingProvider,
}

_CHAT_PROVIDERS = {
    "gemini": GeminiChatProvider,
    "fake": FakeChatProvider,
}


def _build(registry: dict, name: str):
    announce_models()
    try:
        return registry[name]()
    except KeyError:
        raise ValueError(
            f"unknown RAG_PROVIDER {name!r}; expected one of {sorted(registry)}"
        ) from None


def get_embedding_provider(name: str = None) -> EmbeddingProvider:
    return _build(_EMBEDDING_PROVIDERS, name or RAG_PROVIDER)


def get_chat_provider(name: str = None) -> ChatProvider:
    return _build(_CHAT_PROVIDERS, name or RAG_PROVIDER)
