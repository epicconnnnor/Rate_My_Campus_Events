"""
The seam between this app and whoever is doing the embedding and the talking.

Nothing outside this module names a model or an SDK. Swapping Gemini out means
writing another pair of classes here and pointing RAG_PROVIDER at them.

    from app.rag.providers import get_embedding_provider

    provider = get_embedding_provider()
    vectors = provider.embed_documents(["..."])
"""

import hashlib
import math
from typing import List, Protocol

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

        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
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
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
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
