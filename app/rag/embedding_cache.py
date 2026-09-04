"""
A disk cache in front of whoever is doing the embedding.

The eval re-embeds the same 102 frozen events on every run. They do not change
-- that is the entire point of freezing them -- so every run after the first
spent 102 of the day's 1000 free embedding requests recomputing vectors it had
already computed. This keeps them between runs.

Off unless EMBEDDING_CACHE_PATH names a directory. The app leaves it unset and
so never sees this at all; the eval sets it, and CI carries the directory from
one run to the next.

Entries are content-addressed -- the key is a hash of the model, the width, the
task type and the text itself -- so nothing in here can go stale. Changing the
model or EMBEDDING_DIMENSIONS does not invalidate the cache, it simply misses
everything, which is the same thing without a step to remember.

Nothing here is allowed to fail a run. An unreadable entry, a directory that
cannot be written, a half-written file: all of them mean "embed it again",
never an exception.
"""

import hashlib
import json
import logging
import os
import pathlib
from typing import List, Optional

log = logging.getLogger("embedding_cache")

DOCUMENT = "RETRIEVAL_DOCUMENT"
QUERY = "RETRIEVAL_QUERY"


def cache_key(model: str, dimensions: int, task_type: str, text: str) -> str:
    """What identifies a vector.

    Everything that changes the vector goes in. Documents and queries are
    embedded differently by the same model, so the task type is part of it too
    -- otherwise a question that happens to read like an event description
    would be served the wrong vector.
    """
    digest = hashlib.sha256()
    for part in (model, str(dimensions), task_type, text):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class CachedEmbeddingProvider:
    """Wraps a real provider and answers from disk where it can."""

    def __init__(self, inner, directory, model: str, dimensions: int) -> None:
        self._inner = inner
        self._directory = pathlib.Path(directory)
        self._model = model
        self._dimensions = dimensions
        self.hits = 0
        self.misses = 0

    # -- the provider interface ---------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = [self._read(DOCUMENT, text) for text in texts]
        missing = [index for index, vector in enumerate(vectors)
                   if vector is None]

        self.hits += len(texts) - len(missing)
        self.misses += len(missing)

        if missing:
            fresh = self._inner.embed_documents([texts[index]
                                                 for index in missing])
            for index, vector in zip(missing, fresh):
                vectors[index] = vector
                self._write(DOCUMENT, texts[index], vector)

        log.info(
            "embedding cache: %d of %d documents answered from disk "
            "(%d hits, %d misses so far)",
            len(texts) - len(missing), len(texts), self.hits, self.misses,
        )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        cached = self._read(QUERY, text)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        vector = self._inner.embed_query(text)
        self._write(QUERY, text, vector)
        return vector

    def chargeable(self, texts: List[str]) -> int:
        """How many of these would actually cost a request.

        The indexer's pacer sleeps out the rest of the minute once the quota is
        spent. A cached document costs nothing, so counting it would buy a
        minute of waiting for work that is not happening.
        """
        return sum(1 for text in texts if self._read(DOCUMENT, text) is None)

    # -- the disk -----------------------------------------------------------

    def _path(self, task_type: str, text: str) -> pathlib.Path:
        key = cache_key(self._model, self._dimensions, task_type, text)
        return self._directory / f"{key}.json"

    def _read(self, task_type: str, text: str) -> Optional[List[float]]:
        path = self._path(task_type, text)

        try:
            with path.open(encoding="utf-8") as handle:
                entry = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            log.warning("cache entry %s is unreadable (%s); re-embedding",
                        path.name, error)
            return None

        # The file names itself after a hash of its own text, so this is one
        # comparison that rules out the only way a content-addressed cache can
        # quietly lie.
        if entry.get("text") != text:
            log.warning("cache entry %s holds different text; re-embedding",
                        path.name)
            return None

        vector = entry.get("vector")
        if not isinstance(vector, list) or len(vector) != self._dimensions:
            log.warning("cache entry %s is not %d floats; re-embedding",
                        path.name, self._dimensions)
            return None

        return vector

    def _write(self, task_type: str, text: str, vector: List[float]) -> None:
        path = self._path(task_type, text)

        # Written beside the real name and renamed over it, so a run that dies
        # mid-write leaves no half-file for the next one to read.
        temporary = path.parent / f"{path.name}.{os.getpid()}.tmp"

        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump({
                    "model": self._model,
                    "dimensions": self._dimensions,
                    "task_type": task_type,
                    "text": text,
                    "vector": vector,
                }, handle)
            os.replace(temporary, path)
        except OSError as error:
            # A cache that cannot be written is slow, not broken.
            log.warning("could not cache %s: %s", path.name, error)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
