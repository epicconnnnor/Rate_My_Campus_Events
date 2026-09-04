"""
Tests for the disk cache in front of the embedding provider.

The point of the cache is that the eval stops re-embedding 102 unchanged events
on every run, so the thing worth asserting is not that vectors come back -- it
is that the provider underneath is not called. Every test here counts calls.

The other half is safety. The cache is content-addressed, and these check that
it cannot serve a vector that belongs to different text, a different model, a
different width, or the other task type.
"""

import json

import pytest

from app.rag.embedding_cache import CachedEmbeddingProvider

MODEL = "gemini-embedding-001"
DIMENSIONS = 4


class CountingProvider:
    """Deterministic vectors, and a tally of how much it was asked for."""

    def __init__(self, dimensions=DIMENSIONS):
        self.dimensions = dimensions
        self.documents = []
        self.queries = []

    def _vector(self, text):
        return [float(len(text)), 1.0, 2.0, 3.0][:self.dimensions]

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return self._vector(text)

    def chargeable(self, texts):
        return len(texts)


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "embeddings"


def build(cache_dir, inner, model=MODEL, dimensions=DIMENSIONS):
    return CachedEmbeddingProvider(inner, cache_dir, model, dimensions)


# =============================================================================
# NOT SPENDING THE QUOTA TWICE
# =============================================================================


def test_the_second_run_sends_nothing(cache_dir):
    """The whole point: a fresh provider over the same directory, which is what
    the next CI run is, does not embed anything again."""
    events = ["one event", "another event", "a third"]

    first = CountingProvider()
    build(cache_dir, first).embed_documents(events)
    assert first.documents == events

    second = CountingProvider()
    build(cache_dir, second).embed_documents(events)
    assert second.documents == []


def test_the_cached_vectors_are_the_ones_that_were_computed(cache_dir):
    events = ["one event", "another event"]

    live = build(cache_dir, CountingProvider()).embed_documents(events)
    cached = build(cache_dir, CountingProvider()).embed_documents(events)

    assert cached == live


def test_only_the_new_documents_are_sent(cache_dir):
    """The case a refreshed fixture hits: most of the file is unchanged."""
    build(cache_dir, CountingProvider()).embed_documents(["old", "older"])

    inner = CountingProvider()
    vectors = build(cache_dir, inner).embed_documents(["old", "new", "older"])

    assert inner.documents == ["new"]
    assert len(vectors) == 3
    assert all(vector is not None for vector in vectors)


def test_the_vectors_come_back_in_the_order_they_were_asked_for(cache_dir):
    """A partial hit rebuilds the list from two sources, so the order is worth
    pinning: the indexer zips it against the events."""
    build(cache_dir, CountingProvider()).embed_documents(["bb"])

    provider = build(cache_dir, CountingProvider())
    vectors = provider.embed_documents(["a", "bb", "ccc", "dddd"])

    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0]


def test_queries_are_cached_too(cache_dir):
    """The golden questions are asked verbatim every run, so their query
    vectors are as reusable as the documents."""
    build(cache_dir, CountingProvider()).embed_query("what is on friday?")

    inner = CountingProvider()
    build(cache_dir, inner).embed_query("what is on friday?")
    assert inner.queries == []


# =============================================================================
# PACING
# =============================================================================


def test_chargeable_counts_only_what_will_be_sent(cache_dir):
    """The indexer paces on this. Counting a cached document would buy a minute
    of sleeping for a request nobody is making."""
    build(cache_dir, CountingProvider()).embed_documents(["old"])

    provider = build(cache_dir, CountingProvider())
    assert provider.chargeable(["old", "new"]) == 1
    assert provider.chargeable(["old"]) == 0
    assert provider.chargeable(["new", "newer"]) == 2


# =============================================================================
# IT CANNOT SERVE THE WRONG VECTOR
# =============================================================================


def test_a_different_model_does_not_hit(cache_dir):
    build(cache_dir, CountingProvider()).embed_documents(["an event"])

    inner = CountingProvider()
    build(cache_dir, inner, model="some-other-model").embed_documents(
        ["an event"]
    )
    assert inner.documents == ["an event"]


def test_a_different_width_does_not_hit(cache_dir):
    """EMBEDDING_DIMENSIONS is a schema decision. A cache that answered across
    a change in it would hand the database vectors of the wrong length."""
    build(cache_dir, CountingProvider()).embed_documents(["an event"])

    inner = CountingProvider(dimensions=3)
    build(cache_dir, inner, dimensions=3).embed_documents(["an event"])
    assert inner.documents == ["an event"]


def test_a_document_and_a_query_do_not_share_an_entry(cache_dir):
    """The same model embeds the two differently, so text that appears as both
    must not be answered from the other one's vector."""
    text = "free things next week"
    build(cache_dir, CountingProvider()).embed_documents([text])

    inner = CountingProvider()
    build(cache_dir, inner).embed_query(text)
    assert inner.queries == [text]


def test_an_entry_holding_other_text_is_ignored(cache_dir):
    """The file is named after a hash of its own text. If the two ever disagree
    the entry is wrong, whatever the reason, and re-embedding is the only safe
    answer."""
    build(cache_dir, CountingProvider()).embed_documents(["an event"])

    entry = next(cache_dir.glob("*.json"))
    written = json.loads(entry.read_text(encoding="utf-8"))
    written["text"] = "something else entirely"
    entry.write_text(json.dumps(written), encoding="utf-8")

    inner = CountingProvider()
    build(cache_dir, inner).embed_documents(["an event"])
    assert inner.documents == ["an event"]


def test_an_entry_of_the_wrong_length_is_ignored(cache_dir):
    build(cache_dir, CountingProvider()).embed_documents(["an event"])

    entry = next(cache_dir.glob("*.json"))
    written = json.loads(entry.read_text(encoding="utf-8"))
    written["vector"] = [0.0, 0.0]
    entry.write_text(json.dumps(written), encoding="utf-8")

    inner = CountingProvider()
    build(cache_dir, inner).embed_documents(["an event"])
    assert inner.documents == ["an event"]


# =============================================================================
# A BROKEN CACHE IS SLOW, NOT FATAL
# =============================================================================


def test_a_truncated_file_is_re_embedded_rather_than_raising(cache_dir):
    """A run killed mid-write used to be able to leave one of these. It cannot
    any more -- writes are renamed into place -- but reading one must still be
    survivable."""
    build(cache_dir, CountingProvider()).embed_documents(["an event"])
    next(cache_dir.glob("*.json")).write_text('{"text": "an ev',
                                              encoding="utf-8")

    inner = CountingProvider()
    vectors = build(cache_dir, inner).embed_documents(["an event"])
    assert inner.documents == ["an event"]
    assert vectors == [inner._vector("an event")]


def test_a_directory_that_cannot_be_written_still_embeds(tmp_path):
    """Somewhere unwritable means no cache, which is exactly how the app runs
    anyway. It must not be the reason a run fails."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("this is a file", encoding="utf-8")

    inner = CountingProvider()
    provider = build(blocked / "embeddings", inner)

    assert provider.embed_documents(["an event"]) == [inner._vector("an event")]
    assert inner.documents == ["an event"]


def test_an_empty_directory_is_all_misses(cache_dir):
    """Nothing on disk is the first run, not an error."""
    provider = build(cache_dir, CountingProvider())
    assert provider.chargeable(["a", "b"]) == 2
    provider.embed_documents(["a", "b"])
    assert (provider.hits, provider.misses) == (0, 2)
