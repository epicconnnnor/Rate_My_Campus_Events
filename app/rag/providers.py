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
from typing import Callable, Dict, List, Optional, Protocol

log = logging.getLogger("providers")

from app.rag.pacing import Pacer

from app.core.config import (
    CHAT_MODEL,
    CHAT_REQUESTS_PER_MINUTE,
    JUDGE_MODEL,
    EMBEDDING_CACHE_PATH,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    RAG_PROVIDER,
)


class ProviderNotConfigured(RuntimeError):
    """The provider cannot be built from the settings it was given.

    Kept apart from everything that can go wrong while talking to a model,
    because the two want opposite handling. A 503 is weather: say something
    calm and retry. A missing key is answerable by whoever is running the app,
    and only by them, so the message is written to be repeated out loud rather
    than swallowed.

    Six identical "something went wrong reaching the assistant" pages, all of
    them a RuntimeError raised in a constructor before any request was made, is
    what this class is for.
    """



class EmbeddingProvider(Protocol):
    """Turns text into vectors of exactly EMBEDDING_DIMENSIONS floats."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed text that is being stored and searched over."""

    def embed_query(self, text: str) -> List[float]:
        """Embed a question. Kept separate because providers tune the two
        differently, even when the model is the same."""

    def chargeable(self, texts: List[str]) -> int:
        """How many of these will actually cost a request.

        Everything that talks to the API answers len(texts). Only the disk
        cache answers anything smaller, and the indexer's pacer needs to know:
        sleeping out a quota window for documents nobody is sending is a minute
        spent on nothing."""


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
        "rag provider=%s | embedding=%s (%d dims) | chat=%s | judge=%s",
        RAG_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, CHAT_MODEL,
        JUDGE_MODEL,
    )
    log.info(
        "embedding cache: %s",
        EMBEDDING_CACHE_PATH or "off -- every embedding is a request",
    )
    if JUDGE_MODEL == CHAT_MODEL:
        log.warning(
            "judge and chat are the same model (%s): they share one daily "
            "quota, and the judge is reading its own writing", CHAT_MODEL,
        )


# =============================================================================
# RETRYING
# =============================================================================

# Pacing in the indexer is what keeps us inside the quota. This is only for the
# cases pacing cannot see -- another job on the same project, a per-model limit
# we are not tracking -- so the numbers are small on purpose. The SDK's own
# retry gives up against a sustained quota refusal, hence our own.
_SDK_ERRORS = None

QUOTA_RETRY_ATTEMPTS = 4
QUOTA_FALLBACK_DELAY_SECONDS = 20

# A 5xx is a different animal: the request was fine and the far end is simply
# busy. It carries no retryDelay, only "try again later", so the only sensible
# answer is to back off and keep asking.
#
# 10 attempts of 2s doubling, capped at 120, is about eight minutes of patience
# per call. It was one minute across 6 attempts, and one minute was not enough:
# the eval kept dying partway through the nine golden questions on 503
# UNAVAILABLE while the pipeline itself was fine -- the scuba question, which
# runs the same path end to end, passed in the same run.
#
# Eight minutes is a real cost and it is spent per call, not per run, so a
# genuine outage now fails the job about eight minutes after the first question
# that cannot get through rather than one. That is the trade: CI time against
# throwing a run away for a spike that was over in ninety seconds.
#
# The cap only starts mattering here. At 6 attempts the doubling never reached
# 32, so 60 was never a ceiling anything touched; at 10 it binds from the
# seventh attempt on, which is what stops the last few from being 4 and 8
# minutes of dead air.
SERVER_RETRY_ATTEMPTS = 10
SERVER_BACKOFF_BASE_SECONDS = 2
SERVER_BACKOFF_CAP_SECONDS = 120

# e.g. "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
_QUOTA_ID = re.compile(r"quotaId['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]",
                       re.IGNORECASE)

# e.g. "retryDelay": "14.044580725s"
_RETRY_DELAY = re.compile(r"retry[_-]?delay['\"]?\s*[:=]\s*['\"]?([0-9.]+)s",
                          re.IGNORECASE)


def _sdk_error_types() -> tuple:
    """The SDK exception types we know how to wait out.

    Gathered from whichever SDKs are actually installed, so the retry wrapper
    is not tied to one vendor and an uninstalled SDK is not an import error.
    An empty tuple is a valid `except`: it catches nothing, which is the right
    answer when neither SDK is present.
    """
    global _SDK_ERRORS
    if _SDK_ERRORS is not None:
        return _SDK_ERRORS

    found = []
    try:
        from google.genai import errors as genai_errors
        found.append(genai_errors.APIError)
    except ImportError:
        pass
    try:
        import openai
        found.append(openai.APIError)
    except ImportError:
        pass

    _SDK_ERRORS = tuple(found)
    return _SDK_ERRORS


def _status_code(error) -> Optional[int]:
    """The HTTP status behind an SDK error, whichever SDK raised it.

    google-genai puts an int on `.code`. openai puts an int on `.status_code`
    and uses `.code` for a string slug like "rate_limit_exceeded", so
    status_code is checked first and a non-numeric code is ignored rather than
    compared against 429 and quietly losing.
    """
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _is_connection_error(error) -> bool:
    """Never reached the far end at all. Worth the same patience as a 5xx."""
    try:
        import openai
    except ImportError:
        return False
    return isinstance(error, openai.APIConnectionError)


def _is_quota_error(error) -> bool:
    if _status_code(error) == 429:
        return True
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "rate_limit" in text.lower()


def _quota_ids(error) -> List[str]:
    """Which allowances the refusal says we broke."""
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        details = details.get("error", {}).get("details")

    found = []
    if isinstance(details, list):
        for entry in details:
            if not isinstance(entry, dict):
                continue
            if entry.get("@type", "").endswith("QuotaFailure"):
                for violation in entry.get("violations") or []:
                    if isinstance(violation, dict) and violation.get("quotaId"):
                        found.append(violation["quotaId"])

    return found or _QUOTA_ID.findall(str(error))


def _is_daily_quota(error) -> bool:
    """Whether the allowance we broke resets tomorrow rather than shortly.

    A per-day refusal still carries a retryDelay -- 58 seconds, for a budget
    that comes back at midnight -- so honouring it spends CI time waiting for
    something that cannot happen. Per-minute quotas are worth sitting out;
    per-day ones never are.
    """
    return any("perday" in quota_id.lower() for quota_id in _quota_ids(error))


def _is_server_error(error) -> bool:
    """Busy or broken at the far end, rather than anything about our request."""
    code = _status_code(error)
    if code is not None and 500 <= code < 600:
        return True
    if _is_connection_error(error):
        return True
    return any(
        status in str(error)
        for status in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED")
    )


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

    # openai puts it in a response header instead of the body.
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        for name in ("retry-after", "x-ratelimit-reset-requests"):
            try:
                raw = headers.get(name)
            except AttributeError:
                raw = None
            if raw:
                seconds = _seconds_from_header(str(raw))
                if seconds is not None:
                    return seconds

    match = _RETRY_DELAY.search(str(error))
    return float(match.group(1)) if match else None


def _seconds_from_header(raw: str) -> Optional[float]:
    """A retry-after that may be seconds, or may be "1.5s" / "300ms".

    Whole HTTP-date forms are not handled: nothing we talk to sends one, and
    guessing wrong here means waiting until tomorrow.
    """
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    match = re.fullmatch(r"([0-9.]+)\s*(ms|s|m)", raw, re.IGNORECASE)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2).lower()
    return {"ms": value / 1000, "s": value, "m": value * 60}[unit]


def _quota_delay(error, attempt: int) -> float:
    """Quota: the refusal says how long to wait, so wait that long.

    Guessing when we have been told is pointless, and guessing short just gets
    refused again. The extra second is slack rather than racing the window.
    """
    told = _retry_after(error)
    if told is not None:
        return told + 1
    return QUOTA_FALLBACK_DELAY_SECONDS * attempt


def _server_delay(error, attempt: int) -> float:
    """Server: nothing to go on but "later", so double and cap."""
    return min(
        SERVER_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
        SERVER_BACKOFF_CAP_SECONDS,
    )


def _with_retries(call: Callable):
    """Run `call`, riding out the two failures that are worth waiting through.

    Quota and capacity are different failure modes and get separate budgets, so
    a flapping 503 cannot spend the patience reserved for a 429. Anything else
    -- a bad model id, a malformed request -- is a real failure and raises.
    """
    retryable = _sdk_error_types()

    quota_attempt = 0
    server_attempt = 0

    while True:
        try:
            return call()
        except retryable as error:
            if _is_quota_error(error) and _is_daily_quota(error):
                log.error(
                    "daily quota exhausted (%s) -- not retrying, it does not "
                    "come back today",
                    ", ".join(_quota_ids(error)) or "quota id unknown",
                )
                raise

            if _is_quota_error(error):
                quota_attempt += 1
                if quota_attempt >= QUOTA_RETRY_ATTEMPTS:
                    raise
                kind, delay = "quota", _quota_delay(error, quota_attempt)
                attempt, budget = quota_attempt, QUOTA_RETRY_ATTEMPTS

            elif _is_server_error(error):
                server_attempt += 1
                if server_attempt >= SERVER_RETRY_ATTEMPTS:
                    raise
                kind, delay = "server", _server_delay(error, server_attempt)
                attempt, budget = server_attempt, SERVER_RETRY_ATTEMPTS

            else:
                raise

            log.warning(
                "%s error from the model; waiting %.1fs then retrying "
                "(%s attempt %d of %d)",
                kind, delay, kind, attempt, budget,
            )
            time.sleep(delay)


# =============================================================================
# PACING
# =============================================================================

# One pacer per model id, held here rather than on the provider.
#
# Quotas are per project per model, so chat and judge each need their own count
# -- and answer_question builds a fresh provider for every question, so a pacer
# kept on the instance would start a new minute each time and pace nothing.
_CHAT_PACERS: Dict[str, Pacer] = {}


def chat_pacer(model: str) -> Pacer:
    """The per-minute allowance for one model, shared by everything using it."""
    if model not in _CHAT_PACERS:
        _CHAT_PACERS[model] = Pacer(CHAT_REQUESTS_PER_MINUTE, "requests")
    return _CHAT_PACERS[model]


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
            raise ProviderNotConfigured(
                "GEMINI_API_KEY is not set. Export it, or set "
                "RAG_PROVIDER=openai with an OPENAI_API_KEY, or "
                "RAG_PROVIDER=fake for stub vectors."
            )
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._model = EMBEDDING_MODEL

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        from google.genai import types

        response = _with_retries(lambda: self._client.models.embed_content(
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

    def chargeable(self, texts: List[str]) -> int:
        return len(texts)


class GeminiChatProvider:
    def __init__(self, model: Optional[str] = None) -> None:
        from google import genai

        if not GEMINI_API_KEY:
            raise ProviderNotConfigured(
                "GEMINI_API_KEY is not set. Export it, or set "
                "RAG_PROVIDER=openai with an OPENAI_API_KEY, or "
                "RAG_PROVIDER=fake for stub vectors."
            )
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._model = model or CHAT_MODEL

    def complete(self, prompt: str) -> str:
        # Before the call, not after the refusal: a per-minute 429 is entirely
        # avoidable by arriving slower, and the retry budget is better kept for
        # the failures that are not our doing.
        chat_pacer(self._model).reserve(1)

        response = _with_retries(lambda: self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        ))
        return response.text



# =============================================================================
# OPENAI
# =============================================================================


class OpenAIEmbeddingProvider:
    def __init__(self) -> None:
        # Imported lazily so the app runs without the SDK until something
        # actually embeds.
        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise ProviderNotConfigured(
                "OPENAI_API_KEY is not set. Export it, or set "
                "RAG_PROVIDER=gemini with a GEMINI_API_KEY, or "
                "RAG_PROVIDER=fake for stub vectors."
            )
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._model = EMBEDDING_MODEL

    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = _with_retries(lambda: self._client.embeddings.create(
            model=self._model,
            input=texts,
            # text-embedding-3-small is 1536 wide already, so this asks for
            # what it would return anyway. Said out loud because the column is
            # vector(1536) and a default that ever moved would be a migration,
            # not a surprise.
            dimensions=EMBEDDING_DIMENSIONS,
        ))
        # Documented to come back in order, and each item also carries its
        # index. Sorting on what is actually there beats trusting the sentence.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [_normalize(item.embedding) for item in ordered]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        # OpenAI has no equivalent of Gemini's task_type: a document and a
        # question are embedded the same way. The disk cache still files the
        # two separately, so identical text asked both ways is stored twice --
        # a wasted entry, never a wrong vector, and not worth special-casing.
        return self._embed([text])[0]

    def chargeable(self, texts: List[str]) -> int:
        return len(texts)


class OpenAIChatProvider:
    def __init__(self, model: Optional[str] = None) -> None:
        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise ProviderNotConfigured(
                "OPENAI_API_KEY is not set. Export it, or set "
                "RAG_PROVIDER=gemini with a GEMINI_API_KEY, or "
                "RAG_PROVIDER=fake for canned replies."
            )
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._model = model or CHAT_MODEL

    def complete(self, prompt: str) -> str:
        # Before the call, not after the refusal -- same reasoning as Gemini.
        chat_pacer(self._model).reserve(1)

        response = _with_retries(lambda: self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        ))
        # A refusal or a length stop can leave content empty. Returning "" lets
        # parse_json_object log a bad extraction and fall back to a broad
        # search, which is what it already does for any other unusable reply.
        return response.choices[0].message.content or ""


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

    def chargeable(self, texts: List[str]) -> int:
        return len(texts)

    @staticmethod
    def _vector(text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [
            digest[index % len(digest)] / 255.0
            for index in range(EMBEDDING_DIMENSIONS)
        ]
        return _normalize(raw)


class FakeChatProvider:
    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or CHAT_MODEL

    def complete(self, prompt: str) -> str:
        return "fake completion"


# =============================================================================
# SELECTION
# =============================================================================

_EMBEDDING_PROVIDERS = {
    "openai": OpenAIEmbeddingProvider,
    "gemini": GeminiEmbeddingProvider,
    "fake": FakeEmbeddingProvider,
}

_CHAT_PROVIDERS = {
    "openai": OpenAIChatProvider,
    "gemini": GeminiChatProvider,
    "fake": FakeChatProvider,
}


def _build(registry: dict, name: str, **kwargs):
    announce_models()
    try:
        return registry[name](**kwargs)
    except KeyError:
        raise ProviderNotConfigured(
            f"RAG_PROVIDER is {name!r}, which is not a provider. "
            f"Expected one of {sorted(registry)}."
        ) from None


def get_embedding_provider(name: str = None) -> EmbeddingProvider:
    name = name or RAG_PROVIDER
    provider = _build(_EMBEDDING_PROVIDERS, name)

    # The fake provider is deliberately not wrapped. Its vectors cost nothing,
    # so there is nothing to save, and caching them would leave stub vectors on
    # disk for a real run to find.
    if EMBEDDING_CACHE_PATH and name != "fake":
        from app.rag.embedding_cache import CachedEmbeddingProvider

        provider = CachedEmbeddingProvider(
            provider, EMBEDDING_CACHE_PATH, EMBEDDING_MODEL,
            EMBEDDING_DIMENSIONS,
        )

    return provider


def get_chat_provider(name: str = None) -> ChatProvider:
    """The model that answers questions."""
    return _build(_CHAT_PROVIDERS, name or RAG_PROVIDER, model=CHAT_MODEL)


def get_judge_provider(name: str = None) -> ChatProvider:
    """The model that checks an answer against what retrieval found.

    Separate from the answering model on purpose: its own quota bucket, and a
    reader that did not write what it is marking.
    """
    return _build(_CHAT_PROVIDERS, name or RAG_PROVIDER, model=JUDGE_MODEL)
