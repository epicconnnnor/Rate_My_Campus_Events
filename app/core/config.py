"""

Global configuration

"""

import os
from datetime import date

# =============================================================================
# JWT CONFIGURATION
# =============================================================================

# Signs the login tokens and the OAuth session cookie.
#
# No default. A signing key committed to the source is not a secret, and the
# one that used to sit here is in this repository's history, so anyone who has
# ever seen the repo can mint a valid token with it.
SECRET_KEY = os.getenv("SECRET_KEY")

SECRET_KEY_MISSING = (
    "SECRET_KEY is not set.\n\n"
    "It signs the login tokens and the OAuth session cookie, so the app will "
    "not start without one. Generate a value:\n\n"
    '    python -c "import secrets; print(secrets.token_hex(32))"\n\n'
    "then put it in .env, export it, or set it as a secret wherever this is "
    "deployed. See .env.example."
)


def require_secret_key() -> str:
    """The key, or a refusal to continue without one.

    Checked here rather than at import so that the migrations and the ingest,
    which sign nothing, still run without it.
    """
    if not SECRET_KEY:
        raise RuntimeError(SECRET_KEY_MISSING)
    return SECRET_KEY

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Browser sessions are kept out of URLs. Leave this true in production; set it
# to false only for a local HTTP development server.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() in {"1", "true", "yes"}
EVENT_ADMIN_EMAILS = {
    email.strip().casefold()
    for email in os.getenv("EVENT_ADMIN_EMAILS", "").split(",")
    if email.strip()
}
EVENT_SUBMISSION_DAILY_LIMIT = int(os.getenv("EVENT_SUBMISSION_DAILY_LIMIT", "3"))

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# PostgreSQL connection configuration
DATABASE_USER = os.getenv("DATABASE_USER", "app")
DATABASE_PASS = os.getenv("DATABASE_PASS", "app")
DATABASE_HOST = os.getenv("DATABASE_HOST", "dev_pg")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "db")

DATABASE_URL = f"postgresql+psycopg2://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

APP_TITLE = "RateMyCampusEvents Demo"

APP_DESCRIPTION = "An app for students/organizers to view, rate, and comment on campus events."


# =============================================================================
# RAG CONFIGURATION
# =============================================================================

# Which provider backs embeddings and chat. "openai" and "gemini" are both
# real; "fake" swaps in a deterministic local stand-in so the indexer can be
# exercised without an API key or network.
#
# OpenAI is the default. Gemini is kept whole and working -- switching back is
# RAG_PROVIDER=gemini and a key, with no other setting to remember, which is
# the entire reason the model ids and the rate limits below are looked up per
# provider rather than written down once.
RAG_PROVIDER = os.getenv("RAG_PROVIDER") or "openai"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Every one of these is read as "or", not as a getenv default: an unset CI
# variable arrives as an empty string rather than as absent, and os.getenv's
# default does not catch that. Asking for a model called "" is a 404 three
# layers down.
#
# Defaults are per provider, because a Gemini model id means nothing to OpenAI
# and the other way round. An id set in the environment wins over both, so
# CHAT_MODEL=gpt-4o still works with RAG_PROVIDER=openai.
DEFAULT_MODELS = {
    "openai": {
        "chat": "gpt-4o-mini",
        # A different model, for the reason below. Not a different quota:
        # OpenAI meters the organisation rather than the model, so the split
        # here buys independence of judgement and nothing else. That half of
        # the argument was always the more important one.
        "judge": "gpt-4.1-mini",
        # 1536 natively, which is exactly the width migration 0004 wrote into
        # the column. Requested explicitly below all the same, so a change to
        # what the model returns by default cannot quietly widen the vector.
        "embedding": "text-embedding-3-small",
    },
    "gemini": {
        # Both Flash-Lite, and that is a quota decision rather than a quality
        # one. On the free tier every non-lite flash model allows 20 requests
        # per day, and one eval run spends about that on its own. Answering is
        # the heavier caller -- two calls a question against the judge's one --
        # so it sits on whichever of the two is quieter.
        "chat": "gemini-3.1-flash-lite",
        "judge": "gemini-3.5-flash-lite",
        "embedding": "gemini-embedding-001",
    },
}

# "fake" borrows OpenAI's names. It never sends them anywhere; they exist so
# that a log line from a fake run still reads like a real one.
_models = DEFAULT_MODELS.get(RAG_PROVIDER, DEFAULT_MODELS["openai"])

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or _models["embedding"]

CHAT_MODEL = os.getenv("CHAT_MODEL") or _models["chat"]

# The model the eval's judge reads with. Deliberately its own setting, and its
# own default.
#
# It is the right shape: a judge checking an answer for invented detail should
# not be the same model that wrote it. On Gemini it also buys a second daily
# allowance, since quotas there are per project per model.
#
# It does not fall back to CHAT_MODEL. That fallback existed only until a
# second model was chosen, and quietly collapsing the split back into one
# bucket is the exact failure it was meant to avoid. Setting the two equal by
# hand is still allowed, and announce_models says so out loud.
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or _models["judge"]

# Baked into the vector column type by migration 0004. Changing it is a schema
# change plus a full re-index, not a config tweak -- keep the two in step.
#
# 1536 rather than the model's native 3072 because pgvector cannot index a
# vector wider than 2000 dimensions, and gemini-embedding-001 is trained so
# that a truncated prefix stays useful (Matryoshka).
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# Where to keep embedding vectors between runs, if anywhere. Empty means no
# cache, which is how the app runs: it embeds a question once and never sees it
# again, so there is nothing to reuse.
#
# The eval is the opposite. It re-embeds the same 102 frozen events on every
# run, which was spending a tenth of the day's free embedding requests to
# recompute vectors that had not changed. It sets this; see
# app/rag/embedding_cache.py.
EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", "").strip()

# Retrieval tuning. MAX_DISTANCE is a cosine distance, so 0 is identical and 2
# is opposite; anything above the cutoff is treated as not really an answer.
#
# 0.70 is measured rather than guessed, against text-embedding-3-small and the
# 102 frozen events. It was 0.6, which was measured against Gemini's vectors --
# a different model puts related things at a different distance, and 0.6 under
# this one rejected "Libraries Outreach Series" for the query "library".
#
# The number belongs to the embedding model, not to the app. Changing
# EMBEDDING_MODEL means measuring this again: too low and every question falls
# through to the near-miss branch, too high and unrelated events are offered as
# answers.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.70"))

# Requests per minute, per provider. Both numbers are deliberately under the
# published ceiling rather than on it: pacing to exactly the limit means
# arriving at the last request of every minute with nothing left for what one
# process cannot see -- a second job on the same key, a local run, a clock a
# little behind the server's.
#
# Gemini's Flash-Lite models publish 15 a minute, hence 12. OpenAI's limits are
# far higher -- gpt-4o-mini is in the hundreds per minute even on the smallest
# paid tier -- so 60 is not really a constraint there, it is a brake that keeps
# a runaway loop from spending real money before anybody notices. Raise it with
# CHAT_REQUESTS_PER_MINUTE if a run is actually waiting on this.
#
# Per model matters. Chat and judge are different models and so, on Gemini,
# different buckets, and one eval question costs two chat calls -- one to read
# the question, one to write the answer -- which arrive as fast as the network
# allows.
CHAT_RPM_BY_PROVIDER = {"openai": 60, "gemini": 12, "fake": 100000}

CHAT_REQUESTS_PER_MINUTE = int(
    os.getenv("CHAT_REQUESTS_PER_MINUTE")
    or CHAT_RPM_BY_PROVIDER.get(RAG_PROVIDER, 12)
)

# Documents per minute for embedding, per provider.
#
# Counted in documents rather than calls because that is how Google meters it:
# one embed_content request per document, so a batch of 32 spends 32 of the
# free tier's 100 a minute.
#
# OpenAI meters the embeddings endpoint by request and by token instead, and a
# batch of 32 inputs is one request there, so counting documents over-paces --
# on purpose. 1000 a minute is far below what the endpoint allows and still
# fast enough that the 102 fixture events never wait at all.
EMBEDDING_DPM_BY_PROVIDER = {"openai": 1000, "gemini": 100, "fake": 100000}

EMBEDDING_DOCUMENTS_PER_MINUTE = int(
    os.getenv("EMBEDDING_DOCUMENTS_PER_MINUTE")
    or EMBEDDING_DPM_BY_PROVIDER.get(RAG_PROVIDER, 100)
)

# How many chat questions one user may ask per calendar day, campus time.
CHAT_DAILY_LIMIT = int(os.getenv("CHAT_DAILY_LIMIT", "20"))


def _parse_demo_date(value):
    """DEMO_DATE, or nothing, and a clear complaint in between.

    Silently ignoring a typo here is the cruellest option: the app comes up,
    every question returns nothing, and the calendar looks broken rather than
    misconfigured.
    """
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise RuntimeError(
            f"DEMO_DATE={value!r} is not a date. Use YYYY-MM-DD, for example "
            "DEMO_DATE=2026-09-02, which is the first day of the frozen demo "
            "events in app/test/fixtures/events.json."
        ) from None


# What the app calls "today" when it goes looking for events. Unset means the
# real date, which is what anything deployed wants.
#
# A local run wants the opposite. The frozen events run from 2026-09-02 to
# 2026-11-12, so a clone started outside that window searches a calendar it has
# already walked past, every question comes back empty, and the bot looks
# broken rather than the database looking stale. Pinning this to a day inside
# the window gives a demo something to find -- the same trick EVAL_TODAY plays
# for the eval, for the same reason.
#
# It moves the calendar only. The daily question quota still counts real days,
# because that is about the person asking, not about what is on.
DEMO_DATE = _parse_demo_date(os.getenv("DEMO_DATE"))


# =============================================================================
# OAUTH
# =============================================================================

# Google and GitHub only. Each provider is registered only if both halves of
# its pair are present, so the app runs fine with neither configured -- the
# buttons simply do not appear.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
