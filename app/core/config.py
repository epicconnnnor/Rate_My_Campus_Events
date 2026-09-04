"""

Global configuration

"""

import os

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

# Which provider backs embeddings and chat. "gemini" for real work; "fake"
# swaps in a deterministic local stand-in so the indexer can be exercised
# without an API key or network.
RAG_PROVIDER = os.getenv("RAG_PROVIDER", "gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Every one of these is read as "or", not as a getenv default: an unset CI
# variable arrives as an empty string rather than as absent, and os.getenv's
# default does not catch that. Asking for a model called "" is a 404 three
# layers down.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"

# Both generation models are Flash-Lite, and that is a quota decision rather
# than a quality one. On the free tier every non-lite flash model -- 3.6 among
# them -- allows 20 requests per day, and one eval run spends about 20 on its
# own, so a single run was the whole day's budget. The lite models allow 500 a
# day at 15 a minute, which is room to actually iterate.
#
# Which of the two lite ids sits here is a load decision, and the pair swapped
# once. Answering is the heavier caller -- two calls a question, one to read it
# and one to write the answer, against the judge's one -- so it belongs on
# whichever model is quieter. 3.5-flash-lite was running at 14 of its 15 a
# minute while 3.1 sat at 6, and the eval was dying partway through the nine
# questions on 503 UNAVAILABLE, so the heavy side moved here.
CHAT_MODEL = os.getenv("CHAT_MODEL") or "gemini-3.1-flash-lite"

# The model the eval's judge reads with. Deliberately its own setting, and its
# own default, for two reasons.
#
# Quotas are per project per model, so a different model is a different daily
# allowance: the judge cannot be starved by a day of chat, and chat cannot be
# starved by a day of evals.
#
# It is also the right shape: a judge checking an answer for invented detail
# should not be the same model that wrote it.
#
# It no longer falls back to CHAT_MODEL. That fallback existed only until a
# second model was chosen, and quietly collapsing the split back into one
# bucket is the exact failure it was meant to avoid. Setting the two equal by
# hand is still allowed, and announce_models says so out loud.
#
# It holds the busier of the two lite ids because it is the lighter caller: one
# request per question, where answering costs two. See CHAT_MODEL.
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or "gemini-3.5-flash-lite"

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
# It wants tuning against real embeddings once there is a key to generate them.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.6"))

# Requests per minute for a generation call, counted per model. The Flash-Lite
# models allow 15. The non-lite ones allow 5, so this belongs with CHAT_MODEL:
# moving off Lite means moving this too.
#
# Set below the published ceiling rather than at it. Pacing to exactly 15 meant
# arriving at 14 and 15 of 15 every minute, with nothing left for what this
# pacer cannot see -- another job on the same project, a local run, a clock a
# little behind the server's. Three requests of slack is cheap: the eval's
# burst is 18 calls, which sits out one window at 12 a minute exactly as it did
# at 15.
#
# It is not a fix for the 503s. Those are capacity at the far end rather than
# our quota, and _with_retries is what rides them out; this only stops us
# adding to the crowd at the ceiling.
#
# Per model matters. Chat and judge are different models and so different
# buckets, and one eval question costs two chat calls -- one to read the
# question, one to write the answer -- which arrive as fast as the network
# allows.
CHAT_REQUESTS_PER_MINUTE = int(os.getenv("CHAT_REQUESTS_PER_MINUTE") or "12")

# How many chat questions one user may ask per calendar day, campus time.
CHAT_DAILY_LIMIT = int(os.getenv("CHAT_DAILY_LIMIT", "20"))


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
