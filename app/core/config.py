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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# Baked into the vector column type by migration 0004. Changing it is a schema
# change plus a full re-index, not a config tweak -- keep the two in step.
#
# 1536 rather than the model's native 3072 because pgvector cannot index a
# vector wider than 2000 dimensions, and gemini-embedding-001 is trained so
# that a truncated prefix stays useful (Matryoshka).
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# Retrieval tuning. MAX_DISTANCE is a cosine distance, so 0 is identical and 2
# is opposite; anything above the cutoff is treated as not really an answer.
# It wants tuning against real embeddings once there is a key to generate them.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.6"))

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
