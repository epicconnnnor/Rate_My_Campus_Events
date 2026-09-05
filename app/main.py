from app.api.routes_auth import router as auth_router
from app.api.routes_chat import router as chat_router
from app.api.routes_events import router as events_router
import logging

from app.core.config import COOKIE_SECURE, require_secret_key
from app.rag.providers import announce_models
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="RateMyCampusEvents",
              version="0.1.0")
# Checked before anything can be served: a missing key would otherwise only
# surface at the first login, as a confusing 500.
_secret_key = require_secret_key()

# Names the model ids in the boot log, so a retired one is identifiable
# before it becomes a 404 inside somebody's question.
logging.getLogger("providers").setLevel(logging.INFO)
announce_models()

# Holds the OAuth state between the redirect out and the callback back.
# Nothing else uses it; the app itself is still authenticated by JWT.
app.add_middleware(
    SessionMiddleware, secret_key=_secret_key, same_site="lax", https_only=COOKIE_SECURE
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="templates")

# Schema is managed by Alembic -- run `alembic upgrade head` before starting.

# routers
app.include_router(auth_router)
app.include_router(events_router)
app.include_router(chat_router)
