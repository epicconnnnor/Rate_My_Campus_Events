from app.api.routes_auth import router as auth_router
from app.api.routes_chat import router as chat_router
from app.api.routes_events import router as events_router
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="RateMyCampusEvents",
              version="0.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="templates")

# Schema is managed by Alembic -- run `alembic upgrade head` before starting.

# routers
app.include_router(auth_router)
app.include_router(events_router)
app.include_router(chat_router)
