from app.api.routes_auth import router as auth_router
from app.api.routes_events import router as events_router
from app.db.database import create_db_and_tables
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="RateMyCampusEvents",
              version="0.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="templates")


# Initialize database tables on startup
@app.on_event("startup")
def on_startup():
    create_db_and_tables()


# routers
app.include_router(auth_router)
app.include_router(events_router)
