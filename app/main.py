from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from app.api.routes import broadcast_updates, router as api_router
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Real-time stock analysis with technical indicators and AI predictions",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent

app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "index.html", media_type="text/html")


@app.on_event("startup")
async def startup() -> None:
    logger.info("Starting %s …", settings.app_name)
    asyncio.create_task(broadcast_updates())
    logger.info(
        "Background refresh every %ds for watchlist: %s",
        settings.refresh_interval_seconds,
        settings.default_watchlist,
    )
