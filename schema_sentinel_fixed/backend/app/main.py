from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)

    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        logging.warning("Could not create DB tables at startup: %s", e)

    # ✅ CORS for Vite frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/v1")
    return app


app = create_app()