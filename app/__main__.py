import os

import marimo
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from fluid.utils import log
from app.utils.paths import APP_PATH, head_snippet
from quantflow import __version__

from .api.deps import instrument_app
from .api.rates import rates_router
from .api.status import status_router
from .api.volatility import volatility_router

PORT = int(os.environ.get("MICRO_SERVICE_PORT", "8001"))


def crate_app() -> FastAPI:
    # Create a marimo asgi app
    html_head = head_snippet(APP_PATH / "docs")
    server = marimo.create_asgi_app(include_code=True, html_head=html_head)
    for path in APP_PATH.glob("*.py"):
        if path.name.startswith("_"):
            continue
        dashed = path.stem.replace("_", "-")
        server = server.with_app(path=f"/{dashed}", root=f"./app/{path.name}")
    app = FastAPI(
        version=__version__,
        title="Quantflow API",
        description="API for Quantflow",
    )
    instrument_app(app)
    cors_origins = [
        origin.strip()
        for origin in os.environ.get("QUANTFLOW_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    api = APIRouter(prefix="/.api")

    @api.get("/redoc", include_in_schema=False)
    async def api_redoc() -> HTMLResponse:
        return get_redoc_html(
            openapi_url="/openapi.json",
            title="Quantflow API",
        )

    api.include_router(rates_router)
    api.include_router(volatility_router)
    app.include_router(api)
    app.include_router(status_router, include_in_schema=False)
    app.mount("/examples", server.build())
    app.mount("/", StaticFiles(directory=APP_PATH / "docs", html=True), name="static")
    return app


# Run the server
if __name__ == "__main__":
    import uvicorn
    log.config()
    uvicorn.run(crate_app(), host="0.0.0.0", port=PORT)
