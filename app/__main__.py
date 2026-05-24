import os

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fluid.utils import log

from app.utils.paths import APP_PATH
from quantflow import __version__

from .api.cointegration import cointegration_router
from .api.deps import instrument_app
from .api.heston import heston_router
from .api.hurst import hurst_router
from .api.rates import rates_router
from .api.sampling import sampling_router
from .api.smoother import smoother_router
from .api.status import status_router
from .api.volatility import volatility_router
from .utils.static import HtmlFallbackStaticFiles


def crate_app() -> FastAPI:
    load_dotenv()
    log.config()
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

    api.include_router(cointegration_router)
    api.include_router(heston_router)
    api.include_router(hurst_router)
    api.include_router(rates_router)
    api.include_router(sampling_router)
    api.include_router(smoother_router)
    api.include_router(volatility_router)
    app.include_router(api)
    app.include_router(status_router, include_in_schema=False)
    examples_dir = APP_PATH / "examples"
    if examples_dir.is_dir():
        app.mount(
            "/examples",
            HtmlFallbackStaticFiles(directory=examples_dir, html=True),
            name="examples",
        )
    docs_dir = APP_PATH / "docs"
    if docs_dir.is_dir():
        app.mount("/", StaticFiles(directory=docs_dir, html=True), name="static")
    return app


# Run the server
if __name__ == "__main__":
    import uvicorn

    PORT = int(os.environ.get("MICRO_SERVICE_PORT", "8001"))
    HOST = os.environ.get("MICRO_SERVICE_HOST", "0.0.0.0")
    uvicorn.run(
        crate_app(), host=HOST, port=PORT, proxy_headers=True, forwarded_allow_ips="*"
    )
