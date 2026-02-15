import os
from pathlib import Path

import marimo
from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles

APP_PATH = Path(__file__).parent
PORT = int(os.environ.get("MICRO_SERVICE_PORT", "8001"))
status_router = APIRouter()


def crate_app() -> FastAPI:
    # Create a marimo asgi app
    server = marimo.create_asgi_app(include_code=True)
    for path in APP_PATH.glob("*.py"):
        if path.name.startswith("_"):
            continue
        dashed = path.stem.replace("_", "-")
        server = server.with_app(path=f"/{dashed}", root=f"./app/{path.name}")
    # Create a FastAPI app
    app = FastAPI()
    app.include_router(status_router)
    app.mount("/examples", server.build())
    app.mount("/", StaticFiles(directory=APP_PATH / "docs", html=True), name="static")
    return app

@status_router.get("/status")
async def service_status() -> dict:
    return {"status": "ok"}


@status_router.get("/ready")
async def service_ready() -> dict:
    return {"status": "ok"}


# Run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(crate_app(), host="0.0.0.0", port=PORT)
