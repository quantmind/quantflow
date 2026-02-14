import marimo
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from pathlib import Path

APP_PATH = Path(__file__).parent


def crate_app() -> FastAPI:
    # Create a marimo asgi app
    server = marimo.create_asgi_app()
    for path in APP_PATH.glob("*.py"):
        if path.name.startswith("_"):
            continue
        dashed = path.stem.replace("_", "-")
        server = server.with_app(path=f"/{dashed}", root=f"./app/{path.name}")
    # Create a FastAPI app
    app = FastAPI()
    app.mount("/examples", server.build())
    app.mount("/", StaticFiles(directory=APP_PATH / "docs", html=True), name="static")
    return app

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(crate_app(), host="0.0.0.0", port=8001)
