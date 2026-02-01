from typing import Annotated, Callable, Coroutine
from fastapi.responses import HTMLResponse, RedirectResponse
import marimo
from fastapi import FastAPI, Form, Request, Response


def crate_app() -> FastAPI:
    # Create a marimo asgi app
    server = (
        marimo.create_asgi_app()
        .with_app(path="/supersmoother", root="./app/supersmoother.py")
    )
    # Create a FastAPI app
    app = FastAPI()
    app.mount("/", server.build())
    return app

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(crate_app(), host="localhost", port=8001)
