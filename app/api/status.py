from fastapi import APIRouter

status_router = APIRouter()


@status_router.get("/status")
async def service_status() -> dict:
    return {"status": "ok"}


@status_router.get("/ready")
async def service_ready() -> dict:
    return {"status": "ok"}
