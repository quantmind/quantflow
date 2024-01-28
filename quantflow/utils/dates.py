from datetime import datetime, timezone, date


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(date: str | date) -> str:
    if isinstance(date, str):
        return date
    return date.isoformat()
