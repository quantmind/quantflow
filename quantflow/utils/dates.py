from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)
