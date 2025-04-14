from datetime import date, datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: date | None = None) -> datetime:
    if dt is None:
        return utcnow()
    elif isinstance(dt, datetime):
        return dt.astimezone(timezone.utc)
    else:
        return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def isoformat(date: str | date) -> str:
    if isinstance(date, str):
        return date
    return date.isoformat()


def start_of_day(dt: date | None = None) -> datetime:
    return as_utc(dt).replace(hour=0, minute=0, second=0, microsecond=0)


def as_date(dt: date | None = None) -> date:
    if dt is None:
        return date.today()
    elif isinstance(dt, datetime):
        return dt.date()
    else:
        return dt
