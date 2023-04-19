from datetime import date


def isoformat(date: str | date) -> str:
    if isinstance(date, str):
        return date
    return date.isoformat()
