import subprocess
import sys
import time
from collections.abc import Callable
from datetime import timedelta
from functools import wraps
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

EXAMPLE_DIR = Path(__file__).parent
FIXTURES = EXAMPLE_DIR / "fixtures"
CACHE_DIR = EXAMPLE_DIR / "cache"
OUT_DIR = EXAMPLE_DIR / "output"
ASSET_DIR = EXAMPLE_DIR.parent / "assets" / "examples"


def cached_df(
    ttl: timedelta,
) -> Callable[[Callable[[], pd.DataFrame]], Callable[[], pd.DataFrame]]:
    """Cache a DataFrame-returning function to a parquet file with a TTL.

    The wrapped function is only re-executed when the parquet cache is missing
    or older than ``ttl``; otherwise the cached frame is read back from disk.
    The cache file is named after the function and stored under ``cache/``.
    """

    def decorator(func: Callable[[], pd.DataFrame]) -> Callable[[], pd.DataFrame]:
        path = CACHE_DIR / f"{func.__name__}.parquet"

        @wraps(func)
        def wrapper() -> pd.DataFrame:
            fresh = (
                path.exists()
                and time.time() - path.stat().st_mtime < ttl.total_seconds()
            )
            if fresh:
                return pd.read_parquet(path)
            df = func()
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            return df

        return wrapper

    return decorator


def print_model(model: BaseModel) -> None:
    """Helper function to print a Pydantic model as pretty JSON"""
    text = model.model_dump_json(indent=2)
    text_data = ["", "```json", text, "```"]
    print("\n".join(text_data))


def assets_path(filename: str) -> str:
    """Helper function to get the path to an asset file in the docs"""
    return f"docs/assets/examples/{filename}"


def build_examples() -> list[Path]:
    failed = []
    OUT_DIR.mkdir(exist_ok=True)
    ASSET_DIR.mkdir(exist_ok=True)
    for script in sorted(EXAMPLE_DIR.glob("*.py")):
        if script.stem.startswith("_"):
            continue
        out_file = OUT_DIR / script.with_suffix(".out").name
        print(f"running {script} -> {out_file}")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"FAILED: {script}\n{result.stderr}", file=sys.stderr)
            failed.append(script)
        else:
            out_file.write_text(result.stdout)
    return failed
