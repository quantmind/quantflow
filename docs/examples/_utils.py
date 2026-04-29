import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

EXAMPLE_DIR = Path(__file__).parent
OUT_DIR = EXAMPLE_DIR / "output"
ASSET_DIR = EXAMPLE_DIR.parent / "assets" / "examples"


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
