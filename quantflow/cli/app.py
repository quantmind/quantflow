import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import click
import dotenv
import pandas as pd
from asciichartpy import plot
from ccy.cli.console import df_to_rich
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.text import Text

from quantflow.data.fmp import FMP

from . import settings

dotenv.load_dotenv()

FREQUENCIES = tuple(FMP().historical_frequencies())


@click.group()
def qf() -> None:
    pass


@qf.command()
@click.argument("symbol")
def profile(symbol: str) -> None:
    """Company profile"""
    data = asyncio.run(get_profile(symbol))[0]
    main.print(data.pop("description"))
    df = pd.DataFrame(data.items(), columns=["Key", "Value"])
    main.print(df_to_rich(df))


@qf.command()
@click.argument("symbol")
@click.option(
    "-h",
    "--height",
    type=int,
    default=20,
    show_default=True,
    help="Chart height",
)
@click.option(
    "-l",
    "--length",
    type=int,
    default=100,
    show_default=True,
    help="Number of data points",
)
@click.option(
    "-f",
    "--frequency",
    type=click.Choice(FREQUENCIES),
    default="",
    help="Number of data points",
)
def chart(symbol: str, height: int, length: int, frequency: str) -> None:
    """Symbol chart"""
    df = asyncio.run(get_prices(symbol, frequency))
    data = list(reversed(df["close"].tolist()[:length]))
    print(plot(data, {"height": height}))


async def get_prices(symbol: str, frequency: str) -> pd.DataFrame:
    async with FMP() as cli:
        return await cli.prices(symbol, frequency)


async def get_profile(symbol: str) -> list[dict]:
    async with FMP() as cli:
        return await cli.profile(symbol)


@dataclass
class App:
    console: Console = field(default_factory=Console)

    def __call__(self) -> None:
        os.makedirs(settings.SETTINGS_DIRECTORY, exist_ok=True)
        history = FileHistory(str(settings.HIST_FILE_PATH))
        session: PromptSession = PromptSession(history=history)

        self.print("Welcome to QuantFlow!", style="bold green")
        self.handle_command("help")

        try:
            while True:
                try:
                    text = session.prompt("quantflow> ")
                except KeyboardInterrupt:
                    break
                else:
                    self.handle_command(text)
        except click.Abort:
            self.console.print(Text("Bye!", style="bold magenta"))

    def print(self, text_alike: Any, style: str = "") -> None:
        if isinstance(text_alike, str):
            style = style or "cyan"
            text_alike = Text(f"\n{text_alike}\n", style="cyan")
        self.console.print(text_alike)

    def error(self, err: str | Exception) -> None:
        self.console.print(Text(f"\n{err}\n", style="bold red"))

    def handle_command(self, text: str) -> None:
        self.current_command = text.split(" ")[0].strip()
        if not text:
            return
        elif text == "help":
            return qf.main(["--help"], standalone_mode=False)
        elif text == "exit":
            raise click.Abort()

        try:
            qf.main(text.split(), standalone_mode=False)
        except click.exceptions.MissingParameter as e:
            self.error(e)
        except click.exceptions.NoSuchOption as e:
            self.error(e)
