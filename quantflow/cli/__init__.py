import asyncio

import click
import dotenv
import pandas as pd
from asciichartpy import plot

from quantflow.data.fmp import FMP

dotenv.load_dotenv()


@click.group()
def qf():
    pass


@qf.command()
@click.argument("symbol")
def profile(symbol: str):
    """Company profile"""
    data = asyncio.run(get_profile(symbol))[0]
    print(data.pop("description"))


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
def chart(symbol: str, height: int, length: int):
    """Symbol chart"""
    df = asyncio.run(get_prices(symbol))
    data = df["close"].tolist()
    print(plot(data[:length], {"height": height}))


async def get_prices(symbol: str) -> pd.DataFrame:
    async with FMP() as cli:
        return await cli.prices(symbol)


async def get_profile(symbol: str) -> dict:
    async with FMP() as cli:
        return await cli.profile(symbol)
