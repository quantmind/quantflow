from __future__ import annotations

import click
import asyncio
import pandas as pd
from typing import TYPE_CHECKING
from asciichartpy import plot
from ccy.cli.console import df_to_rich
from quantflow.data.fmp import FMP

FREQUENCIES = tuple(FMP().historical_frequencies())

if TYPE_CHECKING:
    from quantflow.cli.app import QfApp


def from_context(ctx: click.Context) -> QfApp:
    return ctx.obj  # type: ignore


@click.command()
def exit() -> None:
    """Exit the program"""
    raise click.Abort()


@click.command()
@click.argument("symbol")
@click.pass_context
def profile(ctx: click.Context, symbol: str) -> None:
    """Company profile"""
    app = from_context(ctx)
    data = asyncio.run(get_profile(symbol))
    if not data:
        raise click.UsageError(f"Company {symbol} not found - try searching")
    else:
        d = data[0]
        app.print(d.pop("description") or "")
        df = pd.DataFrame(d.items(), columns=["Key", "Value"])
        app.print(df_to_rich(df))


@click.command()
@click.argument("text")
@click.pass_context
def search(ctx: click.Context, text: str) -> None:
    """Search companies"""
    app = from_context(ctx)
    data = asyncio.run(search_company(text))
    df = pd.DataFrame(data, columns=["symbol", "name", "currency", "stockExchange"])
    app.print(df_to_rich(df))


@click.command()
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
    help="Frequency of data - if not provided it is daily",
)
def chart(symbol: str, height: int, length: int, frequency: str) -> None:
    """Symbol chart"""
    df = asyncio.run(get_prices(symbol, frequency))
    if df.empty:
        raise click.UsageError(
            f"No data for {symbol} - are you sure the symbol exists?"
        )
    data = list(reversed(df["close"].tolist()[:length]))
    print(plot(data, {"height": height}))


async def get_prices(symbol: str, frequency: str) -> pd.DataFrame:
    async with FMP() as cli:
        return await cli.prices(symbol, frequency)


async def get_profile(symbol: str) -> list[dict]:
    async with FMP() as cli:
        return await cli.profile(symbol)


async def search_company(text: str) -> list[dict]:
    async with FMP() as cli:
        return await cli.search(text)
