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


@click.group(invoke_without_command=True)
@click.pass_context
def stocks(ctx: click.Context) -> None:
    """Stocks commands"""
    if ctx.invoked_subcommand is None:
        app = from_context(ctx)
        app.print("Welcome to the stocks commands!")
        app.print(ctx.get_help())


@stocks.command()
@click.argument("symbol")
@click.pass_context
def profile(ctx: click.Context, symbol: str) -> None:
    """Company profile"""
    app = from_context(ctx)
    data = asyncio.run(get_profile(app, symbol))
    if not data:
        raise click.UsageError(f"Company {symbol} not found - try searching")
    else:
        d = data[0]
        app.print(d.pop("description") or "")
        df = pd.DataFrame(d.items(), columns=["Key", "Value"])
        app.print(df_to_rich(df))


@stocks.command()
@click.argument("text")
@click.pass_context
def search(ctx: click.Context, text: str) -> None:
    """Search companies"""
    app = from_context(ctx)
    data = asyncio.run(search_company(app, text))
    df = pd.DataFrame(data, columns=["symbol", "name", "currency", "stockExchange"])
    app.print(df_to_rich(df))


@stocks.command()
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
@click.pass_context
def chart(
    ctx: click.Context, symbol: str, height: int, length: int, frequency: str
) -> None:
    """Symbol chart"""
    app = from_context(ctx)
    df = asyncio.run(get_prices(app, symbol, frequency))
    if df.empty:
        raise click.UsageError(
            f"No data for {symbol} - are you sure the symbol exists?"
        )
    data = list(reversed(df["close"].tolist()[:length]))
    print(plot(data, {"height": height}))


async def get_prices(app: QfApp, symbol: str, frequency: str) -> pd.DataFrame:
    async with app.fmp() as cli:
        return await cli.prices(symbol, frequency)


async def get_profile(app: QfApp, symbol: str) -> list[dict]:
    async with app.fmp() as cli:
        return await cli.profile(symbol)


async def search_company(app: QfApp, text: str) -> list[dict]:
    async with app.fmp() as cli:
        return await cli.search(text)
