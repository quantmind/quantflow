from __future__ import annotations

import click
import asyncio
import pandas as pd
from asciichartpy import plot
from ccy.cli.console import df_to_rich
from quantflow.data.fmp import FMP
from .base import QuantGroup, QuantContext

FREQUENCIES = tuple(FMP().historical_frequencies())


@click.group(invoke_without_command=True, cls=QuantGroup)
def stocks() -> None:
    """Stocks commands"""
    ctx = QuantContext.current()
    if ctx.invoked_subcommand is None:
        ctx.qf.print("Welcome to the stocks commands!")
        ctx.qf.print(ctx.get_help())


@stocks.command()
@click.argument("symbol")
def profile(symbol: str) -> None:
    """Company profile"""
    ctx = QuantContext.current()
    data = asyncio.run(get_profile(ctx, symbol))
    if not data:
        raise click.UsageError(f"Company {symbol} not found - try searching")
    else:
        d = data[0]
        ctx.qf.print(d.pop("description") or "")
        df = pd.DataFrame(d.items(), columns=["Key", "Value"])
        ctx.qf.print(df_to_rich(df))


@stocks.command()
@click.argument("text")
def search(text: str) -> None:
    """Search companies"""
    ctx = QuantContext.current()
    data = asyncio.run(search_company(ctx, text))
    df = pd.DataFrame(data, columns=["symbol", "name", "currency", "stockExchange"])
    ctx.qf.print(df_to_rich(df))


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
def chart(symbol: str, height: int, length: int, frequency: str) -> None:
    """Symbol chart"""
    ctx = QuantContext.current()
    df = asyncio.run(get_prices(ctx, symbol, frequency))
    if df.empty:
        raise click.UsageError(
            f"No data for {symbol} - are you sure the symbol exists?"
        )
    data = list(reversed(df["close"].tolist()[:length]))
    print(plot(data, {"height": height}))


async def get_prices(ctx: QuantContext, symbol: str, frequency: str) -> pd.DataFrame:
    async with ctx.fmp() as cli:
        return await cli.prices(symbol, frequency)


async def get_profile(ctx: QuantContext, symbol: str) -> list[dict]:
    async with ctx.fmp() as cli:
        return await cli.profile(symbol)


async def search_company(ctx: QuantContext, text: str) -> list[dict]:
    async with ctx.fmp() as cli:
        return await cli.search(text)
