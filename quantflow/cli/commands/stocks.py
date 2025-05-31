from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import cast

import click
import pandas as pd
from asciichartpy import plot
from ccy import period as to_period
from ccy.cli.console import df_to_rich
from ccy.tradingcentres import prevbizday

from quantflow.utils.dates import utcnow

from .base import HistoricalPeriod, QuantContext, options, quant_group


@quant_group()
def stocks() -> None:
    """Stocks commands"""
    ctx = QuantContext.current()
    if ctx.invoked_subcommand is None:
        ctx.set_as_section()


@stocks.command()
def indices() -> None:
    """Search companies"""
    ctx = QuantContext.current()
    data = asyncio.run(get_indices(ctx))
    df = pd.DataFrame(data)
    ctx.qf.print(df_to_rich(df))


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
@options.height
@options.length
@options.frequency
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


@stocks.command()
@options.period
def sectors(period: str) -> None:
    """Sectors performance and PE ratios"""
    ctx = QuantContext.current()
    data = asyncio.run(sector_performance(ctx, HistoricalPeriod(period)))
    df = pd.DataFrame(data, columns=["sector", "performance", "pe"]).sort_values(
        "performance", ascending=False
    )
    ctx.qf.print(df_to_rich(df))


async def get_indices(ctx: QuantContext) -> list[dict]:
    async with ctx.fmp() as cli:
        return await cli.indices()


async def get_prices(ctx: QuantContext, symbol: str, frequency: str) -> pd.DataFrame:
    async with ctx.fmp() as cli:
        return await cli.prices(symbol, frequency)


async def get_profile(ctx: QuantContext, symbol: str) -> list[dict]:
    async with ctx.fmp() as cli:
        return await cli.profile(symbol)


async def search_company(ctx: QuantContext, text: str) -> list[dict]:
    async with ctx.fmp() as cli:
        return await cli.search(text)


async def sector_performance(
    ctx: QuantContext, period: HistoricalPeriod
) -> dict | list[dict]:
    async with ctx.fmp() as cli:
        to_date = utcnow().date()
        if period != HistoricalPeriod.day:
            from_date = to_date - timedelta(days=to_period(period.value).totaldays)
            sp = await cli.sector_performance(
                from_date=prevbizday(from_date, 0).isoformat(),  # type: ignore
                to_date=prevbizday(to_date, 0).isoformat(),  # type: ignore
                summary=True,
            )
        else:
            sp = await cli.sector_performance()
        spd = cast(dict, sp)
        pe = await cli.sector_pe(params=dict(date=prevbizday(to_date, 0).isoformat()))  # type: ignore
        pes = {}
        for k in pe:
            sector = k["sector"]
            if sector in spd:
                pes[sector] = round(float(k["pe"]), 3)
        return [
            dict(sector=k, performance=float(v), pe=pes.get(k, float("nan")))
            for k, v in spd.items()
        ]
