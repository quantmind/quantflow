from __future__ import annotations

import asyncio

import click
import pandas as pd
from asciichartpy import plot
from cache import AsyncTTL
from ccy.cli.console import df_to_rich

from quantflow.data.deribit import Deribit, InstrumentKind
from quantflow.options.surface import VolSurface
from quantflow.utils.numbers import round_to_step

from .base import QuantContext, options, quant_group
from .stocks import get_prices


@quant_group()
def crypto() -> None:
    """Crypto currencies commands"""
    ctx = QuantContext.current()
    if ctx.invoked_subcommand is None:
        ctx.set_as_section()


@crypto.command()
@click.argument("currency")
@click.option(
    "-k",
    "--kind",
    type=click.Choice(list(InstrumentKind)),
    default=InstrumentKind.spot.value,
)
def instruments(currency: str, kind: str) -> None:
    """Provides information about instruments

    Instruments for given cryptocurrency from Deribit API"""
    ctx = QuantContext.current()
    data = asyncio.run(get_instruments(ctx, currency, kind))
    df = pd.DataFrame(data)
    ctx.qf.print(df_to_rich(df))


@crypto.command()
@click.argument("currency")
@options.length
@options.height
@options.chart
def volatility(currency: str, length: int, height: int, chart: bool) -> None:
    """Provides information about historical volatility

    Historical volatility for given cryptocurrency from Deribit API
    """
    ctx = QuantContext.current()
    df = asyncio.run(get_volatility(ctx, currency))
    df["volatility"] = df["volatility"].map(lambda p: round_to_step(p, "0.01"))
    if chart:
        data = df["volatility"].tolist()[:length]
        ctx.qf.print(plot(data, {"height": height}))
    else:
        ctx.qf.print(df_to_rich(df))


@crypto.command()
@click.argument("currency")
def term_structure(currency: str) -> None:
    """Provides information about the term structure for given cryptocurrency"""
    ctx = QuantContext.current()
    vs = asyncio.run(get_vol_surface(currency))
    ts = vs.term_structure().round({"ttm": 4})
    ts["open_interest"] = ts["open_interest"].map("{:,d}".format)
    ts["volume"] = ts["volume"].map("{:,d}".format)
    ctx.qf.print(df_to_rich(ts))


@crypto.command()
@click.argument("currency")
@options.index
@options.height
@options.chart
def implied_vol(currency: str, index: int, height: int, chart: bool) -> None:
    """Display the Volatility Surface for given cryptocurrency
    at a given maturity index
    """
    ctx = QuantContext.current()
    vs = asyncio.run(get_vol_surface(currency))
    index_or_none = None if index < 0 else index
    vs.bs(index=index_or_none)
    df = vs.options_df(index=index_or_none)
    if chart:
        data = (df["implied_vol"] * 100).tolist()
        ctx.qf.print(plot(data, {"height": height}))
    else:
        df[["ttm", "moneyness", "moneyness_ttm"]] = df[
            ["ttm", "moneyness", "moneyness_ttm"]
        ].map("{:.4f}".format)
        df["implied_vol"] = df["implied_vol"].map("{:.2%}".format)
        df["price"] = df["price"].map(lambda p: round_to_step(p, vs.tick_size_options))
        df["forward_price"] = df["forward_price"].map(
            lambda p: round_to_step(p, vs.tick_size_forwards)
        )
        ctx.qf.print(df_to_rich(df))


@crypto.command()
@click.argument("symbol")
@options.height
@options.length
@options.chart
@options.frequency
def prices(symbol: str, height: int, length: int, chart: bool, frequency: str) -> None:
    """Fetch OHLC prices for given cryptocurrency"""
    ctx = QuantContext.current()
    df = asyncio.run(get_prices(ctx, symbol, frequency))
    if df.empty:
        raise click.UsageError(
            f"No data for {symbol} - are you sure the symbol exists?"
        )
    if chart:
        data = list(reversed(df["close"].tolist()[:length]))
        ctx.qf.print(plot(data, {"height": height}))
    else:
        ctx.qf.print(
            df_to_rich(
                df[["date", "open", "high", "low", "close", "volume"]].sort_values(
                    "date"
                )
            )
        )


async def get_instruments(ctx: QuantContext, currency: str, kind: str) -> list[dict]:
    async with Deribit() as client:
        return await client.get_instruments(
            currency=currency, kind=InstrumentKind(kind)
        )


async def get_volatility(ctx: QuantContext, currency: str) -> pd.DataFrame:
    async with Deribit() as client:
        return await client.get_volatility(currency)


@AsyncTTL(time_to_live=10)
async def get_vol_surface(currency: str) -> VolSurface:
    async with Deribit() as client:
        loader = await client.volatility_surface_loader(currency)
        return loader.surface()
