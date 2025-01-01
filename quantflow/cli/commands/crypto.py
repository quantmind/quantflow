from __future__ import annotations

import asyncio

import click
import pandas as pd
from asciichartpy import plot
from cache import AsyncTTL
from ccy.cli.console import df_to_rich

from quantflow.data.deribit import Deribit
from quantflow.options.surface import VolSurface

from .base import QuantContext, options, quant_group


@quant_group()
def crypto() -> None:
    """Crypto currencies commands"""
    ctx = QuantContext.current()
    if ctx.invoked_subcommand is None:
        ctx.set_as_section()


@crypto.command()
@click.argument("currency")
@options.length
@options.height
@options.chart
def volatility(currency: str, length: int, height: int, chart: bool) -> None:
    """Provides information about historical volatility for given cryptocurrency"""
    ctx = QuantContext.current()
    df = asyncio.run(get_volatility(ctx, currency))
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
    df["implied_vol"] = df["implied_vol"] * 100
    df = df.round({"ttm": 4, "moneyness": 4, "moneyness_ttm": 4, "implied_vol": 5})
    if chart:
        data = df["implied_vol"].tolist()
        ctx.qf.print(plot(data, {"height": height}))
    else:
        ctx.qf.print(df_to_rich(df))


async def get_volatility(ctx: QuantContext, currency: str) -> pd.DataFrame:
    async with Deribit() as client:
        return await client.get_volatility(params=dict(currency=currency))


@AsyncTTL(time_to_live=10)
async def get_vol_surface(currency: str) -> VolSurface:
    async with Deribit() as client:
        loader = await client.volatility_surface_loader(currency)
        return loader.surface()
