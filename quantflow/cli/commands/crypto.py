from __future__ import annotations

import asyncio

import click
import pandas as pd
from asciichartpy import plot
from ccy.cli.console import df_to_rich

from quantflow.data.deribit import Deribit

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


async def get_volatility(ctx: QuantContext, currency: str) -> pd.DataFrame:
    async with Deribit() as client:
        return await client.get_volatility(params=dict(currency=currency))
