from __future__ import annotations

import click
import asyncio
import pandas as pd
from typing import TYPE_CHECKING
from asciichartpy import plot
from fluid.utils.data import compact_dict
from fluid.utils.http_client import HttpResponseError
from ccy.cli.console import df_to_rich
from quantflow.data.fmp import FMP
from .base import QuantGroup, QuantContext


FREQUENCIES = tuple(FMP().historical_frequencies())

if TYPE_CHECKING:
    pass


@click.group(invoke_without_command=True, cls=QuantGroup)
def fred() -> None:
    """Federal Reserve of St. Louis data"""
    ctx = QuantContext.current()
    if ctx.invoked_subcommand is None:
        ctx.qf.print("Welcome to FRED data commands!")
        ctx.qf.print(ctx.get_help())


@fred.command()
@click.argument("category-id", required=False)
def subcategories(category_id: str | None = None) -> None:
    """List subcategories for a Fred category"""
    ctx = QuantContext.current()
    try:
        data = asyncio.run(get_subcategories(ctx, category_id))
    except HttpResponseError as e:
        ctx.qf.error(e)
    else:
        df = pd.DataFrame(data["categories"], columns=["id", "name"])
        ctx.qf.print(df_to_rich(df))


@fred.command()
@click.argument("category-id")
def series(category_id: str) -> None:
    """List series for a Fred category"""
    ctx = QuantContext.current()
    try:
        data = asyncio.run(get_series(ctx, category_id))
    except HttpResponseError as e:
        ctx.qf.error(e)
    else:
        ctx.qf.print(data)
        # df = pd.DataFrame(data["categories"], columns=["id", "name"])
        # app.print(df_to_rich(df))


@fred.command()
@click.argument("series-id")
@click.option(
    "-l",
    "--length",
    type=int,
    default=100,
    show_default=True,
    help="Number of data points",
)
def data(series_id: str, length: int) -> None:
    """Display a series data"""
    ctx = QuantContext.current()
    try:
        df = asyncio.run(get_serie_data(ctx, series_id, length))
    except HttpResponseError as e:
        ctx.qf.error(e)
    else:
        ctx.qf.print(df_to_rich(df))

@fred.command()
@click.argument("series-id")
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
def chart(series_id: str, height: int, length: int) -> None:
    """Chart a serie"""
    ctx = QuantContext.current()
    try:
        df = asyncio.run(get_serie_data(ctx, series_id, length))
    except HttpResponseError as e:
        ctx.qf.error(e)
    else:
        data = list(reversed(df["value"].tolist()[:length]))
        ctx.qf.print(plot(data, {"height": height}))


async def get_subcategories(ctx: QuantContext, category_id: str | None) -> dict:
    async with ctx.fred() as cli:
        return await cli.subcategories(params=compact_dict(category_id=category_id))


async def get_series(ctx: QuantContext, category_id: str) -> dict:
    async with ctx.fred() as cli:
        return await cli.series(params=compact_dict(category_id=category_id))


async def get_serie_data(ctx: QuantContext, series_id: str, length: int) -> dict:
    async with ctx.fred() as cli:
        return await cli.serie_data(params=dict(series_id=series_id, limit=length))
