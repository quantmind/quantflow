from __future__ import annotations

import asyncio

import click
import pandas as pd
from asciichartpy import plot
from ccy.cli.console import df_to_rich
from fluid.utils.data import compact_dict
from fluid.utils.http_client import HttpResponseError

from quantflow.data.fred import Fred

from .base import QuantContext, options, quant_group

FREQUENCIES = tuple(Fred.freq)


@quant_group()
def fred() -> None:
    """Federal Reserve of St. Louis data commands"""
    ctx = QuantContext.current()
    if ctx.invoked_subcommand is None:
        ctx.set_as_section()


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
@click.option("-j", "--json", is_flag=True, help="Output as JSON")
def series(category_id: str, json: bool = False) -> None:
    """List series for a Fred category"""
    ctx = QuantContext.current()
    try:
        data = asyncio.run(get_series(ctx, category_id))
    except HttpResponseError as e:
        ctx.qf.error(e)
    else:
        if json:
            ctx.qf.print(data)
        else:
            df = pd.DataFrame(
                data["seriess"],
                columns=[
                    "id",
                    "popularity",
                    "title",
                    "frequency",
                    "observation_start",
                    "observation_end",
                ],
            ).sort_values("popularity", ascending=False)
            ctx.qf.print(df_to_rich(df))


@fred.command()
@click.argument("series-id")
@options.length
@options.height
@options.chart
@click.option(
    "-f",
    "--frequency",
    type=click.Choice(FREQUENCIES),
    default="d",
    show_default=True,
    help="Frequency of data",
)
def data(series_id: str, length: int, height: int, chart: bool, frequency: str) -> None:
    """Display a series data"""
    ctx = QuantContext.current()
    try:
        df = asyncio.run(get_serie_data(ctx, series_id, length, frequency))
    except HttpResponseError as e:
        ctx.qf.error(e)
    else:
        if chart:
            data = list(reversed(df["value"].tolist()[:length]))
            ctx.qf.print(plot(data, {"height": height}))
        else:
            ctx.qf.print(df_to_rich(df))


async def get_subcategories(ctx: QuantContext, category_id: str | None) -> dict:
    async with ctx.fred() as cli:
        return await cli.subcategories(params=compact_dict(category_id=category_id))


async def get_series(ctx: QuantContext, category_id: str) -> dict:
    async with ctx.fred() as cli:
        return await cli.series(params=compact_dict(category_id=category_id))


async def get_serie_data(
    ctx: QuantContext, series_id: str, length: int, frequency: str
) -> dict:
    async with ctx.fred() as cli:
        return await cli.serie_data(
            params=dict(
                series_id=series_id,
                limit=length,
                frequency=frequency,
                sort_order="desc",
            )
        )
