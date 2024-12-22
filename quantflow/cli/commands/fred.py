from __future__ import annotations

import click
import asyncio
import pandas as pd
from typing import TYPE_CHECKING
from fluid.utils.data import compact_dict
from fluid.utils.http_client import HttpResponseError
from ccy.cli.console import df_to_rich
from quantflow.data.fmp import FMP
from .stocks import from_context


FREQUENCIES = tuple(FMP().historical_frequencies())

if TYPE_CHECKING:
    from quantflow.cli.app import QfApp


@click.group(invoke_without_command=True)
@click.pass_context
def fred(ctx: click.Context) -> None:
    """Federal Reserve of St. Louis data"""
    if ctx.invoked_subcommand is None:
        app = from_context(ctx)
        app.print("Welcome to FRED data commands!")
        app.print(ctx.get_help())


@fred.command()
@click.pass_context
@click.argument("category-id", required=False)
def subcategories(ctx: click.Context, category_id: str | None = None) -> None:
    """List subcategories for a Fred category"""
    app = from_context(ctx)
    try:
        data = asyncio.run(get_subcategories(app, category_id))
    except HttpResponseError as e:
        app.error(e)
    else:
        df = pd.DataFrame(data["categories"], columns=["id", "name"])
        app.print(df_to_rich(df))


@fred.command()
@click.pass_context
@click.argument("category-id")
def series(ctx: click.Context, category_id: str) -> None:
    """List series for a Fred category"""
    app = from_context(ctx)
    try:
        data = asyncio.run(get_series(app, category_id))
    except HttpResponseError as e:
        app.error(e)
    else:
        app.print(data)
        # df = pd.DataFrame(data["categories"], columns=["id", "name"])
        # app.print(df_to_rich(df))


async def get_subcategories(app: QfApp, category_id: str | None) -> dict:
    async with app.fred() as cli:
        return await cli.subcategories(params=compact_dict(category_id=category_id))


async def get_series(app: QfApp, category_id: str) -> dict:
    async with app.fred() as cli:
        return await cli.series(params=compact_dict(category_id=category_id))
