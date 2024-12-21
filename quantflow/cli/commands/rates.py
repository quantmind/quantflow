from __future__ import annotations

import click
import asyncio
import pandas as pd
from typing import TYPE_CHECKING
from asciichartpy import plot
from ccy.cli.console import df_to_rich
from quantflow.data.fmp import FMP
from .stocks import from_context


FREQUENCIES = tuple(FMP().historical_frequencies())

if TYPE_CHECKING:
    from quantflow.cli.app import QfApp


@click.group(invoke_without_command=True)
@click.pass_context
def rates(ctx: click.Context) -> None:
    """Interest rates commands"""
    if ctx.invoked_subcommand is None:
        app = from_context(ctx)
        app.print("Welcome to the rates commands!")
        app.print(ctx.get_help())


@rates.command()
@click.pass_context
def category(ctx: click.Context) -> None:
    """List interest rate categories"""
    app = from_context(ctx)
    data = asyncio.run(get_category(app))
    #df = pd.DataFrame(data, columns=["Category"])
    #df = df_to_rich(df)
    app.print(data)


async def get_category(app: QfApp) -> list[str]:
    async with app.fred() as cli:
        return await cli.categiories()
