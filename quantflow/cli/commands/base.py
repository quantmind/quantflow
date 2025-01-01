from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Self, cast

import click

from quantflow.data.fmp import FMP
from quantflow.data.fred import Fred

if TYPE_CHECKING:
    from quantflow.cli.app import QfApp


FREQUENCIES = tuple(FMP().historical_frequencies())


class HistoricalPeriod(enum.StrEnum):
    day = "1d"
    week = "1w"
    month = "1m"
    three_months = "3m"
    six_months = "6m"
    year = "1y"


class QuantContext(click.Context):

    @classmethod
    def current(cls) -> Self:
        return cast(Self, click.get_current_context())

    @property
    def qf(self) -> QfApp:
        return self.obj  # type: ignore

    def set_as_section(self) -> None:
        group = cast(QuantGroup, self.command)
        group.add_command(back)
        self.qf.set_section(group)
        self.qf.print(self.get_help())

    def fmp(self) -> FMP:
        if key := self.qf.vault.get("fmp"):
            return FMP(key=key)
        else:
            raise click.UsageError("No FMP API key found")

    def fred(self) -> Fred:
        if key := self.qf.vault.get("fred"):
            return Fred(key=key)
        else:
            raise click.UsageError("No FRED API key found")


class QuantCommand(click.Command):
    context_class = QuantContext


class QuantGroup(click.Group):
    context_class = QuantContext
    command_class = QuantCommand


@click.command(cls=QuantCommand)
def exit() -> None:
    """Exit the program"""
    raise click.Abort()


@click.command(cls=QuantCommand)
def help() -> None:
    """display the commands"""
    if ctx := QuantContext.current().parent:
        cast(QuantContext, ctx).qf.print(ctx.get_help())


@click.command(cls=QuantCommand)
def back() -> None:
    """Exit the current section"""
    ctx = QuantContext.current()
    ctx.qf.back()
    ctx.qf.handle_command("help")


def quant_group() -> Any:
    return click.group(
        cls=QuantGroup,
        commands=[exit, help],
        invoke_without_command=True,
        add_help_option=False,
    )


class options:
    length = click.option(
        "-l",
        "--length",
        type=int,
        default=100,
        show_default=True,
        help="Number of data points",
    )
    height = click.option(
        "-h",
        "--height",
        type=int,
        default=20,
        show_default=True,
        help="Chart height",
    )
    chart = click.option("-c", "--chart", is_flag=True, help="Display chart")
    period = click.option(
        "-p",
        "--period",
        type=click.Choice(tuple(p.value for p in HistoricalPeriod)),
        default="1d",
        show_default=True,
        help="Historical period",
    )
    index = click.option(
        "-i",
        "--index",
        type=int,
        default=-1,
        help="maturity index",
    )
    frequency = click.option(
        "-f",
        "--frequency",
        type=click.Choice(FREQUENCIES),
        default="",
        help="Frequency of data - if not provided it is daily",
    )
