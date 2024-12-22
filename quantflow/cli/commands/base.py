from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import click

from quantflow.data.fmp import FMP
from quantflow.data.fred import Fred

if TYPE_CHECKING:
    from quantflow.cli.app import QfApp


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
