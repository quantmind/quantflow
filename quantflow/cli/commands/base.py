from __future__ import annotations

import click
from typing import TYPE_CHECKING, cast, Self
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
