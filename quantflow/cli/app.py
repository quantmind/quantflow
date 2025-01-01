import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import click
from fluid.utils.http_client import HttpResponseError
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.text import Text

from quantflow.data.vault import Vault

from . import settings
from .commands import quantflow
from .commands.base import QuantGroup


@dataclass
class QfApp:
    console: Console = field(default_factory=Console)
    vault: Vault = field(default_factory=partial(Vault, settings.VAULT_FILE_PATH))
    sections: list[QuantGroup] = field(default_factory=lambda: [quantflow])

    def __call__(self) -> None:
        os.makedirs(settings.SETTINGS_DIRECTORY, exist_ok=True)
        history = FileHistory(str(settings.HIST_FILE_PATH))
        session: PromptSession = PromptSession(history=history)

        self.print("Welcome to QuantFlow!", style="bold green")
        self.handle_command("help")

        try:
            while True:
                try:
                    text = session.prompt(
                        self.prompt_message(),
                        completer=self.prompt_completer(),
                        complete_while_typing=True,
                        bottom_toolbar=self.bottom_toolbar,
                    )
                except KeyboardInterrupt:
                    break
                else:
                    self.handle_command(text)
        except click.Abort:
            self.console.print(Text("Bye!", style="bold magenta"))

    def prompt_message(self) -> str:
        name = ":".join([str(section.name) for section in self.sections])
        return f"{name} > "

    def prompt_completer(self) -> NestedCompleter:
        return NestedCompleter.from_nested_dict(
            {command: None for command in self.sections[-1].commands}
        )

    def set_section(self, section: QuantGroup) -> None:
        self.sections.append(section)

    def back(self) -> None:
        self.sections.pop()

    def print(self, text_alike: Any, style: str = "") -> None:
        if isinstance(text_alike, str):
            style = style or "cyan"
            text_alike = Text(f"\n{text_alike}\n", style="cyan")
        self.console.print(text_alike)

    def error(self, err: str | Exception) -> None:
        self.console.print(Text(f"\n{err}\n", style="bold red"))

    def handle_command(self, text: str) -> None:
        if not text:
            return
        command = self.sections[-1]
        try:
            command.main(text.split(), standalone_mode=False, obj=self)
        except (
            click.exceptions.MissingParameter,
            click.exceptions.NoSuchOption,
            click.exceptions.UsageError,
            HttpResponseError,
        ) as e:
            self.error(e)

    def bottom_toolbar(self) -> HTML:
        sections = "/".join([str(section.name) for section in self.sections])
        back = (
            (' <b><style bg="ansired">back</style></b> ' "to exit the current section,")
            if len(self.sections) > 1
            else ""
        )
        return HTML(
            f"Your are in <strong>{sections}</strong>, type{back} "
            '<b><style bg="ansired">exit</style></b> to exit'
        )
