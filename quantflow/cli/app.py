import os
from dataclasses import dataclass, field
from typing import Any

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.text import Text


from . import settings, commands


@click.group()
def qf() -> None:
    pass


qf.add_command(commands.exit)
qf.add_command(commands.profile)
qf.add_command(commands.search)
qf.add_command(commands.chart)


@dataclass
class QfApp:
    console: Console = field(default_factory=Console)

    def __call__(self) -> None:
        os.makedirs(settings.SETTINGS_DIRECTORY, exist_ok=True)
        history = FileHistory(str(settings.HIST_FILE_PATH))
        session: PromptSession = PromptSession(history=history)

        self.print("Welcome to QuantFlow!", style="bold green")
        self.handle_command("help")

        try:
            while True:
                try:
                    text = session.prompt("quantflow> ")
                except KeyboardInterrupt:
                    break
                else:
                    self.handle_command(text)
        except click.Abort:
            self.console.print(Text("Bye!", style="bold magenta"))

    def print(self, text_alike: Any, style: str = "") -> None:
        if isinstance(text_alike, str):
            style = style or "cyan"
            text_alike = Text(f"\n{text_alike}\n", style="cyan")
        self.console.print(text_alike)

    def error(self, err: str | Exception) -> None:
        self.console.print(Text(f"\n{err}\n", style="bold red"))

    def handle_command(self, text: str) -> None:
        self.current_command = text.split(" ")[0].strip()
        if not text:
            return
        elif text == "help":
            return qf.main(["--help"], standalone_mode=False, obj=self)

        try:
            qf.main(text.split(), standalone_mode=False, obj=self)
        except click.exceptions.MissingParameter as e:
            self.error(e)
        except click.exceptions.NoSuchOption as e:
            self.error(e)
        except click.exceptions.UsageError as e:
            self.error(e)
