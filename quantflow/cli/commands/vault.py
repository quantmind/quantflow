import click

from .base import QuantContext, quant_group

API_KEYS = ("fmp", "fred")


@quant_group()
def vault() -> None:
    """Manage vault secrets"""
    ctx = QuantContext.current()
    if ctx.invoked_subcommand is None:
        ctx.set_as_section()


@vault.command()
@click.argument("key", type=click.Choice(API_KEYS))
@click.argument("value")
def add(key: str, value: str) -> None:
    """Add an API key to the vault"""
    app = QuantContext.current().qf
    app.vault.add(key, value)


@vault.command()
@click.argument("key")
def delete(key: str) -> None:
    """Delete an API key from the vault"""
    app = QuantContext.current().qf
    if app.vault.delete(key):
        app.print(f"Deleted key {key}")
    else:
        app.error(f"Key {key} not found")


@vault.command()
@click.argument("key")
def show(key: str) -> None:
    """Show the value of an API key"""
    app = QuantContext.current().qf
    if value := app.vault.get(key):
        app.print(value)
    else:
        app.error(f"Key {key} not found")


@vault.command()
def keys() -> None:
    """Show the keys in the vault"""
    app = QuantContext.current().qf
    for key in app.vault.keys():
        app.print(key)
