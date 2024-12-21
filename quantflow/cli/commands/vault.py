import click
from .stocks import from_context


API_KEYS = ("fmp", "fred")


@click.group()
def vault() -> None:
    """Manage vault secrets"""
    pass


@vault.command()
@click.argument("key", type=click.Choice(API_KEYS))
@click.argument("value")
@click.pass_context
def add(ctx: click.Context, key: str, value: str) -> None:
    """Add an API key to the vault"""
    vault = from_context(ctx).vault
    vault.add(key, value)


@vault.command()
@click.argument("key")
@click.pass_context
def delete(ctx: click.Context, key: str) -> None:
    """Delete an API key from the vault"""
    app = from_context(ctx)
    if app.vault.delete(key):
        app.print(f"Deleted key {key}")
    else:
        app.error(f"Key {key} not found")


@vault.command()
@click.argument("key")
@click.pass_context
def show(ctx: click.Context, key: str) -> None:
    """Show the value of an API key"""
    app = from_context(ctx)
    if key := app.vault.get(key):
        app.print(key)
    else:
        app.error(f"Key {key} not found")


@vault.command()
@click.pass_context
def keys(ctx: click.Context) -> None:
    """Show the keys in the vault"""
    app = from_context(ctx)
    for key in app.vault.keys():
        app.print(key)
