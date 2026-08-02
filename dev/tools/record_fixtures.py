"""Record market data fixtures from live sources.

Each recorder fetches live data, trims it down to the fields consumed by the
loaders and writes it to docs/examples/fixtures. The regularMarketTime entry
of the underlying quote is preserved so loaders can recover the as of date of
the snapshot.
"""

import asyncio
import gzip
import json
from datetime import datetime, timezone
from typing import Awaitable, Callable

import click

from docs.examples._utils import FIXTURES
from quantflow.data.deribit import Deribit, InstrumentKind
from quantflow.data.yahoo import Yahoo
from quantflow.utils.dates import utcnow

OPTION_FIELDS = ("strike", "bid", "ask", "openInterest", "volume")
QUOTE_FIELDS = ("bid", "ask", "regularMarketPrice", "regularMarketTime")

BOOK_FIELDS = ("instrument_name", "bid_price", "ask_price", "open_interest", "volume_usd")
INSTRUMENT_FIELDS = (
    "instrument_name",
    "tick_size",
    "settlement_period",
    "expiration_timestamp",
    "strike",
    "option_type",
)


def trim_chain(chain: dict) -> dict:
    """Trim a Yahoo option chain payload to the fields used by the loaders"""
    quote = chain.get("quote") or {}
    return {
        "underlyingSymbol": chain.get("underlyingSymbol", ""),
        "quote": {key: quote.get(key) for key in QUOTE_FIELDS},
        "options": [
            {
                "expirationDate": expiry["expirationDate"],
                "calls": trim_contracts(expiry.get("calls", [])),
                "puts": trim_contracts(expiry.get("puts", [])),
            }
            for expiry in chain.get("options", [])
        ],
    }


def trim_contracts(contracts: list[dict]) -> list[dict]:
    return [
        {key: c.get(key) for key in OPTION_FIELDS if c.get(key) is not None}
        for c in contracts
    ]


async def record_yahoo(dry_run: bool) -> None:
    """Record the SPX option chain fixture from Yahoo Finance"""
    async with Yahoo() as cli:
        chain = await cli.option_chain("^SPX")
    trimmed = trim_chain(chain)
    path = FIXTURES / "yahoo_spx.json.gz"
    expiries = len(trimmed["options"])
    as_of = "unknown"
    if market_time := trimmed["quote"].get("regularMarketTime"):
        as_of = datetime.fromtimestamp(market_time, tz=timezone.utc).isoformat()
    if dry_run:
        click.echo(
            f"dry run, would record {path} as of {as_of} with {expiries} expiries"
        )
        return
    path.write_bytes(gzip.compress(json.dumps(trimmed).encode()))
    click.echo(f"recorded {path} as of {as_of}")


def trim_records(records: list[dict], fields: tuple[str, ...]) -> list[dict]:
    # keep keys with a null value: the loaders expect them to be present
    return [{key: r[key] for key in fields if key in r} for r in records]


async def record_deribit(dry_run: bool) -> None:
    """Record the BTC inverse option chain fixture from Deribit.

    The fixture bundles the three payloads consumed by
    Deribit.volatility_surface_loader (instruments, futures book and options
    book) together with the snapshot time, so the loader can be rebuilt
    offline with a fixed reference date.
    """
    async with Deribit() as cli:
        futures = await cli.get_book_summary_by_currency(
            currency="btc", kind=InstrumentKind.FUTURE
        )
        options = await cli.get_book_summary_by_currency(
            currency="btc", kind=InstrumentKind.OPTION
        )
        instruments = await cli.get_instruments(currency="btc")
    as_of = utcnow()
    bundle = {
        "as_of": as_of.isoformat(),
        "futures": trim_records(futures, BOOK_FIELDS),
        "options": trim_records(options, BOOK_FIELDS),
        "instruments": trim_records(instruments, INSTRUMENT_FIELDS),
    }
    path = FIXTURES / "deribit_btc.json.gz"
    if dry_run:
        click.echo(
            f"dry run, would record {path} as of {as_of.isoformat()} with "
            f"{len(bundle['options'])} options and {len(bundle['futures'])} futures"
        )
        return
    path.write_bytes(gzip.compress(json.dumps(bundle).encode()))
    click.echo(f"recorded {path} as of {as_of.isoformat()}")


RECORDERS: dict[str, Callable[[bool], Awaitable[None]]] = {
    "deribit": record_deribit,
    "yahoo": record_yahoo,
}


@click.command()
@click.argument("sources", nargs=-1, type=click.Choice(sorted(RECORDERS)))
@click.option(
    "--dry-run", is_flag=True, help="Fetch and trim but do not write fixture files."
)
def main(sources: tuple[str, ...], dry_run: bool) -> None:
    """Record market data fixtures from live sources.

    With no arguments all fixtures are recorded, otherwise only the given
    SOURCES are recorded.
    """
    for name in sources or sorted(RECORDERS):
        asyncio.run(RECORDERS[name](dry_run))


if __name__ == "__main__":
    main()
