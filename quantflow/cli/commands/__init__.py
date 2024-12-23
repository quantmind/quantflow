from .base import QuantContext, quant_group
from .crypto import crypto
from .fred import fred
from .stocks import stocks
from .vault import vault


@quant_group()
def quantflow() -> None:
    ctx = QuantContext.current()
    if ctx.invoked_subcommand is None:
        ctx.qf.print(ctx.get_help())


quantflow.add_command(vault)
quantflow.add_command(crypto)
quantflow.add_command(stocks)
quantflow.add_command(fred)
