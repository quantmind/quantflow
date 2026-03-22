from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd
from ccy.cli.console import df_to_rich
from mcp.server.fastmcp.exceptions import ToolError
from rich.console import Console

from quantflow.data.fmp import FMP
from quantflow.data.fred import Fred
from quantflow.data.vault import Vault

VAULT_PATH = Path.home() / ".quantflow" / ".vault"


@dataclass
class McpTool:
    vault: Vault = field(default_factory=lambda: Vault(VAULT_PATH))

    def fmp(self) -> FMP:
        key = self.vault.get("fmp")
        if not key:
            raise ToolError(
                "FMP API key not found in vault. "
                " Please add it using the vault_add tool."
            )
        return FMP(key=key)

    def fred(self) -> Fred:
        key = self.vault.get("fred")
        if not key:
            raise ToolError(
                "FRED API key not found in vault. "
                " Please add it using the vault_add tool."
            )
        return Fred(key=key)

    def rich(self, df: pd.DataFrame) -> str:
        table = df_to_rich(df)
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        console.print(table)
        return buf.getvalue()
