from dataclasses import dataclass, field
from pathlib import Path

from mcp.server.fastmcp.exceptions import ToolError

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
