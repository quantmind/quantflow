import tempfile
from pathlib import Path

import pytest

from quantflow.data.vault import Vault


@pytest.fixture
def vault():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Vault(Path(temp_dir) / "test_vault")


def test_vault(vault: Vault) -> None:
    assert vault.path
    assert not vault.data
    vault.add("hello", "world")
    assert vault.data
    assert vault.get("hello") == "world"

    v2 = Vault(vault.path)
    assert v2.path == vault.path
    assert v2.data == vault.data
    assert v2.get("hello") == "world"

    assert vault.keys() == ["hello"]

    vault.add("foo", "bar")
    assert vault.keys() == ["foo", "hello"]
    assert vault.delete("foo")
    assert vault.keys() == ["hello"]
    assert not vault.delete("foo")
    assert vault.keys() == ["hello"]
