"""Tests for the public database package surface."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_grawiki_db_does_not_eagerly_import_falkordb() -> None:
    """Importing ``grawiki.db`` should not import optional backend modules."""

    sys.modules.pop("grawiki.db", None)
    sys.modules.pop("grawiki.db.falkordb", None)

    module = importlib.import_module("grawiki.db")

    assert hasattr(module, "GraphDB")
    assert "grawiki.db.falkordb" not in sys.modules


def test_missing_falkordb_extras_raise_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional backend access should explain how to install the extras."""

    module = importlib.import_module("grawiki.db")

    def fake_import_module(name: str):
        error = ModuleNotFoundError("No module named 'falkordb'")
        error.name = "falkordb"
        raise error

    monkeypatch.setattr(module, "import_module", fake_import_module)

    with pytest.raises(
        ModuleNotFoundError, match="grawiki\\[falkordb\\].*grawiki\\[falkordblite\\]"
    ):
        getattr(module, "FalkorGraphDB")
