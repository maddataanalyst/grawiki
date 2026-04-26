"""Database adapters and abstractions.

`GraphDB` is part of the base package surface. `FalkorGraphDB` is loaded lazily
so base installs do not require the optional Falkor dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from grawiki.db.base import GraphDB

if TYPE_CHECKING:
    from grawiki.db.falkordb import FalkorGraphDB

__all__ = ["GraphDB", "FalkorGraphDB"]


def __getattr__(name: str) -> Any:
    """Load optional database adapters on demand.

    Parameters
    ----------
    name : str
        Module attribute requested by the caller.

    Returns
    -------
    Any
        The requested adapter class.

    Raises
    ------
    AttributeError
        Raised when the attribute does not exist.
    ModuleNotFoundError
        Raised with a clear install hint when Falkor extras are missing.
    """

    if name != "FalkorGraphDB":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        return import_module("grawiki.db.falkordb").FalkorGraphDB
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".", maxsplit=1)[0]
        if missing_root in {"falkordb", "redislite", "redis"}:
            raise ModuleNotFoundError(
                "FalkorGraphDB requires optional Falkor dependencies. "
                "Install `grawiki[falkordb]` for server mode or "
                "`grawiki[falkordblite]` for the embedded file-backed mode."
            ) from exc
        raise


def __dir__() -> list[str]:
    """Return the module attributes exposed as the public surface."""

    return sorted(__all__)
