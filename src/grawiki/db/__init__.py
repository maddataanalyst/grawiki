"""Database adapters and abstractions."""

from grawiki.db.base import GraphDB
from grawiki.db.falkordb import FalkorGraphDB

__all__ = ["GraphDB", "FalkorGraphDB"]
