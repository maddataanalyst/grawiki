"""Database adapters and abstractions."""

from src.grawiki.db.base import GraphDB
from src.grawiki.db.falkordb import FalkorGraphDB

__all__ = ["GraphDB", "FalkorGraphDB"]
