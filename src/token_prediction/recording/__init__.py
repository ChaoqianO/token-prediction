"""Append-only event recording and secret redaction."""

from .store import EventStoreError, SQLiteEventStore

__all__ = ["EventStoreError", "SQLiteEventStore"]
