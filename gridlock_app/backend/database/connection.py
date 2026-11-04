"""Database Connection Manager (Neon PostgreSQL & Local Fallback).

Provides a thread-safe connection pool connected directly to Neon Serverless Postgres
via DATABASE_URL, with seamless query translation and automatic connection pooling.
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
DB_DIR = Path(__file__).resolve().parent
DEFAULT_SQLITE_PATH = DB_DIR / "gridlock.db"
SCHEMA_SQLITE_PATH = DB_DIR / "schema_sqlite.sql"


class Database:
    """Universal Database client routing to Neon PostgreSQL or SQLite fallback."""

    def __init__(self):
        self.is_postgres = False
        self._pg_pool = None
        self.db_path = str(DEFAULT_SQLITE_PATH)

        if DATABASE_URL and ("neon.tech" in DATABASE_URL or "postgresql://" in DATABASE_URL):
            try:
                import psycopg2
                from psycopg2 import pool
                # Initialize Neon connection pool
                self._pg_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=DATABASE_URL,
                )
                # Verify live connection
                conn = self._pg_pool.getconn()
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                self._pg_pool.putconn(conn)
                self.is_postgres = True
                print(f"[Database] Successfully connected to Neon Serverless Postgres via pooled connection.")
            except Exception as err:
                print(f"[Database] Warning: Could not connect to Neon Postgres ({err}). Falling back to SQLite.")
                self.is_postgres = False

        if not self.is_postgres:
            self._init_sqlite()

    def init_db(self) -> None:
        """Initializes database schema if not present."""
        if not self.is_postgres:
            self._init_sqlite()
        else:
            schema_file = ROOT_DIR / "database" / "schema.sql"
            if schema_file.exists():
                with open(schema_file, "r", encoding="utf-8") as f:
                    sql = f.read()
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(sql)
                    conn.commit()

    def _init_sqlite(self) -> None:
        """Executes SQLite DDL schema if tables do not exist."""
        if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
            with sqlite3.connect(self.db_path) as conn:
                with open(SCHEMA_SQLITE_PATH, "r", encoding="utf-8") as f:
                    conn.executescript(f.read())
                conn.commit()

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        if self.is_postgres and self._pg_pool:
            conn = self._pg_pool.getconn()
            try:
                yield conn
            finally:
                self._pg_pool.putconn(conn)
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            try:
                yield conn
            finally:
                conn.close()

    def _format_query(self, query: str) -> str:
        """Translates SQLite query syntax to PostgreSQL syntax when connected to Neon."""
        if self.is_postgres:
            # Replace SQLite date functions with PostgreSQL equivalents
            formatted = query.replace("datetime('now', '+15 minutes')", "NOW() + INTERVAL '15 minutes'")
            formatted = formatted.replace("datetime('now')", "NOW()")
            # Replace ? with %s
            return formatted.replace("?", "%s")
        return query

    def execute(self, query: str, params: tuple = ()) -> int:
        """Executes INSERT/UPDATE/DELETE and returns lastrowid or affected count."""
        formatted = self._format_query(query)

        if self.is_postgres:
            from psycopg2.extras import RealDictCursor
            is_insert = formatted.strip().upper().startswith("INSERT")
            if is_insert and "RETURNING" not in formatted.upper():
                formatted = formatted.rstrip("; ") + " RETURNING id;"

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(formatted, params)
                    conn.commit()
                    if is_insert:
                        row = cur.fetchone()
                        return row["id"] if row and "id" in row else 0
                    return cur.rowcount
        else:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(formatted, params)
                conn.commit()
                return cursor.lastrowid

    def _clean_row(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Converts Decimal objects to float for seamless Python/JSON compatibility."""
        if not row:
            return None
        import decimal
        return {
            k: float(v) if isinstance(v, decimal.Decimal) else v
            for k, v in row.items()
        }

    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Executes SELECT and returns single row as dictionary."""
        formatted = self._format_query(query)

        if self.is_postgres:
            from psycopg2.extras import RealDictCursor
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(formatted, params)
                    row = cur.fetchone()
                    return self._clean_row(dict(row)) if row else None
        else:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(formatted, params)
                row = cursor.fetchone()
                return self._clean_row(dict(row)) if row else None

    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Executes SELECT and returns all rows as list of dictionaries."""
        formatted = self._format_query(query)

        if self.is_postgres:
            from psycopg2.extras import RealDictCursor
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(formatted, params)
                    rows = cur.fetchall()
                    return [self._clean_row(dict(r)) for r in rows]
        else:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(formatted, params)
                rows = cursor.fetchall()
                return [self._clean_row(dict(r)) for r in rows]


# Global singleton instance
db = Database()
