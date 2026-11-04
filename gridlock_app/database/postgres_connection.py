"""PostgreSQL Thread-Safe Connection Pool & Client for Project Gridlock (Neon & Local).

Handles relational database transactions for Users, Zones, Sensor records,
Incidents, and Officer Dispatches with Neon Serverless Postgres compatibility.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

# Load environment configuration from .env
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Database connection URL (Neon or Local Postgres)
DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback parameters if DATABASE_URL is not set
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
PG_DB = os.getenv("POSTGRES_DB", "gridlock")
PG_USER = os.getenv("POSTGRES_USER", "gridlock_admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "gridlock_secret_2026")


class PostgresClient:
    """Thread-safe connection pool client for PostgreSQL / Neon."""

    def __init__(
        self,
        database_url: Optional[str] = DATABASE_URL,
        minconn: int = 1,
        maxconn: int = 10,
    ):
        self.database_url = database_url
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._init_pool(minconn, maxconn)

    def _init_pool(self, minconn: int, maxconn: int) -> None:
        try:
            if self.database_url:
                # Connect via DSN (Neon pooled connection)
                self._pool = pool.ThreadedConnectionPool(
                    minconn,
                    maxconn,
                    dsn=self.database_url,
                )
            else:
                self._pool = pool.ThreadedConnectionPool(
                    minconn,
                    maxconn,
                    host=PG_HOST,
                    port=PG_PORT,
                    database=PG_DB,
                    user=PG_USER,
                    password=PG_PASSWORD,
                )
        except Exception as err:
            print(f"[PostgresClient] Pool init warning: {err}")
            self._pool = None

    def is_connected(self) -> bool:
        """Returns True if database connection is active."""
        if not self._pool:
            try:
                self._init_pool(1, 5)
            except Exception:
                return False
        if not self._pool:
            return False
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    return True
        except Exception:
            return False

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Context manager yielding a pooled connection with automatic release."""
        if not self._pool:
            self._init_pool(1, 5)
        if not self._pool:
            raise ConnectionError(f"Cannot connect to PostgreSQL via {self.database_url or PG_HOST}")

        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def execute(self, query: str, params: tuple = ()) -> Any:
        """Executes INSERT/UPDATE/DELETE query and returns RETURNING id or affected count."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                conn.commit()
                try:
                    res = cur.fetchone()
                    return res["id"] if res and "id" in res else cur.rowcount
                except psycopg2.ProgrammingError:
                    return cur.rowcount

    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Executes a query and returns a single row as a dictionary."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                return dict(row) if row else None

    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Executes a query and returns all matching rows as dictionaries."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def close(self) -> None:
        """Closes all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None


# Global singleton instance
pg_client = PostgresClient()
