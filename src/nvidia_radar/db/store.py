"""Structured store for runs and startups.

Defaults to a local SQLite database (zero-config). If ``DATABASE_URL`` points at
PostgreSQL and ``psycopg`` is installed, it transparently uses Postgres instead —
the same schema and queries (only the parameter placeholder differs).
"""
from __future__ import annotations

import json
import sqlite3
import threading

from ..config import ensure_dirs, get_settings
from ..models import RadarRun, StartupProfile

_LOCK = threading.Lock()
_SQLITE_CONN: sqlite3.Connection | None = None

_DDL = [
    """CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        query TEXT,
        created_at TEXT,
        status TEXT,
        portfolio_summary TEXT,
        stats TEXT,
        errors TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS startups (
        run_id TEXT,
        name TEXT,
        classification TEXT,
        sector TEXT,
        maturity_overall INTEGER,
        lab_threat INTEGER,
        inception_fit INTEGER,
        created_at TEXT,
        profile TEXT,
        PRIMARY KEY (run_id, name)
    )""",
]


def _is_pg() -> bool:
    return get_settings().database_url.startswith(("postgres://", "postgresql://"))


def _get_conn():
    """Return (connection, placeholder). SQLite conn is reused; PG is per-call."""
    if _is_pg():
        import psycopg  # type: ignore

        return psycopg.connect(get_settings().database_url), "%s"
    global _SQLITE_CONN
    if _SQLITE_CONN is None:
        ensure_dirs()
        _SQLITE_CONN = sqlite3.connect(str(get_settings().sqlite_path), check_same_thread=False)
        _SQLITE_CONN.row_factory = sqlite3.Row
    return _SQLITE_CONN, "?"


def _q(sql: str, ph: str) -> str:
    return sql.replace("?", ph) if ph != "?" else sql


def init_db() -> None:
    conn, ph = _get_conn()
    with _LOCK:
        cur = conn.cursor()
        for ddl in _DDL:
            cur.execute(ddl)
        conn.commit()
        if _is_pg():
            conn.close()


def save_run(run: RadarRun) -> None:
    init_db()
    conn, ph = _get_conn()
    with _LOCK:
        cur = conn.cursor()
        cur.execute(_q("DELETE FROM startups WHERE run_id = ?", ph), (run.run_id,))
        cur.execute(_q("DELETE FROM runs WHERE run_id = ?", ph), (run.run_id,))
        cur.execute(
            _q("INSERT INTO runs (run_id, query, created_at, status, portfolio_summary, stats, errors)"
               " VALUES (?, ?, ?, ?, ?, ?, ?)", ph),
            (run.run_id, run.query, run.created_at, run.status,
             run.portfolio_summary, json.dumps(run.stats), json.dumps(run.errors)),
        )
        for p in run.startups:
            cur.execute(
                _q("INSERT INTO startups (run_id, name, classification, sector, maturity_overall,"
                   " lab_threat, inception_fit, created_at, profile) VALUES (?,?,?,?,?,?,?,?,?)", ph),
                (
                    run.run_id, p.name, p.classification, p.sector,
                    p.maturity.overall if p.maturity else None,
                    p.lab_threat.risk_score if p.lab_threat else None,
                    p.inception_fit, run.created_at, p.model_dump_json(),
                ),
            )
        conn.commit()
        if _is_pg():
            conn.close()


def list_runs(limit: int = 50) -> list[dict]:
    conn, ph = _get_conn()
    with _LOCK:
        cur = conn.cursor()
        cur.execute(
            _q("SELECT r.run_id, r.query, r.created_at, r.status, r.stats,"
               " (SELECT COUNT(*) FROM startups s WHERE s.run_id = r.run_id) AS n"
               " FROM runs r ORDER BY r.created_at DESC LIMIT ?", ph),
            (limit,),
        )
        rows = cur.fetchall()
        if _is_pg():
            conn.close()
    out = []
    for row in rows:
        run_id, query, created_at, status, stats, n = (
            (row["run_id"], row["query"], row["created_at"], row["status"], row["stats"], row["n"])
            if isinstance(row, sqlite3.Row) else tuple(row)
        )
        out.append({
            "run_id": run_id, "query": query, "created_at": created_at,
            "status": status, "n_startups": n,
            "stats": json.loads(stats) if stats else {},
        })
    return out


def get_run(run_id: str) -> RadarRun | None:
    conn, ph = _get_conn()
    with _LOCK:
        cur = conn.cursor()
        cur.execute(_q("SELECT run_id, query, created_at, status, portfolio_summary, stats, errors"
                       " FROM runs WHERE run_id = ?", ph), (run_id,))
        run_row = cur.fetchone()
        if not run_row:
            if _is_pg():
                conn.close()
            return None
        cur.execute(_q("SELECT profile FROM startups WHERE run_id = ? ORDER BY inception_fit DESC", ph),
                    (run_id,))
        startup_rows = cur.fetchall()
        if _is_pg():
            conn.close()

    r = dict(run_row) if isinstance(run_row, sqlite3.Row) else {
        k: v for k, v in zip(
            ["run_id", "query", "created_at", "status", "portfolio_summary", "stats", "errors"], run_row)
    }
    startups = [
        StartupProfile.model_validate_json(
            row["profile"] if isinstance(row, sqlite3.Row) else row[0]
        )
        for row in startup_rows
    ]
    return RadarRun(
        run_id=r["run_id"], query=r["query"], created_at=r["created_at"], status=r["status"],
        portfolio_summary=r["portfolio_summary"],
        stats=json.loads(r["stats"]) if r["stats"] else {},
        errors=json.loads(r["errors"]) if r["errors"] else [],
        startups=startups,
    )


def list_startups(limit: int = 200, classification: str | None = None) -> list[dict]:
    conn, ph = _get_conn()
    sql = ("SELECT s.run_id, s.name, s.classification, s.sector, s.maturity_overall,"
           " s.lab_threat, s.inception_fit, s.created_at, r.query"
           " FROM startups s JOIN runs r ON r.run_id = s.run_id")
    params: list = []
    if classification:
        sql += " WHERE s.classification = ?"
        params.append(classification)
    sql += " ORDER BY s.inception_fit DESC LIMIT ?"
    params.append(limit)
    with _LOCK:
        cur = conn.cursor()
        cur.execute(_q(sql, ph), tuple(params))
        rows = cur.fetchall()
        if _is_pg():
            conn.close()
    cols = ["run_id", "name", "classification", "sector", "maturity_overall",
            "lab_threat", "inception_fit", "created_at", "query"]
    return [
        dict(row) if isinstance(row, sqlite3.Row) else dict(zip(cols, row))
        for row in rows
    ]


def get_startup(run_id: str, name: str) -> StartupProfile | None:
    conn, ph = _get_conn()
    with _LOCK:
        cur = conn.cursor()
        cur.execute(_q("SELECT profile FROM startups WHERE run_id = ? AND name = ?", ph),
                    (run_id, name))
        row = cur.fetchone()
        if _is_pg():
            conn.close()
    if not row:
        return None
    profile = row["profile"] if isinstance(row, sqlite3.Row) else row[0]
    return StartupProfile.model_validate_json(profile)
