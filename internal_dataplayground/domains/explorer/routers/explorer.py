"""
SQL Explorer — Read-Only Query Interface

Endpoints:
  GET  /explorer          → The BigQuery-style UI
  GET  /explorer/schema   → Returns all tables + columns + row counts as JSON
  POST /explorer/query    → Executes a validated SELECT query, returns rows as JSON

Security:
  - Keyword blocklist rejects any query containing write operations
  - Row cap of 500 rows prevents accidental full-table dumps
"""

import re
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/explorer", tags=["Explorer"])

# ── Security ───────────────────────────────────────────────────────────────────

# Any query containing these keywords is rejected before hitting the DB.
# The DB user should also have SELECT-only grants as a second layer of defense.
BLOCKED_PATTERN = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|'
    r'GRANT|REVOKE|LOAD\s+DATA|CALL|EXEC(?:UTE)?|'
    r'INTO\s+OUTFILE|INTO\s+DUMPFILE)\b',
    re.IGNORECASE
)

# Maximum rows returned — prevents accidental full-table dumps
ROW_CAP = 500

# Tables to hide from the schema browser (internal Airflow/Alembic metadata)
HIDDEN_TABLES = {
    "alembic_version",
    "dag", "dag_run", "task_instance", "job", "log",
    "xcom", "serialized_dag", "import_error",
}


class QueryRequest(BaseModel):
    sql: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_sql(sql: str) -> Optional[str]:
    """
    Returns an error message string if the query is blocked, else None.
    Strips comments before checking so -- INSERT or /* DROP */ don't sneak through.
    """
    # Strip single-line comments
    stripped = re.sub(r'--[^\n]*', '', sql)
    # Strip block comments
    stripped = re.sub(r'/\*.*?\*/', '', stripped, flags=re.DOTALL)
    stripped = stripped.strip()

    if not stripped:
        return "Query is empty."

    match = BLOCKED_PATTERN.search(stripped)
    if match:
        return f"Write operations are not permitted. Detected: {match.group(0).upper()}"

    # Must start with SELECT (after stripping comments and whitespace)
    first_word = stripped.split()[0].upper()
    if first_word not in ("SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"):
        return f"Only SELECT queries are permitted. Got: {first_word}"

    return None


def _infer_column_type(type_str: str) -> str:
    """Simplifies raw MariaDB type strings into short display labels."""
    t = type_str.upper()
    if any(k in t for k in ["INT", "BIGINT", "SMALLINT", "TINYINT"]): return "int"
    if any(k in t for k in ["DECIMAL", "FLOAT", "DOUBLE", "NUMERIC"]): return "num"
    if any(k in t for k in ["VARCHAR", "CHAR", "TEXT", "ENUM"]): return "str"
    if any(k in t for k in ["DATE", "TIME", "DATETIME", "TIMESTAMP"]): return "date"
    if "BOOL" in t: return "bool"
    if "JSON" in t: return "json"
    return "str"


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def explorer_ui(request: Request):
    return templates.TemplateResponse(
        "explorer.html",
        {"request": request, "active_module": "explorer"},
    )


@router.get("/schema")
async def get_schema(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """
    Returns the schema of all user tables as:
    {
      "table_name": {
        "columns": [{"name": str, "type": str, "is_pk": bool}],
        "row_count": int
      }
    }
    """
    schema: dict[str, Any] = {}

    # 1. Get all tables in the 'jobs' database
    tables_result = await db.execute(
        text("SELECT TABLE_NAME FROM information_schema.TABLES "
             "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE' "
             "ORDER BY TABLE_NAME")
    )
    tables = [row[0] for row in tables_result.fetchall() if row[0] not in HIDDEN_TABLES]

    for table in tables:
        # 2. Get columns for each table
        cols_result = await db.execute(
            text(
                "SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_KEY "
                "FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table "
                "ORDER BY ORDINAL_POSITION"
            ),
            {"table": table}
        )
        columns = [
            {
                "name": row[0],
                "type": _infer_column_type(row[1]),
                "is_pk": row[2] == "PRI",
            }
            for row in cols_result.fetchall()
        ]

        # 3. Get approximate row count (fast — uses engine stats)
        try:
            count_result = await db.execute(
                text(
                    "SELECT TABLE_ROWS FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table"
                ),
                {"table": table}
            )
            row_count = count_result.scalar() or 0
        except Exception:
            row_count = 0

        schema[table] = {"columns": columns, "row_count": row_count}

    return JSONResponse(content=schema)


@router.post("/query")
async def run_query(payload: QueryRequest, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """
    Executes a validated read-only SQL query.
    Returns: { rows: [...], columns: [...], row_count: int, capped: bool }
    """
    sql = payload.sql.strip()

    # Server-side validation (client also validates, but never trust the client)
    error = _validate_sql(sql)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Inject a LIMIT cap if the query doesn't already have one
    # This is a safety net, not a replacement for the user adding LIMIT
    sql_upper = sql.upper()
    has_limit = bool(re.search(r'\bLIMIT\b', sql_upper))

    # Wrap in a subquery with hard cap to prevent runaway queries
    if not has_limit:
        safe_sql = f"SELECT * FROM ({sql.rstrip(';')}) AS _explorer_subq LIMIT {ROW_CAP}"
        capped = True
    else:
        safe_sql = sql.rstrip(';')
        capped = False

    try:
        result = await db.execute(text(safe_sql))
        raw_rows = result.fetchall()
        columns = list(result.keys())

        # Serialize rows — handle non-JSON-serializable types (dates, Decimals)
        rows = []
        for row in raw_rows:
            serialized = {}
            for col, val in zip(columns, row):
                if val is None:
                    serialized[col] = None
                elif hasattr(val, 'isoformat'):
                    serialized[col] = val.isoformat()
                elif hasattr(val, '__float__'):
                    serialized[col] = float(val)
                else:
                    serialized[col] = str(val) if not isinstance(val, (int, float, bool, str)) else val
            rows.append(serialized)

        # Check if results were naturally capped by the injected LIMIT
        if not has_limit and len(rows) >= ROW_CAP:
            capped = True

        return JSONResponse(content={
            "rows": rows,
            "columns": columns,
            "row_count": len(rows),
            "capped": capped,
        })

    except Exception as exc:
        log.warning("Explorer query failed: %s | SQL: %s", exc, sql)
        raise HTTPException(status_code=400, detail=str(exc))
