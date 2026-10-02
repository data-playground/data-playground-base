"""
Explorer Settings — assign tables to domains

Endpoints:
  GET  /explorer/settings              → settings page
  POST /explorer/settings/assign       → set/clear one table's domain, returns one <tr>
  POST /explorer/settings/bulk-assign  → set/clear many, returns the full <tbody> contents

Domain input is normalized to lowercase [a-z0-9_]. Blank (or "unassigned")
clears the assignment, which puts the table back in the explorer's
"unassigned" group.

New tables need no code or migration: they appear here automatically (the
table list comes from information_schema) with a blank domain.
"""
import logging
import re
from typing import List, Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from core.templating import templates
from database import get_db
from domains.explorer.models import ExplorerTableDomain
from domains.explorer.table_domains import UNASSIGNED, get_domain_map, list_tables
from routers._helpers import html_error

log = logging.getLogger(__name__)

router = APIRouter(prefix="/explorer/settings", tags=["Explorer Settings"])

_DOMAIN_RE = re.compile(r"^[a-z0-9_]{1,50}$")
_BAD_DOMAIN_MSG = "Domain must be 1–50 characters: letters, numbers, underscores."
_SAVE_FAILED_MSG = "Could not save — has the explorer_table_domains migration been applied?"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_domain(raw: str) -> Optional[str]:
    """Returns the cleaned domain, "" meaning 'clear the assignment', or None if invalid."""
    d = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    if d in ("", UNASSIGNED):
        return ""
    return d if _DOMAIN_RE.match(d) else None


async def _build_rows(db: AsyncSession) -> list[dict]:
    """Unassigned first, then by domain, then by table name."""
    domain_map = await get_domain_map(db)
    rows = [{"name": t, "domain": domain_map.get(t, "")} for t in await list_tables(db)]
    rows.sort(key=lambda r: (r["domain"] != "", r["domain"], r["name"]))
    return rows


async def _apply(db: AsyncSession, names: list[str], domain: str) -> bool:
    """
    Set (or, for domain == "", clear) the mapping for each table.
    Returns False (after rolling back) if the write fails — most likely because
    the explorer_table_domains migration hasn't been applied yet.

    Uses merge() — a select-then-insert/update by primary key — rather than a
    MySQL-specific upsert, so it's dialect-neutral. At <100 rows that's fine.
    """
    try:
        if domain == "":
            await db.execute(
                delete(ExplorerTableDomain).where(ExplorerTableDomain.table_name.in_(names))
            )
        else:
            for n in names:
                await db.merge(ExplorerTableDomain(table_name=n, domain=domain))
        await db.commit()
        return True
    except SQLAlchemyError:
        log.exception("Explorer settings: could not save table->domain assignment")
        await db.rollback()
        return False


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def settings_page(request: Request, db: AsyncSession = Depends(get_db)):
    rows = await _build_rows(db)
    return templates.TemplateResponse(
        "explorer_settings.html",
        {
            "request": request,
            "active_module": "explorer",
            "tables": rows,
            "domains": sorted({r["domain"] for r in rows if r["domain"]}),
        },
    )


@router.post("/assign", response_class=HTMLResponse)
async def assign_one(
    request: Request,
    table_name: str = Form(...),
    domain: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    normalized = _normalize_domain(domain)
    if normalized is None:
        return html_error(_BAD_DOMAIN_MSG, status_code=422)
    if table_name not in await list_tables(db):
        return html_error("Unknown table.", status_code=404)

    if not await _apply(db, [table_name], normalized):
        return html_error(_SAVE_FAILED_MSG, status_code=500)
    return templates.TemplateResponse(
        "partials/explorer_settings_row.html",
        {"request": request, "t": {"name": table_name, "domain": normalized}},
    )


@router.post("/bulk-assign", response_class=HTMLResponse)
async def assign_many(
    request: Request,
    table_names: List[str] = Form(default=[]),
    domain: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    normalized = _normalize_domain(domain)
    if normalized is None:
        return html_error(_BAD_DOMAIN_MSG, status_code=422)

    valid = set(await list_tables(db))
    names = [n for n in table_names if n in valid]
    if names and not await _apply(db, names, normalized):
        return html_error(_SAVE_FAILED_MSG, status_code=500)

    return templates.TemplateResponse(
        "partials/explorer_settings_rows.html",
        {"request": request, "tables": await _build_rows(db)},
    )
