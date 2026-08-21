# routers/finance_ledger.py
"""
Finance Ledger

Endpoints:
  GET  /finance/ledger                          → Full transaction table with filters
  PATCH /finance/transactions/{id}/category     → Inline category correction (HTMX)
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import select, extract, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.templating import templates
from database import get_db
from domains.finance.models import Account, Category, Transaction
from domains.finance.queries import get_active_categories

log = logging.getLogger(__name__)

router = APIRouter(prefix="/finance", tags=["Finance"])


@router.get("/ledger", response_class=HTMLResponse)
async def finance_ledger(
    request: Request,
    # Accept as Optional[str] to handle empty string submitted by the form
    account_id: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    month: Optional[int] = Query(default=None),
    year: Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    today = datetime.date.today()

    # Safely coerce account_id — empty string from form submit becomes None
    acct_id: Optional[int] = int(account_id) if account_id and account_id.strip() else None

    stmt = select(Transaction).order_by(desc(Transaction.date), desc(Transaction.id))

    if acct_id:
        stmt = stmt.where(Transaction.account_id == acct_id)
    if category and category.strip():
        # Transaction.category is a Python property (backed by category_id),
        # not a queryable column — filter via a join on the real FK instead.
        stmt = stmt.join(Category, Transaction.category_id == Category.id).where(
            Category.name == category
        )
    if month:
        stmt = stmt.where(extract("month", Transaction.date) == month)
    if year:
        stmt = stmt.where(extract("year", Transaction.date) == (year or today.year))

    result = await db.execute(stmt.limit(500))
    transactions = result.scalars().all()

    accounts_result = await db.execute(select(Account).order_by(Account.name))
    accounts = accounts_result.scalars().all()

    categories = await get_active_categories(db)

    return templates.TemplateResponse(
        "finance_ledger.html",
        {
            "request": request,
            "transactions": transactions,
            "accounts": accounts,
            "categories": categories,
            "sel_account": acct_id,
            "sel_category": category,
            "sel_month": month or today.month,
            "sel_year": year or today.year,
            "active_module": "finance_ledger",
        },
    )


@router.patch("/transactions/{txn_id}/category", response_class=HTMLResponse)
async def update_category(
    txn_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Inline category correction triggered by the ledger's editCat() JS function.
    Returns a single <td> HTML fragment for HTMX to swap in place.
    """
    form = await request.form()
    category_str = str(form.get("category", "")).strip()

    # Validate against DB — reject anything not in the active category list
    existing = await db.execute(
        select(Category).where(Category.name == category_str)
    )
    cat = existing.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=422, detail="Invalid category")

    txn = await db.get(Transaction, txn_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    txn.category_id = cat.id
    await db.commit()

    slug = category_str.lower().replace(" ", "-").replace("&", "").replace("--", "-")
    return HTMLResponse(
        f'<td class="cat-cell" id="cat-{txn_id}" onclick="editCat({txn_id}, this)">'
        f'<span class="cat-badge cat-{slug}">{category_str}</span></td>'
    )
