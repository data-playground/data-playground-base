# routers/finance_summary.py
"""
Finance Summary Dashboard

Endpoints:
  GET /finance   → Monthly KPI cards + category chart + recent transactions
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func, extract, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.templating import templates
from database import get_db
from domains.finance.models import Account, Category, Transaction
from domains.finance.queries import get_active_categories

log = logging.getLogger(__name__)

router = APIRouter(prefix="/finance", tags=["Finance"])


@router.get("", response_class=HTMLResponse)
async def finance_summary(
    request: Request,
    month: Optional[int] = None,
    year: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    today = datetime.date.today()
    sel_month = month or today.month
    sel_year = year or today.year

    # Outer join so transactions with no matching category_id (legacy rows
    # that couldn't be backfilled, or a category deleted out from under a
    # transaction) still show up, grouped under "Other" — matching the
    # pre-FK fallback behavior of Transaction.category.
    cat_name = func.coalesce(Category.name, "Other").label("category_name")
    stmt = (
        select(cat_name, func.sum(Transaction.amount).label("total"))
        .select_from(Transaction)
        .outerjoin(Category, Transaction.category_id == Category.id)
        .where(extract("month", Transaction.date) == sel_month)
        .where(extract("year", Transaction.date) == sel_year)
        .group_by(cat_name)
    )
    result = await db.execute(stmt)
    rows = result.all()

    category_totals = {r.category_name: float(r.total) for r in rows}
    total_income = sum(v for v in category_totals.values() if v > 0)
    total_expenses = sum(v for v in category_totals.values() if v < 0)
    net = total_income + total_expenses

    recent_result = await db.execute(
        select(Transaction)
        .order_by(desc(Transaction.date), desc(Transaction.id))
        .limit(10)
    )
    recent_txns = recent_result.scalars().all()

    accounts_result = await db.execute(select(Account).order_by(Account.name))
    accounts = accounts_result.scalars().all()

    categories = await get_active_categories(db)

    return templates.TemplateResponse(
        "finance.html",
        {
            "request": request,
            "sel_month": sel_month,
            "sel_year": sel_year,
            "category_totals": category_totals,
            "total_income": total_income,
            "total_expenses": total_expenses,
            "net": net,
            "recent_txns": recent_txns,
            "accounts": accounts,
            "categories": categories,
            "active_module": "finance",
        },
    )
