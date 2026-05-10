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
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, extract, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Account, Transaction

log = logging.getLogger(__name__)

router = APIRouter(prefix="/finance", tags=["Finance"])
templates = Jinja2Templates(directory="templates")


async def _get_active_categories(db: AsyncSession) -> list[str]:
    """Returns names of all active categories, ordered alphabetically."""
    from models import Category
    result = await db.execute(
        select(Category.name)
        .where(Category.is_active == True)
        .order_by(Category.name)
    )
    return [r[0] for r in result.all()]


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

    stmt = (
        select(Transaction.category, func.sum(Transaction.amount).label("total"))
        .where(extract("month", Transaction.date) == sel_month)
        .where(extract("year", Transaction.date) == sel_year)
        .group_by(Transaction.category)
    )
    result = await db.execute(stmt)
    rows = result.all()

    category_totals = {r.category: float(r.total) for r in rows}
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

    categories = await _get_active_categories(db)

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
