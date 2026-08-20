# routers/finance_settings.py
"""
Finance Settings — Accounts and Categories management

Endpoints:
  GET    /finance/settings              → Settings page
  GET    /finance/accounts              → Account list as JSON (for dropdown refresh)
  POST   /finance/accounts              → Create account
  DELETE /finance/accounts/{id}         → Delete account
  POST   /finance/categories            → Create category
  PATCH  /finance/categories/{id}/toggle → Toggle category active/inactive
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.templating import templates
from database import get_db
from domains.finance.models import Account, AccountType, Category

log = logging.getLogger(__name__)

router = APIRouter(prefix="/finance", tags=["Finance"])


# ── Settings page ──────────────────────────────────────────────────────────────

@router.get("/settings", response_class=HTMLResponse)
async def finance_settings(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    accounts_result = await db.execute(select(Account).order_by(Account.name))
    accounts = accounts_result.scalars().all()

    categories_result = await db.execute(select(Category).order_by(Category.name))
    categories = categories_result.scalars().all()

    return templates.TemplateResponse(
        "finance_settings.html",
        {
            "request": request,
            "accounts": accounts,
            "categories": categories,
            "active_module": "finance_settings",
        },
    )


# ── Account CRUD ───────────────────────────────────────────────────────────────

@router.get("/accounts", response_class=JSONResponse)
async def list_accounts(db: AsyncSession = Depends(get_db)):
    """JSON list used to refresh the account dropdown on the upload page."""
    result = await db.execute(select(Account).order_by(Account.name))
    accounts = result.scalars().all()
    return [
        {"id": a.id, "name": a.name, "type": a.account_type.value, "last_four": a.last_four}
        for a in accounts
    ]


@router.post("/accounts", response_class=HTMLResponse)
async def create_account(
    request: Request,
    name: str = Form(...),
    account_type: str = Form(...),
    last_four: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    try:
        atype = AccountType(account_type)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid account type")

    acct = Account(name=name, account_type=atype, last_four=last_four or None)
    db.add(acct)
    await db.commit()

    result = await db.execute(select(Account).order_by(Account.name))
    accounts = result.scalars().all()
    return templates.TemplateResponse(
        "partials/account_list.html",
        {"request": request, "accounts": accounts},
    )


@router.delete("/accounts/{acct_id}", response_class=HTMLResponse)
async def delete_account(
    acct_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    acct = await db.get(Account, acct_id)
    if not acct:
        raise HTTPException(status_code=404, detail="Account not found")
    await db.delete(acct)
    await db.commit()

    result = await db.execute(select(Account).order_by(Account.name))
    accounts = result.scalars().all()
    return templates.TemplateResponse(
        "partials/account_list.html",
        {"request": request, "accounts": accounts},
    )


# ── Category CRUD ──────────────────────────────────────────────────────────────

@router.post("/categories", response_class=HTMLResponse)
async def create_category(
    request: Request,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(
        select(Category).where(Category.name == name.strip())
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Category '{name}' already exists")

    cat = Category(name=name.strip(), description=description or None, is_active=True)
    db.add(cat)
    await db.commit()
    await db.refresh(cat)

    result = await db.execute(select(Category).order_by(Category.name))
    categories = result.scalars().all()
    return templates.TemplateResponse(
        "partials/category_list.html",
        {"request": request, "categories": categories},
    )


@router.patch("/categories/{cat_id}/toggle", response_class=HTMLResponse)
async def toggle_category(
    cat_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    cat = await db.get(Category, cat_id)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    cat.is_active = not cat.is_active
    await db.commit()

    result = await db.execute(select(Category).order_by(Category.name))
    categories = result.scalars().all()
    return templates.TemplateResponse(
        "partials/category_list.html",
        {"request": request, "categories": categories},
    )
