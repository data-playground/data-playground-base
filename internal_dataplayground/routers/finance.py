# routers/finance.py
"""
Finance Module — dynamic categories, settings page, filter bug fix.

Endpoints:
  GET  /finance                          → Summary dashboard
  GET  /finance/ledger                   → Full transaction ledger
  GET  /finance/upload                   → CSV upload form
  POST /finance/upload                   → Process CSV + Gemma categorise
  GET  /finance/settings                 → Admin: accounts + categories
  POST /finance/accounts                 → Create account
  DELETE /finance/accounts/{id}          → Deactivate account
  POST /finance/categories               → Create category
  PATCH /finance/categories/{id}/toggle  → Toggle active
  PATCH /finance/transactions/{id}/category → Manual category correction
"""

import csv
import io
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Optional

from google import genai
from google.genai import types
from fastapi import APIRouter, Depends, HTTPException, Request, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, extract, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, get_key
from models import Account, AccountType, Category, Transaction

log = logging.getLogger(__name__)

router = APIRouter(prefix="/finance", tags=["Finance"])
templates = Jinja2Templates(directory="templates")


# ── Gemma helper ───────────────────────────────────────────────────────────────

def _get_client():
    return genai.Client(api_key=get_key("Gemini-API"))


async def _get_active_categories(db: AsyncSession) -> list[str]:
    result = await db.execute(
        select(Category.name)
        .where(Category.is_active == True)
        .order_by(Category.name)
    )
    return [r[0] for r in result.all()]


def _categorise_batch(rows: list[dict], categories: list[str]) -> list[str]:
    if not rows:
        return []

    client = _get_client()
    cat_list = ", ".join(categories)
    numbered = "\n".join(
        f"{i+1}. description=\"{r['description']}\" amount={r['amount']}"
        for i, r in enumerate(rows)
    )
    prompt = f"""You are a personal finance categoriser.
Assign each transaction exactly one category from this list:
{cat_list}

Rules:
- Negative amounts are expenses; positive amounts are income or transfers.
- Payroll/salary → Income
- Rent/mortgage/utilities → Housing
- Restaurants/groceries/coffee → Food & Dining
- Uber/Lyft/gas/subway/parking → Transport
- Netflix/Spotify/gym memberships/recurring SaaS → Subscriptions
- Doctor/pharmacy/insurance → Health
- Movies/concerts/hobbies → Entertainment
- Transfers to savings or investment accounts → Savings Transfer
- Anything that does not fit → Other

Transactions:
{numbered}

Respond ONLY with a JSON array of exactly {len(rows)} strings in order.
Example: ["Food & Dining", "Transport", "Income"]
No explanation, no markdown, no code blocks. Raw JSON array only."""

    try:
        response = _get_client().models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        categories_result = json.loads(raw)

        if not isinstance(categories_result, list):
            raise ValueError("Response is not a list")

        if len(categories_result) != len(rows):
            log.warning("Gemma returned %d categories for %d rows — falling back",
                       len(categories_result), len(rows))
            return ["Other"] * len(rows)

        valid = set(categories)
        return [c if c in valid else "Other" for c in categories_result]

    except Exception as exc:
        log.warning("Gemini categorisation failed: %s", exc)
        return ["Other"] * len(rows)


# ── Settings page ──────────────────────────────────────────────────────────────

@router.get("/settings", response_class=HTMLResponse)
async def finance_settings(request: Request, db: AsyncSession = Depends(get_db)):
    accounts_result = await db.execute(select(Account).order_by(Account.name))
    accounts = accounts_result.scalars().all()

    categories_result = await db.execute(select(Category).order_by(Category.name))
    categories = categories_result.scalars().all()

    return templates.TemplateResponse(
        "finance_settings.html",
        {"request": request, "accounts": accounts, "categories": categories},
    )


# ── Category CRUD ──────────────────────────────────────────────────────────────

@router.post("/categories", response_class=HTMLResponse)
async def create_category(
    request: Request,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    # Check for duplicate
    existing = await db.execute(select(Category).where(Category.name == name.strip()))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Category '{name}' already exists")

    cat = Category(name=name.strip(), description=description or None, is_active=True)
    db.add(cat)
    await db.commit()
    await db.refresh(cat)

    # Return updated category list partial
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


# ── Account CRUD ───────────────────────────────────────────────────────────────

@router.get("/accounts", response_class=JSONResponse)
async def list_accounts(db: AsyncSession = Depends(get_db)):
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


# ── CSV Upload ─────────────────────────────────────────────────────────────────

@router.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Account).order_by(Account.name))
    accounts = result.scalars().all()
    return templates.TemplateResponse(
        "finance_upload.html",
        {"request": request, "accounts": accounts},
    )


@router.post("/upload", response_class=HTMLResponse)
async def process_csv(
    request: Request,
    account_id: int = Form(...),
    date_col: str = Form(...),
    desc_col: str = Form(...),
    amount_col: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    account = await db.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    rows_raw = list(reader)

    if not rows_raw:
        return HTMLResponse('<p class="upload-error">⚠ CSV appears empty.</p>', status_code=422)

    headers = rows_raw[0].keys()
    for col, label in [(date_col, "Date"), (desc_col, "Description"), (amount_col, "Amount")]:
        if col not in headers:
            return HTMLResponse(
                f'<p class="upload-error">⚠ Column "{col}" not found. Available: {", ".join(headers)}</p>',
                status_code=422,
            )

    parsed = []
    skipped = 0
    for row in rows_raw:
        try:
            amt_str = row[amount_col].replace("$", "").replace(",", "").strip()
            amount = Decimal(amt_str)
            parsed.append({
                "date": row[date_col].strip(),
                "description": row[desc_col].strip()[:500],
                "amount": amount,
            })
        except (InvalidOperation, KeyError):
            skipped += 1
            continue

    if not parsed:
        return HTMLResponse('<p class="upload-error">⚠ No valid rows parsed.</p>', status_code=422)

    # Fetch active categories from DB so custom ones are included
    active_categories = await _get_active_categories(db)

    all_categories = []
    chunk_size = 150
    for i in range(0, len(parsed), chunk_size):
        chunk = parsed[i: i + chunk_size]
        all_categories.extend(_categorise_batch(chunk, active_categories))

    import datetime
    for row_data, category_str in zip(parsed, all_categories):
        try:
            date_val = None
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"):
                try:
                    date_val = datetime.datetime.strptime(row_data["date"], fmt).date()
                    break
                except ValueError:
                    continue
            if date_val is None:
                skipped += 1
                continue

            # Validate category exists, fall back to Other
            if category_str not in active_categories:
                category_str = "Other"

            txn = Transaction(
                account_id=account_id,
                date=date_val,
                description=row_data["description"],
                amount=row_data["amount"],
                category=category_str,
            )
            db.add(txn)
        except Exception:
            skipped += 1
            continue

    await db.commit()

    imported = len(parsed) - skipped
    return templates.TemplateResponse(
        "partials/upload_result.html",
        {"request": request, "imported": imported, "skipped": skipped, "account_name": account.name},
    )


# ── Manual category correction ─────────────────────────────────────────────────

@router.patch("/transactions/{txn_id}/category", response_class=HTMLResponse)
async def update_category(
    txn_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    category_str = str(form.get("category", "")).strip()

    # Validate against DB categories
    existing = await db.execute(select(Category).where(Category.name == category_str))
    if not existing.scalar_one_or_none():
        raise HTTPException(status_code=422, detail="Invalid category")

    txn = await db.get(Transaction, txn_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    txn.category = category_str
    await db.commit()

    slug = category_str.lower().replace(" ", "-").replace("&", "").replace("--", "-")
    return HTMLResponse(
        f'<td class="cat-cell" id="cat-{txn_id}" onclick="editCat({txn_id}, this)">'
        f'<span class="cat-badge cat-{slug}">{category_str}</span></td>'
    )


# ── Summary dashboard ──────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def finance_summary(
    request: Request,
    month: Optional[int] = None,
    year: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    import datetime
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
        select(Transaction).order_by(desc(Transaction.date), desc(Transaction.id)).limit(10)
    )
    recent_txns = recent_result.scalars().all()

    accounts_result = await db.execute(select(Account).order_by(Account.name))
    accounts = accounts_result.scalars().all()

    categories = await _get_active_categories(db)

    return templates.TemplateResponse(
        "finance.html",
        {
            "request": request,
            "sel_month": sel_month, "sel_year": sel_year,
            "category_totals": category_totals,
            "total_income": total_income, "total_expenses": total_expenses, "net": net,
            "recent_txns": recent_txns, "accounts": accounts, "categories": categories,
        },
    )


# ── Ledger ─────────────────────────────────────────────────────────────────────

@router.get("/ledger", response_class=HTMLResponse)
async def finance_ledger(
    request: Request,
    # FIX: accept as Optional[str] to handle empty string from form
    account_id: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    month: Optional[int] = Query(default=None),
    year: Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    import datetime
    today = datetime.date.today()

    # Safely coerce account_id
    acct_id: Optional[int] = int(account_id) if account_id and account_id.strip() else None

    stmt = select(Transaction).order_by(desc(Transaction.date), desc(Transaction.id))

    if acct_id:
        stmt = stmt.where(Transaction.account_id == acct_id)
    if category and category.strip():
        stmt = stmt.where(Transaction.category == category)
    if month:
        stmt = stmt.where(extract("month", Transaction.date) == month)
    if year:
        stmt = stmt.where(extract("year", Transaction.date) == (year or today.year))

    result = await db.execute(stmt.limit(500))
    transactions = result.scalars().all()

    accounts_result = await db.execute(select(Account).order_by(Account.name))
    accounts = accounts_result.scalars().all()

    categories = await _get_active_categories(db)

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
        },
    )
