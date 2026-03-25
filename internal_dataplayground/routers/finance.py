# routers/finance.py
"""
Finance Module — CSV import with Gemini auto-categorisation.

Endpoints:
  GET  /finance              → Summary dashboard (monthly income vs spend)
  GET  /finance/ledger       → Full transaction ledger with filters
  GET  /finance/upload       → CSV upload form
  POST /finance/upload       → Process CSV + Gemini categorise
  POST /finance/accounts     → Create a new account
  GET  /finance/accounts     → JSON list of accounts (for form selects)
  PATCH /finance/transactions/{id}/category  → Manual category correction
"""

import csv
import io
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Optional

import google.generativeai as genai
from fastapi import APIRouter, Depends, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, extract, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, get_key
from models import Account, AccountCreate, AccountType, Transaction, TransactionCategory

log = logging.getLogger(__name__)

router = APIRouter(prefix="/finance", tags=["Finance"])
templates = Jinja2Templates(directory="templates")

CATEGORIES = [c.value for c in TransactionCategory]

# ── Gemini helper ──────────────────────────────────────────────────────────────

def _get_gemini_model():
    api_key = get_key("GeminiAPIKey")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash-lite")


def _categorise_batch(rows: list[dict]) -> list[str]:
    """
    Send up to 200 (description, amount) pairs to Gemini.
    Returns a parallel list of category strings.
    Falls back to 'Other' on any error.
    """
    if not rows:
        return []

    model = _get_gemini_model()
    numbered = "\n".join(
        f"{i+1}. description=\"{r['description']}\" amount={r['amount']}"
        for i, r in enumerate(rows)
    )
    prompt = f"""You are a personal finance categoriser.
Assign each transaction exactly one category from this list:
{", ".join(CATEGORIES)}

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

Respond ONLY with a JSON array of {len(rows)} strings in the same order, e.g.:
["Food & Dining", "Transport", ...]
No explanations, no markdown, no extra text."""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        categories = json.loads(raw)
        # Validate every returned value
        valid = {c.value for c in TransactionCategory}
        return [c if c in valid else "Other" for c in categories]
    except Exception as exc:
        log.warning("Gemini categorisation failed: %s", exc)
        return ["Other"] * len(rows)


# ── Account helpers ────────────────────────────────────────────────────────────

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
    await db.refresh(acct)

    # Return updated account dropdown fragment for HTMX swap
    result = await db.execute(select(Account).order_by(Account.name))
    accounts = result.scalars().all()
    return templates.TemplateResponse(
        "partials/account_options.html",
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
    """
    1. Parse the uploaded CSV using user-selected column names.
    2. Batch-send descriptions to Gemini for categorisation.
    3. Bulk-insert transactions.
    4. Return an HTMX partial with an import summary.
    """
    # Validate account exists
    account = await db.get(Account, account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")  # strip BOM if present (common in bank CSVs)
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    rows_raw = list(reader)

    if not rows_raw:
        return HTMLResponse('<p class="upload-error">⚠ CSV appears empty.</p>', status_code=422)

    # Validate columns exist
    headers = rows_raw[0].keys()
    for col, label in [(date_col, "Date"), (desc_col, "Description"), (amount_col, "Amount")]:
        if col not in headers:
            return HTMLResponse(
                f'<p class="upload-error">⚠ Column "{col}" not found. Available: {", ".join(headers)}</p>',
                status_code=422,
            )

    # Parse rows
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
        return HTMLResponse('<p class="upload-error">⚠ No valid rows parsed. Check column names.</p>', status_code=422)

    # Gemini categorisation in chunks of 150
    all_categories = []
    chunk_size = 150
    for i in range(0, len(parsed), chunk_size):
        chunk = parsed[i : i + chunk_size]
        all_categories.extend(_categorise_batch(chunk))

    # Bulk insert
    import datetime
    for row_data, category_str in zip(parsed, all_categories):
        try:
            # Try common date formats
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

            cat_enum = TransactionCategory(category_str) if category_str in [c.value for c in TransactionCategory] else TransactionCategory.OTHER
            txn = Transaction(
                account_id=account_id,
                date=date_val,
                description=row_data["description"],
                amount=row_data["amount"],
                category=cat_enum,
            )
            db.add(txn)
        except Exception:
            skipped += 1
            continue

    await db.commit()

    imported = len(parsed) - skipped
    return templates.TemplateResponse(
        "partials/upload_result.html",
        {
            "request": request,
            "imported": imported,
            "skipped": skipped,
            "account_name": account.name,
        },
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
    try:
        cat = TransactionCategory(category_str)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid category")

    txn = await db.get(Transaction, txn_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    txn.category = cat
    await db.commit()

    # Return just the updated <td> so HTMX can swap it
    return HTMLResponse(
        f'<td class="cat-cell" id="cat-{txn_id}">'
        f'<span class="cat-badge cat-{cat.value.lower().replace(" ", "-").replace("&", "").replace("--", "-")}">'
        f'{cat.value}</span></td>'
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

    # Monthly totals by category
    stmt = (
        select(
            Transaction.category,
            func.sum(Transaction.amount).label("total"),
        )
        .where(extract("month", Transaction.date) == sel_month)
        .where(extract("year", Transaction.date) == sel_year)
        .group_by(Transaction.category)
    )
    result = await db.execute(stmt)
    rows = result.all()

    category_totals = {r.category.value: float(r.total) for r in rows}
    total_income = sum(v for v in category_totals.values() if v > 0)
    total_expenses = sum(v for v in category_totals.values() if v < 0)
    net = total_income + total_expenses

    # Last 10 transactions preview
    recent_stmt = (
        select(Transaction)
        .order_by(desc(Transaction.date), desc(Transaction.id))
        .limit(10)
    )
    recent_result = await db.execute(recent_stmt)
    recent_txns = recent_result.scalars().all()

    # Account list for sidebar
    accounts_result = await db.execute(select(Account).order_by(Account.name))
    accounts = accounts_result.scalars().all()

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
            "categories": CATEGORIES,
        },
    )


# ── Ledger (full drilldown) ────────────────────────────────────────────────────

@router.get("/ledger", response_class=HTMLResponse)
async def finance_ledger(
    request: Request,
    account_id: Optional[int] = None,
    category: Optional[str] = None,
    month: Optional[int] = None,
    year: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    import datetime
    today = datetime.date.today()

    stmt = select(Transaction).order_by(desc(Transaction.date), desc(Transaction.id))

    if account_id:
        stmt = stmt.where(Transaction.account_id == account_id)
    if category:
        try:
            cat_enum = TransactionCategory(category)
            stmt = stmt.where(Transaction.category == cat_enum)
        except ValueError:
            pass
    if month:
        stmt = stmt.where(extract("month", Transaction.date) == month)
    if year:
        stmt = stmt.where(extract("year", Transaction.date) == (year or today.year))

    result = await db.execute(stmt.limit(500))
    transactions = result.scalars().all()

    accounts_result = await db.execute(select(Account).order_by(Account.name))
    accounts = accounts_result.scalars().all()

    return templates.TemplateResponse(
        "finance_ledger.html",
        {
            "request": request,
            "transactions": transactions,
            "accounts": accounts,
            "categories": CATEGORIES,
            "sel_account": account_id,
            "sel_category": category,
            "sel_month": month or today.month,
            "sel_year": year or today.year,
        },
    )
