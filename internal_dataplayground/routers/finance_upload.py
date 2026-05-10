# routers/finance_upload.py
"""
Finance CSV Upload

Endpoints:
  GET  /finance/upload   → Upload form (renders finance_upload.html)
  POST /finance/upload   → Process CSV + Gemini categorisation
"""

import csv
import io
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Optional

from google import genai
from fastapi import APIRouter, Depends, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, get_key
from models import Account, Category, Transaction

log = logging.getLogger(__name__)

router = APIRouter(prefix="/finance", tags=["Finance"])
templates = Jinja2Templates(directory="templates")


# ── Gemini helpers ─────────────────────────────────────────────────────────────

def _get_client():
    return genai.Client(api_key=get_key("Gemini-API"))


async def _get_active_categories(db: AsyncSession) -> list[str]:
    """Returns names of all active categories, ordered alphabetically."""
    result = await db.execute(
        select(Category.name)
        .where(Category.is_active == True)
        .order_by(Category.name)
    )
    return [r[0] for r in result.all()]


def _categorise_batch(rows: list[dict], categories: list[str]) -> list[str]:
    """
    Sends a batch of transactions to Gemini for categorisation.
    Returns a list of category strings in the same order as rows.
    Falls back to "Other" for any row that cannot be categorised.
    """
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
        response = client.models.generate_content(
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
            log.warning(
                "Gemini returned %d categories for %d rows — falling back",
                len(categories_result), len(rows),
            )
            return ["Other"] * len(rows)

        valid = set(categories)
        return [c if c in valid else "Other" for c in categories_result]

    except Exception as exc:
        log.warning("Gemini categorisation failed: %s", exc)
        return ["Other"] * len(rows)


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/upload", response_class=HTMLResponse)
async def upload_form(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Account).order_by(Account.name))
    accounts = result.scalars().all()
    return templates.TemplateResponse(
        "finance_upload.html",
        {
            "request": request,
            "accounts": accounts,
            "active_module": "finance_upload",
        },
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
    import datetime as dt

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
        return HTMLResponse(
            '<p class="upload-error">⚠ CSV appears empty.</p>',
            status_code=422,
        )

    headers = rows_raw[0].keys()
    for col, label in [(date_col, "Date"), (desc_col, "Description"), (amount_col, "Amount")]:
        if col not in headers:
            return HTMLResponse(
                f'<p class="upload-error">⚠ Column "{col}" not found. '
                f'Available: {", ".join(headers)}</p>',
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
        return HTMLResponse(
            '<p class="upload-error">⚠ No valid rows parsed.</p>',
            status_code=422,
        )

    # Fetch active categories from DB so custom ones are included
    active_categories = await _get_active_categories(db)

    all_categories = []
    chunk_size = 150
    for i in range(0, len(parsed), chunk_size):
        chunk = parsed[i: i + chunk_size]
        all_categories.extend(_categorise_batch(chunk, active_categories))

    for row_data, category_str in zip(parsed, all_categories):
        try:
            date_val = None
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"):
                try:
                    date_val = dt.datetime.strptime(row_data["date"], fmt).date()
                    break
                except ValueError:
                    continue
            if date_val is None:
                skipped += 1
                continue

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
        {
            "request": request,
            "imported": imported,
            "skipped": skipped,
            "account_name": account.name,
        },
    )
