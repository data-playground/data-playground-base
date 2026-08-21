# domains/finance/queries.py
"""
Shared read-helpers for the finance domain.

Extracted during the post-WO#5 cleanup pass to remove the
`_get_active_categories()` implementation that was previously duplicated
near-verbatim across finance_ledger.py, finance_summary.py, and
finance_upload.py. Pure refactor — no behavior change from the versions
it replaces.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from domains.finance.models import Category


async def get_active_categories(db: AsyncSession) -> list[str]:
    """Names of all active categories, ordered alphabetically."""
    result = await db.execute(
        select(Category.name).where(Category.is_active == True).order_by(Category.name)
    )
    return [r[0] for r in result.all()]


async def get_active_category_map(db: AsyncSession) -> dict[str, int]:
    """
    {category name: category id} for all active categories.

    Added alongside the category_id FK fix — used wherever a category
    *name* (e.g. Gemini's categorisation output, or a form field) needs
    to be resolved to a Transaction.category_id value before writing.
    """
    result = await db.execute(
        select(Category.id, Category.name).where(Category.is_active == True)
    )
    return {name: cid for cid, name in result.all()}
