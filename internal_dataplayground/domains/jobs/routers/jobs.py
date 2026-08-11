import datetime

from database import get_db
from domains.jobs.models import ApplicationLog, ApplicationStatus, Job
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, desc, func, or_, select

from core.templating import templates
from routers._helpers import html_error

router = APIRouter(prefix="/jobs", tags=["Jobs"])

# Hard cap on rows returned per request, independent of how permissive the
# filters are. Before this existed, GET /jobs had no LIMIT at all and would
# fetch and render every row in linkedin_jobs on every load — with the
# table well past 2,000 rows (and rendering BOTH a desktop table and a
# separate mobile card list for every one of them), that was hanging the
# page for over a minute. Filters should usually keep results well under
# this on their own; this is the backstop for when they don't, so the
# problem can't silently reappear as the table keeps growing. Paired with
# the "⚡ Load More" button + cursor below rather than just truncating.
PAGE_SIZE = 300

# Defaults applied on the initial GET /jobs page load — deliberately
# mirrors what the old client-side filter defaults used to show (Quick
# Filter "High Fit", hide applied, last 10 days), so the first thing a
# person sees is unchanged. The difference is these are now applied in the
# SQL query itself instead of being computed by loading everything and
# hiding most of it with CSS.
DEFAULT_QUICK_FILTER = "high"
DEFAULT_SCORE_MIN = 90
DEFAULT_HIDE_APPLIED = True
DEFAULT_LOOKBACK_DAYS = 10


def _applied_job_ids_subquery():
    """Job IDs with at least one ApplicationLog row — i.e. "has been
    engaged with in the ATS at all", regardless of which status. Shared by
    the 'applied' / 'not_applied' quick filters and the 'hide applied'
    toggle, which all mean this same thing."""
    return select(ApplicationLog.job_id).distinct().scalar_subquery()


def _latest_status_job_ids_subquery(statuses: list[str]):
    """Job IDs whose MOST RECENT ApplicationLog status is one of `statuses`
    (status NAMEs, e.g. "PHONE_SCREEN") — the server-side equivalent of the
    latest_status_key property, for the status multi-select filter."""
    latest_log_subq = (
        select(
            ApplicationLog.job_id,
            func.max(ApplicationLog.created_at).label("max_created_at"),
        )
        .group_by(ApplicationLog.job_id)
        .subquery()
    )
    return (
        select(ApplicationLog.job_id)
        .join(
            latest_log_subq,
            (ApplicationLog.job_id == latest_log_subq.c.job_id)
            & (ApplicationLog.created_at == latest_log_subq.c.max_created_at),
        )
        .where(ApplicationLog.status.in_([ApplicationStatus[s] for s in statuses]))
        .scalar_subquery()
    )


def _parse_cursor(cursor: str | None) -> tuple[int, int] | None:
    """Decodes a "fit_score_id" keyset-pagination cursor. Returns None for
    an absent/malformed cursor, which callers should treat as "start from
    the beginning" — never as an error; a bad cursor from a stale bookmark
    or a manually-edited URL should just restart pagination, not 500."""
    if not cursor:
        return None
    try:
        score_str, id_str = cursor.split("_", 1)
        return int(score_str), int(id_str)
    except (ValueError, AttributeError):
        return None


def _build_job_query(
    quick: str,
    score_min: int,
    company: str | None,
    statuses: list[str],
    hide_applied: bool,
    date_from: str | None,
    date_to: str | None,
):
    """
    Builds the filtered (but not yet paginated) Job select — shared by the
    initial page load and every GET /jobs/rows filter/load-more request, so
    the two can never drift out of sync with each other. This is a direct
    SQL translation of the filter logic that used to live entirely in
    jobs.html's client-side applyAllFilters() — see that function's git
    history for the exact behavior being preserved, including quirks like
    quick='high' hardcoding score>=90 independent of the score_min slider.
    """
    conditions = []

    if quick == "high":
        conditions.append(Job.fit_score >= 90)
    elif quick == "remote":
        conditions.append(Job.remote.is_(True))
    elif quick == "applied":
        conditions.append(Job.ID.in_(_applied_job_ids_subquery()))
    elif quick == "not_applied":
        conditions.append(Job.ID.not_in(_applied_job_ids_subquery()))
    # quick == "all" (or anything unrecognized): no additional condition.

    conditions.append(Job.fit_score >= score_min)

    if company:
        conditions.append(Job.company_name.ilike(f"%{company}%"))

    if date_from:
        conditions.append(Job.search_date >= date_from)
    if date_to:
        conditions.append(Job.search_date <= date_to)

    if statuses:
        conditions.append(Job.ID.in_(_latest_status_job_ids_subquery(statuses)))

    if hide_applied:
        conditions.append(Job.ID.not_in(_applied_job_ids_subquery()))

    return select(Job).where(and_(*conditions)) if conditions else select(Job)


async def _fetch_job_page(
    db: AsyncSession,
    *,
    quick: str,
    score_min: int,
    company: str | None,
    statuses: list[str],
    hide_applied: bool,
    date_from: str | None,
    date_to: str | None,
    cursor: str | None,
) -> tuple[list[Job], str | None, int]:
    """
    Runs the filtered query for one page, using keyset pagination (not
    OFFSET) on (fit_score DESC, ID DESC) — matches the composite index
    declared on Job, and avoids OFFSET's usual problems: it stays fast on
    later pages instead of degrading as the offset grows, and it can't
    skip/duplicate rows if new jobs are inserted between "Load More"
    clicks the way an OFFSET-based page number would.

    Returns (jobs_for_this_page, next_cursor_or_None, total_matching_count).
    total_matching_count ignores the cursor (it's the count for the whole
    filtered set, not just what's left) — cheap enough to compute on every
    call given the composite index, and simpler than trying to cache it
    client-side across requests.
    """
    base_query = _build_job_query(quick, score_min, company, statuses, hide_applied, date_from, date_to)

    total_result = await db.execute(select(func.count()).select_from(base_query.subquery()))
    total_count = total_result.scalar() or 0

    parsed_cursor = _parse_cursor(cursor)
    paged_query = base_query
    if parsed_cursor:
        cursor_score, cursor_id = parsed_cursor
        paged_query = paged_query.where(
            or_(
                Job.fit_score < cursor_score,
                and_(Job.fit_score == cursor_score, Job.ID < cursor_id),
            )
        )

    # Fetch one extra row beyond the page size — a cheap way to know
    # whether there's a next page without a second query.
    paged_query = paged_query.order_by(desc(Job.fit_score), desc(Job.ID)).limit(PAGE_SIZE + 1)
    result = await db.execute(paged_query)
    rows = result.scalars().all()

    has_more = len(rows) > PAGE_SIZE
    page_rows = rows[:PAGE_SIZE]

    next_cursor = None
    if has_more and page_rows:
        last = page_rows[-1]
        next_cursor = f"{last.fit_score}_{last.ID}"

    return page_rows, next_cursor, total_count


def _filters_from_query_params(
    quick: str | None,
    score_min: int | None,
    company: str | None,
    status: list[str] | None,
    hide_applied: bool | None,
    date_from: str | None,
    date_to: str | None,
) -> dict:
    """Applies defaults for any filter param the caller omitted — used so
    GET /jobs/rows behaves sensibly even if a caller doesn't pass every
    field (e.g. a bookmarked or hand-built URL)."""
    return {
        "quick": quick or DEFAULT_QUICK_FILTER,
        "score_min": score_min if score_min is not None else DEFAULT_SCORE_MIN,
        "company": company,
        "statuses": status or [],
        "hide_applied": hide_applied if hide_applied is not None else DEFAULT_HIDE_APPLIED,
        "date_from": date_from,
        "date_to": date_to,
    }


@router.get("", response_class=HTMLResponse)
async def list_jobs_ui(request: Request, db: AsyncSession = Depends(get_db)):
    today = datetime.date.today()
    default_date_from = (today - datetime.timedelta(days=DEFAULT_LOOKBACK_DAYS)).isoformat()
    default_date_to = today.isoformat()

    jobs, next_cursor, total_count = await _fetch_job_page(
        db,
        quick=DEFAULT_QUICK_FILTER,
        score_min=DEFAULT_SCORE_MIN,
        company=None,
        statuses=[],
        hide_applied=DEFAULT_HIDE_APPLIED,
        date_from=default_date_from,
        date_to=default_date_to,
        cursor=None,
    )

    return templates.TemplateResponse("jobs.html", {
        "request": request,
        "jobs": jobs,
        "active_module": "jobs",
        "next_cursor": next_cursor,
        "loaded_count": len(jobs),
        "total_count": total_count,
        "default_date_from": default_date_from,
        "default_date_to": default_date_to,
    })


@router.get("/rows", response_class=HTMLResponse)
async def get_job_rows(
    request: Request,
    db: AsyncSession = Depends(get_db),
    quick: str | None = None,
    score_min: int | None = None,
    company: str | None = None,
    status: list[str] | None = None,
    hide_applied: bool | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    cursor: str | None = None,
):
    """
    Returns one page of job rows as an HTML fragment (two <template>
    blocks — desktop <tr>s and mobile cards — plus pagination metadata),
    for the front end to either REPLACE the current list with (filters
    changed, cursor absent) or APPEND to it (Load More clicked, cursor
    present). Which of those two the front end does is purely a client-
    side decision — this endpoint always does the same thing: "give me the
    next PAGE_SIZE rows matching these filters, after this cursor."
    """
    filters = _filters_from_query_params(quick, score_min, company, status, hide_applied, date_from, date_to)
    jobs, next_cursor, total_count = await _fetch_job_page(db, cursor=cursor, **filters)

    return templates.TemplateResponse("partials/jobs_rows.html", {
        "request": request,
        "jobs": jobs,
        "next_cursor": next_cursor,
        "loaded_count": len(jobs),
        "total_count": total_count,
    })


@router.get("/detail/{job_id}")
async def get_job_detail(job_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.ID == job_id))  # capital ID
    job = result.scalar_one_or_none()
    if not job:
        return html_error(request, f"Job {job_id} not found.", status_code=404)
    return templates.TemplateResponse(
        "partials/job_detail.html",
        {"request": request, "job": job}
    )
