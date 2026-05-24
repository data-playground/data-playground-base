# routers/media_settings.py
"""
Media Settings — streaming service subscription management.

Prefix: /media/settings
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import StreamingService

router = APIRouter(prefix="/media/settings", tags=["Media Settings"])
templates = Jinja2Templates(directory="templates")


@router.get("", response_class=HTMLResponse)
async def settings_page(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(StreamingService).order_by(StreamingService.sort_order)
    )
    services = result.scalars().all()
    return templates.TemplateResponse("media_settings.html", {
        "request": request,
        "active_module": "media",
        "services": services,
    })


@router.post("/subscriptions", response_class=HTMLResponse)
async def update_subscriptions(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Bulk-updates subscription status for all streaming services.
    Accepts a form with checkbox fields named service_{id}.
    Unchecked boxes are not submitted — so any service not in the form = unsubscribed.
    """
    form = await request.form()
    subscribed_ids = {
        int(k.replace("service_", ""))
        for k in form.keys()
        if k.startswith("service_")
    }

    result = await db.execute(select(StreamingService))
    services = result.scalars().all()

    for svc in services:
        svc.is_subscribed = svc.id in subscribed_ids

    await db.commit()

    # Return the updated service list partial
    result = await db.execute(
        select(StreamingService).order_by(StreamingService.sort_order)
    )
    services = result.scalars().all()

    return templates.TemplateResponse("partials/streaming_service_list.html", {
        "request": request,
        "services": services,
    })
