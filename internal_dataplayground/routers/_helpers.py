# routers/_helpers.py
"""
Shared router utilities.

html_error() — returns a rendered error_fragment.html partial for endpoints
that return HTML (HTMX targets). Use this instead of constructing inline
HTMLResponse error strings.

For endpoints that return JSON, raise HTTPException directly — that is correct
and intentional, and does not need this helper.
"""

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

_templates = Jinja2Templates(directory="templates")


def html_error(
    request: Request,
    message: str,
    status_code: int = 400,
    retry_url: str | None = None,
) -> HTMLResponse:
    """
    Renders the error_fragment.html partial and returns it as an HTMLResponse.

    Args:
        request:     The current FastAPI request (required by Jinja2).
        message:     Human-readable error description shown to the user.
        status_code: HTTP status code for the response (default 400).
        retry_url:   Optional URL rendered as a "Try again →" link.

    Returns:
        HTMLResponse containing the rendered error fragment.
    """
    return _templates.TemplateResponse(
        "partials/error_fragment.html",
        {"request": request, "message": message, "retry_url": retry_url},
        status_code=status_code,
    )
