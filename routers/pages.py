from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from config import templates

router = APIRouter()

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

# NOTE: /join/{room_id} is intentionally NOT defined here — that route
# already lives in telehealth.py (patient-facing join page, see its
# page_join docstring) and telehealth_router is included in app.py before
# this one. Defining it again here would just be dead/shadowed code.


@router.get("/", response_class=HTMLResponse)
async def page_index(request: Request):
    return RedirectResponse(url="/analytics")


@router.get("/patients", response_class=HTMLResponse)
async def page_patients(request: Request):
    return templates.TemplateResponse(request, "patients.html", headers=NO_CACHE_HEADERS)


@router.get("/session", response_class=HTMLResponse)
async def page_session(request: Request):
    return templates.TemplateResponse(request, "session.html", headers=NO_CACHE_HEADERS)


@router.get("/reports", response_class=HTMLResponse)
async def page_reports(request: Request):
    return templates.TemplateResponse(request, "reports.html", headers=NO_CACHE_HEADERS)


@router.get("/analytics", response_class=HTMLResponse)
async def page_analytics(request: Request):
    return templates.TemplateResponse(request, "analytics.html", headers=NO_CACHE_HEADERS)
