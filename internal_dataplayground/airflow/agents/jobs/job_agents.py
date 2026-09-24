# airflow/agents/job_agents.py
"""
Job Scout agent functions — LinkedIn search scraping + AI fit scoring.

Provider: Gemini 2.5 Flash-Lite (JSON mode)
Frequency: Once per scheduled run of life_os_job_scout (see DAG for schedule)

Model routing note: flash-lite is used here specifically because this agent
processes large day-to-day batches of job descriptions (up to ~15 batches of
10 jobs each per run). Flash-lite is cheaper and faster than the 2.5 Flash
used elsewhere (README Writer, Editor) where per-call quality matters more
than throughput — this mirrors the routing rationale documented at the top
of blog_agents.py.

ARCHITECTURAL NOTE: this file is imported by DAG tasks only. It has no
dependency on models.py, database.py, or any FastAPI router/service, in
line with the DAG/FastAPI boundary rule in CONTRIBUTING.md.
"""
import json
import logging
import re
import urllib.parse
from collections import defaultdict
from itertools import zip_longest

import requests
from bs4 import BeautifulSoup

from services.ai import call_gemini_json, MODEL_FLASH_LITE

log = logging.getLogger(__name__)

# ── SCRAPER CONFIG ────────────────────────────────────────────────────────────

_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    ),
}

DEFAULT_LOCATION = "New York City Metropolitan Area"
DEFAULT_GEO_ID = "90000070"
DEFAULT_JOB_TYPE = "F"          # Full-time
DEFAULT_EXPERIENCE_LEVEL = "5"  # Mid-Senior level

# Polite delay between job-detail page fetches. LinkedIn's public pages will
# start serving stripped-down / blocked responses if hit too fast from one IP.
# This is the first thing to increase if descriptions start coming back empty.
DETAIL_FETCH_DELAY_SEC = 1.5


# ── LINKEDIN SEARCH SCRAPE ────────────────────────────────────────────────────

def search_linkedin_jobs(
    keywords: str,
    location: str = DEFAULT_LOCATION,
    geo_id: str = DEFAULT_GEO_ID,
    job_type: str = DEFAULT_JOB_TYPE,
    experience_level: str = DEFAULT_EXPERIENCE_LEVEL,
) -> list[dict]:
    """
    Scrapes LinkedIn's public (unauthenticated) job search results page for
    a single query. No login required, but fragile — LinkedIn can change
    class names or start blocking the source IP without notice. If every
    search suddenly returns zero cards, check the CSS selectors below first.
    """
    url = (
        f"https://www.linkedin.com/jobs/search?keywords={urllib.parse.quote(keywords)}"
        f"&location={urllib.parse.quote(location)}&geoId={geo_id}"
        f"&f_SB2={experience_level}&f_TPR=&f_JT={job_type}&position=1&pageNum=0"
    )
    resp = requests.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    jobs = []
    for card in soup.select("div.base-card"):
        entity_urn = card.get("data-entity-urn")
        title_el = card.find("h3", class_="base-search-card__title")
        link_el = card.find("a", class_="base-card__full-link")
        if not (entity_urn and title_el and link_el):
            continue

        company_el = card.find("h4", class_="base-search-card__subtitle")
        location_el = card.find("span", class_="job-search-card__location")
        date_el = card.find("time", class_="job-search-card__listdate")

        jobs.append({
            "job_id":       entity_urn.split(":")[-1],
            "job_title":    title_el.text.strip(),
            "company_name": company_el.text.strip() if company_el else None,
            "location":     location_el.text.strip() if location_el else None,
            "post_date":    date_el.get("datetime", "").strip() if date_el else None,
            "job_link":     link_el.get("href", "").strip(),
            "job_search":   keywords,
        })
    return jobs


def get_job_details(job_link: str) -> tuple[str | None, str | None]:
    """
    Fetches the full description and salary (if listed) from a job's detail
    page. Returns (description, salary) — either may be None if the page
    didn't load the expected markup (removed posting, layout change, etc).
    """
    try:
        resp = requests.get(job_link, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("Failed to fetch job detail page %s: %s", job_link, exc)
        return None, None

    soup = BeautifulSoup(resp.content, "html.parser")

    description = None
    desc_el = soup.find("div", class_="show-more-less-html__markup")
    if desc_el:
        description = desc_el.get_text(separator="\n", strip=True)

    salary = None
    salary_el = soup.find("div", class_="salary compensation__salary")
    if salary_el:
        salary = salary_el.text.strip()
    else:
        snippet_el = soup.find("span", class_="salary-snippet")
        if snippet_el:
            salary = snippet_el.text.strip()

    return description, salary


def get_full_job_posting(job_link: str) -> dict:
    """
    Fetches a LinkedIn job's detail page and extracts everything needed to
    promote a StagingJob straight into linkedin_jobs — title, company,
    location, description, and salary — in a single page fetch.
    """
    result: dict = {
        "job_title": None, "company_name": None, "location": None,
        "description": None, "salary": None,
    }
    try:
        resp = requests.get(job_link, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("Failed to fetch job posting page %s: %s", job_link, exc)
        return result

    soup = BeautifulSoup(resp.content, "html.parser")

    title_el = soup.find("h1", class_="top-card-layout__title") or soup.find("h1")
    if title_el:
        result["job_title"] = title_el.get_text(strip=True)

    company_el = (
        soup.find("a", class_="topcard__org-name-link")
        or soup.find("span", class_="topcard__flavor")
    )
    if company_el:
        result["company_name"] = company_el.get_text(strip=True)

    location_el = soup.find("span", class_="topcard__flavor--bullet")
    if location_el:
        result["location"] = location_el.get_text(strip=True)

    desc_el = soup.find("div", class_="show-more-less-html__markup")
    if desc_el:
        result["description"] = desc_el.get_text(separator="\n", strip=True)

    salary_el = soup.find("div", class_="salary compensation__salary")
    if salary_el:
        result["salary"] = salary_el.text.strip()
    else:
        snippet_el = soup.find("span", class_="salary-snippet")
        if snippet_el:
            result["salary"] = snippet_el.text.strip()

    return result


def deduplicate_jobs(raw_jobs: list[dict]) -> list[dict]:
    """Dedupes by job_id, keeping the first occurrence (order-preserving)."""
    seen_ids = set()
    unique = []
    for job in raw_jobs:
        if job["job_id"] not in seen_ids:
            unique.append(job)
            seen_ids.add(job["job_id"])
    return unique


def clean_date(value: str | None) -> str | None:
    """Truncates an ISO-ish datetime string down to YYYY-MM-DD for DATE columns."""
    if not value:
        return None
    return value[:10]


_JOB_ID_FROM_URL_RE = re.compile(r"/jobs/view/(\d+)")


def extract_linkedin_job_id(job_link: str) -> str | None:
    """
    Pulls the numeric LinkedIn job id out of a job posting URL.
    """
    match = _JOB_ID_FROM_URL_RE.search(job_link)
    return match.group(1) if match else None


# ── BATCH CHUNKING ────────────────────────────────────────────────────────────

def build_scoring_chunks(
    jobs: list[dict],
    chunk_size: int = 10,
    max_chunks: int = 15,
) -> list[list[dict]]:
    by_search: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        by_search[job["job_search"]].append(job)

    ordered = sorted(by_search.items(), key=lambda kv: len(kv[1]), reverse=True)

    chunks: list[list[dict]] = []
    reservoir: list[dict] = []
    small_cat_started = False

    for _key, cat_jobs in ordered:
        if len(chunks) >= max_chunks:
            break

        if len(cat_jobs) >= chunk_size:
            chunks.append(cat_jobs[:chunk_size])
            reservoir.extend(cat_jobs[chunk_size:])
        else:
            if not small_cat_started:
                small_cat_started = True
                groups: dict[str, list[dict]] = defaultdict(list)
                for item in reservoir:
                    groups[item["job_search"]].append(item)
                interleaved = zip_longest(*groups.values())
                reservoir = [item for group in interleaved for item in group if item is not None]

            chunk = list(cat_jobs)
            needed = chunk_size - len(chunk)
            chunk.extend(reservoir[:needed])
            chunks.append(chunk)
            reservoir = reservoir[needed:]

    while reservoir and len(chunks) < max_chunks:
        chunks.append(reservoir[:chunk_size])
        reservoir = reservoir[chunk_size:]

    return chunks


# ── GEMINI FIT SCORER ─────────────────────────────────────────────────────────

_JOB_SCORE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "ID":                     {"type": "STRING"},
            "remote":                 {"type": "BOOLEAN"},
            "explanation":            {"type": "STRING"},
            "fit_score":              {"type": "INTEGER"},
            "qualification_analysis": {"type": "STRING"},
            "skill_gaps":             {"type": "STRING"},
        },
        "required": [
            "ID", "remote", "explanation", "fit_score",
            "qualification_analysis", "skill_gaps",
        ],
    },
}

_SYSTEM_INSTRUCTIONS_TEMPLATE = """
### ROLE
You are an expert Career Agent and Recruitment Specialist. Your task is to
analyze a batch of job descriptions and determine how well each matches the
provided resume.

### CANDIDATE RESUME DATA
```markdown
{resume_markdown}
```

### KEY STRENGTHS TO WEIGHT HEAVILY
{key_strengths}

### TASK
1. Analyze each job description in depth — technical requirements and
   business responsibilities.
2. Verify the technical stack and seniority level against the resume.
3. Assign a fit_score out of 100: technical skill overlap (60%),
   seniority/level alignment (20%), industry/domain experience (20%).
4. Map specific resume achievements directly to the JD's requirements or
   preferred qualifications.
5. Explicitly list any tools, frameworks, or certifications in the JD that
   are missing or weak in the resume (skill_gaps).

### CONSTRAINTS
- Return ONLY a JSON array of objects, one per job, each including the
  job's original "ID" field so results can be matched back to the source job.
- Do not include any conversational text or markdown outside the JSON.
- If a job is completely irrelevant, still return it with a low score (<30)
  rather than omitting it — omitted jobs are treated as "failed to score"
  and dropped from the load step.
"""


def score_job_batch(
    jobs_chunk: list[dict],
    resume_markdown: str,
    key_strengths: str,
) -> list[dict]:
    """
    Sends one batch of jobs to Gemini for fit scoring.
    """
    system = _SYSTEM_INSTRUCTIONS_TEMPLATE.format(
        resume_markdown=resume_markdown,
        key_strengths=key_strengths,
    )
    batch_content = "\n\n\n".join(
        f"ID: {j.get('job_id') or j.get('external_ref')}\n"
        f"Title: {j['job_title']}\nDesc: {j.get('description') or 'Not available'}\n---"
        for j in jobs_chunk
    )
    raw = call_gemini_json(
        batch_content, schema=_JOB_SCORE_SCHEMA, system=system, model=MODEL_FLASH_LITE,
    )
    return json.loads(raw)
