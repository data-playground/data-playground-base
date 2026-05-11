# Contributing to Life OS

## Architectural Rules

### DAG / FastAPI Boundary (Non-Negotiable)

DAG files in `airflow/dags/` must **never** import from:
- `models.py`
- `database.py`
- Any file in `routers/`
- Any file in `services/` (FastAPI services)

**All database access from Airflow DAGs uses `dag_db.py` helpers only:**

```python
# CORRECT
from dag_db import fetch_one, fetch_all, execute

idea = fetch_one("SELECT * FROM blog_ideas WHERE id = %s", (idea_id,))
execute("UPDATE blog_ideas SET status = %s WHERE id = %s", ("ready_to_publish", idea_id))

# WRONG — never do this in a DAG
from models import BlogIdea, BlogIdeaStatus
from dag_db import get_sync_session
session = get_sync_session()
idea = session.get(BlogIdea, idea_id)
```

**Why:** SQLAlchemy models carry FastAPI-specific dependencies. Importing them
into Airflow creates a fragile coupling — a change to `models.py` can silently
break DAG imports and cause Airflow to show parse errors with no clear cause.

**Enum values:** Use the raw string value, not the Python enum:
```python
# CORRECT
execute("UPDATE blog_ideas SET status = %s ...", ("ready_to_publish",))

# WRONG
from models import BlogIdeaStatus
execute("UPDATE blog_ideas SET status = %s ...", (BlogIdeaStatus.READY_TO_PUBLISH.value,))
```

### Other Rules

- Every new page template must extend `base.html`
- Every new module gets its own router file — no adding endpoints to existing routers
- Every schema change requires an Alembic migration
- No router file exceeds 300 lines — split before that threshold
- All credentials come from GCP Secret Manager via `get_key()`
- HTMX partial responses go in `templates/partials/` — never return full pages from HTMX endpoints
- HTML error responses use `html_error()` from `routers/_helpers.py`
- JSON error responses use `raise HTTPException()`
