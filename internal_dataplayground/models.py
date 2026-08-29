import datetime
import enum
import math
from decimal import Decimal
from typing import Optional

# The shared Base class for all tables now lives in core/base_model.py.
# Re-exported here (temporary — see domains/habits pilot roadmap) so every
# other file still doing `from models import Base` keeps working unchanged.
from core.base_model import Base
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

# ── WO#20 shim-removal pass ───────────────────────────────────────────────────
# Every domain's re-export shim (Jobs, Finance, Blog, Code Intel, Habits,
# Journal, Recipes, Workout, Media, Planning) has been removed from this
# file. Each was confirmed to have no consumer left other than
# routers/dashboard.py (now repointed directly at each domain's own
# models.py — see routers/dashboard.py), or, for Code Intel / Recipes /
# Workout / Media / Planning, no consumer at all besides other domains'
# routers, which already import those domains' models.py directly.
#
# NOTE: the imports above (datetime, enum, math, Decimal, Optional, the
# sqlalchemy column/type imports, Mapped/mapped_column/relationship,
# BaseModel) are now unused — they only existed to support the class
# definitions that were relocated out of this file during WO#1–10. Left
# in place deliberately: WO#20's own HARD BOUNDARIES limit this pass to
# "removing a shim and updating exactly one import line in dashboard.py
# per domain," not general dead-code cleanup. See this work order's Notes
# for the recommended follow-up (WO#10 postmortem §4.2, Option 2 — reduce
# this file to a clean 10-line import-registry, or delete it and move the
# import-triggers-registration guarantee into database.py's init_db()).
