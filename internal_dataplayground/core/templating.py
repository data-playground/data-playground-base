from jinja2 import ChoiceLoader, FileSystemLoader
from fastapi.templating import Jinja2Templates

# Shared Jinja2Templates instance for the whole app.
#
# Uses a ChoiceLoader so that template lookups first check the existing
# root `templates/` directory (unchanged for every domain not yet migrated),
# then fall back to `domains/habits/templates/` (the first domain-folder
# pilot). This lets router code keep calling
# `templates.TemplateResponse("habits.html", ...)` — the same bare filename
# as before the move — without needing to know which directory it now
# physically lives in.
#
# As more domains migrate, add their template directories to this loader.
templates = Jinja2Templates(directory="templates")
templates.env.loader = ChoiceLoader([
    FileSystemLoader("templates"),
    FileSystemLoader("domains/habits/templates"),
    FileSystemLoader("domains/blog/templates"),
    FileSystemLoader("domains/code_intel/templates"),
])
