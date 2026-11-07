"""Paths and tunable settings shared by the app, retrieval and LLM layers."""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent

DATA_DIR = REPO_ROOT / "data"
JOBS_CSV = DATA_DIR / "app_jobs.csv"
EMBEDDINGS_NPY = DATA_DIR / "job_embeddings.npy"
STYLE_CSS = APP_DIR / "static" / "style.css"

# Columns the app needs in the jobs CSV. Optional columns (search_role,
# redirect_url) are used when present and skipped when not.
REQUIRED_COLUMNS = ("title", "company.display_name", "location.display_name", "description")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 768

_DEFAULT_GEN_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
# Comma-separated override, e.g. GEMINI_MODELS="model-a,model-b". Tried in order.
GEN_MODELS = [
    name.strip()
    for name in os.environ.get("GEMINI_MODELS", "").split(",")
    if name.strip()
] or _DEFAULT_GEN_MODELS

TOP_K = 8  # listings sent to the model per question
HISTORY_MESSAGES = 6  # earlier chat messages included in the prompt
MAX_QUESTION_CHARS = 1000
REQUEST_TIMEOUT_MS = 30000
ATTEMPTS_PER_MODEL = 2
RETRY_PAUSE_SECONDS = 2.0
SHORT_FOLLOWUP_WORDS = 6  # questions this short get the previous question prepended


def env_api_key() -> str:
    """Return GEMINI_API_KEY from the environment, or an empty string.

    Meant for local runs. Do not set it on a public deployment, or every
    visitor will spend that key's quota.
    """
    return os.environ.get("GEMINI_API_KEY", "").strip()
