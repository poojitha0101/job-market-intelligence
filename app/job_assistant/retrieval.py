"""Corpus loading and nearest-neighbour search over precomputed embeddings.

Search is a brute-force cosine similarity in NumPy. The corpus is a few
hundred rows, so an index or vector database would add moving parts without
making queries any faster in practice.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EMBEDDINGS_NPY,
    JOBS_CSV,
    REQUIRED_COLUMNS,
    SHORT_FOLLOWUP_WORDS,
    TOP_K,
)

SMALL_TALK = re.compile(
    r"^\s*(hi+|hello+|hey+|hola|namaste|namaskaram|నమస్తే|నమస్కారం|హాయ్|హలో|"
    r"good (morning|afternoon|evening|night)|how are you.*|how r u.*|what'?s up|sup|"
    r"thanks?( you)?.*|thank u.*|ధన్యవాదాలు.*|ok(ay)?|cool|nice|great|bye.*|goodbye.*|"
    r"who are you.*|what can you do.*|help)\s*[!?.]*\s*$",
    re.IGNORECASE,
)


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def load_corpus(
    jobs_path: Path = JOBS_CSV, vectors_path: Path = EMBEDDINGS_NPY
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load listings and their embeddings, checking that they line up.

    Raises ValueError if a required column is missing or the row counts differ,
    because a silent mismatch would return the wrong listings for every query.
    """
    jobs = pd.read_csv(jobs_path).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in jobs.columns]
    if missing:
        raise ValueError(f"{jobs_path} is missing columns: {', '.join(missing)}")
    vectors = np.load(vectors_path)
    if len(jobs) != len(vectors):
        raise ValueError(
            f"{len(jobs)} listings but {len(vectors)} embeddings; regenerate the embeddings."
        )
    return jobs, normalize(vectors.astype("float32"))


def is_small_talk(question: str) -> bool:
    """True for greetings and thanks, which do not need a listing search."""
    return bool(SMALL_TALK.match(question))


def build_search_query(question: str, history: list[dict]) -> str:
    """Add the previous user question to very short follow-ups.

    A message like "and in Hyderabad?" means nothing to an embedding model
    on its own, so it borrows context from the last question.
    """
    if len(question.split()) <= SHORT_FOLLOWUP_WORDS:
        earlier = [m["content"] for m in history if m["role"] == "user"]
        if earlier:
            return earlier[-1] + " " + question
    return question


def search_jobs(
    jobs: pd.DataFrame, doc_vecs: np.ndarray, query_vec: np.ndarray, top_k: int = TOP_K
) -> pd.DataFrame:
    """Return the top_k listings closest to the query vector, best first."""
    query = normalize(np.asarray(query_vec, dtype="float32").reshape(1, -1))[0]
    scores = doc_vecs @ query
    top = np.argsort(-scores)[:top_k]
    return jobs.iloc[top]
