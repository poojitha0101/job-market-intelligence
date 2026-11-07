import numpy as np
import pandas as pd
import pytest

from job_assistant.retrieval import (
    build_search_query,
    is_small_talk,
    load_corpus,
    normalize,
    search_jobs,
)


def test_normalize_gives_unit_rows_and_survives_zero_rows():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype="float32"))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.all(np.isfinite(out))


def test_search_jobs_returns_closest_first(jobs, doc_vecs):
    hits = search_jobs(jobs, doc_vecs, np.array([0.1, 0.0, 0.9]), top_k=2)
    assert list(hits["title"]) == ["BI Developer", "Data Analyst"]


def test_search_jobs_ignores_query_scale(jobs, doc_vecs):
    small = search_jobs(jobs, doc_vecs, np.array([0.0, 1.0, 0.0]), top_k=1)
    large = search_jobs(jobs, doc_vecs, np.array([0.0, 50.0, 0.0]), top_k=1)
    assert list(small["title"]) == list(large["title"]) == ["ML Engineer"]


def test_search_jobs_top_k_larger_than_corpus(jobs, doc_vecs):
    assert len(search_jobs(jobs, doc_vecs, np.array([1.0, 0.0, 0.0]), top_k=50)) == 3


@pytest.mark.parametrize(
    "text",
    ["hi", "Hello!", "thanks", "Thank you so much", "good morning", "bye", "who are you?", "నమస్తే", "help"],
)
def test_small_talk_detected(text):
    assert is_small_talk(text)


@pytest.mark.parametrize(
    "text",
    ["which jobs need SQL", "hi, which jobs need SQL?", "explain what RAG means", "help me find python jobs"],
)
def test_real_questions_are_not_small_talk(text):
    assert not is_small_talk(text)


def test_short_followup_borrows_previous_question():
    history = [
        {"role": "user", "content": "Which jobs need Python?"},
        {"role": "assistant", "content": "Several."},
    ]
    assert build_search_query("and in Hyderabad?", history) == "Which jobs need Python? and in Hyderabad?"


def test_long_question_is_left_alone():
    history = [{"role": "user", "content": "earlier"}]
    question = "Which companies are hiring data analysts with Power BI experience"
    assert build_search_query(question, history) == question


def test_short_question_without_history_is_left_alone():
    assert build_search_query("python jobs?", []) == "python jobs?"


def _write_corpus(tmp_path, n_rows, n_vectors, columns=None):
    columns = columns or ["title", "company.display_name", "location.display_name", "description"]
    csv = tmp_path / "jobs.csv"
    npy = tmp_path / "vecs.npy"
    pd.DataFrame({c: ["x"] * n_rows for c in columns}).to_csv(csv, index=False)
    np.save(npy, np.random.default_rng(0).normal(size=(n_vectors, 4)).astype("float32"))
    return csv, npy


def test_load_corpus_ok_and_normalised(tmp_path):
    csv, npy = _write_corpus(tmp_path, 5, 5)
    jobs, vecs = load_corpus(csv, npy)
    assert len(jobs) == 5
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_load_corpus_rejects_row_mismatch(tmp_path):
    csv, npy = _write_corpus(tmp_path, 5, 4)
    with pytest.raises(ValueError, match="5 listings but 4 embeddings"):
        load_corpus(csv, npy)


def test_load_corpus_rejects_missing_column(tmp_path):
    csv, npy = _write_corpus(tmp_path, 3, 3, columns=["title", "description"])
    with pytest.raises(ValueError, match="missing columns"):
        load_corpus(csv, npy)
