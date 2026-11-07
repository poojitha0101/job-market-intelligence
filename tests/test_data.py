"""Checks on the shipped data files. They catch a CSV/embedding mismatch before deploy."""

import numpy as np
import pytest

from job_assistant import config
from job_assistant.retrieval import load_corpus


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_row_counts_match(corpus):
    jobs, vecs = corpus
    assert len(jobs) == len(vecs) > 0


def test_embedding_dimensions(corpus):
    _, vecs = corpus
    assert vecs.shape[1] == config.EMBED_DIMENSIONS


def test_embeddings_are_finite_and_unit_length(corpus):
    _, vecs = corpus
    assert np.all(np.isfinite(vecs))
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)


def test_every_listing_has_title_and_description(corpus):
    jobs, _ = corpus
    assert (jobs["title"].str.strip() != "").all()
    assert (jobs["description"].str.strip() != "").all()


def test_stylesheet_exists():
    assert config.STYLE_CSS.read_text(encoding="utf-8").lstrip().startswith("@import")
