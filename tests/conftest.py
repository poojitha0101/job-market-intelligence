"""Shared fixtures: a tiny in-memory corpus and a fake Gemini client."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def jobs():
    return pd.DataFrame(
        {
            "title": ["Data Analyst", "ML Engineer", "BI Developer"],
            "company.display_name": ["Acme", "Globex", ""],
            "location.display_name": ["Hyderabad, Telangana", "Bangalore, Karnataka", ""],
            "description": ["SQL and Excel", "Python and PyTorch", "Power BI and DAX"],
        }
    )


@pytest.fixture
def doc_vecs():
    # Three orthogonal unit vectors, so each listing has one obvious query.
    return np.eye(3, dtype="float32")


class FakeChunk:
    def __init__(self, text):
        self.text = text


class FakeModels:
    """Scripted stand-in for client.models.

    `script` maps a model name to a list of steps. Each step is either a list
    of text pieces (a stream that succeeds) or an Exception (raised when the
    stream is opened). A step is a tuple (pieces, error) to emit pieces and
    then raise, which models a stream that breaks part-way.
    """

    def __init__(self, script):
        self.script = {name: list(steps) for name, steps in script.items()}
        self.calls = []

    def generate_content_stream(self, model, contents, config):
        self.calls.append(model)
        step = self.script[model].pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, tuple):
            pieces, error = step

            def broken():
                for p in pieces:
                    yield FakeChunk(p)
                raise error

            return broken()
        return iter(FakeChunk(p) for p in step)


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


@pytest.fixture
def fake_client():
    return FakeClient
