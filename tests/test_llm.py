import pytest

from job_assistant.llm import (
    INTERRUPTED_NOTE,
    NO_RESPONSE_NOTE,
    friendly_error,
    is_key_error,
    stream_answer,
)

MODELS = ["m1", "m2"]


def run(client, **kwargs):
    kwargs.setdefault("models", MODELS)
    return list(stream_answer(client, "prompt", "system", pause=0, config=object(), **kwargs))


def test_streams_from_first_model(fake_client):
    client = fake_client({"m1": [["Hel", "lo"]], "m2": []})
    assert "".join(run(client)) == "Hello"
    assert client.models.calls == ["m1"]


def test_retries_same_model_then_falls_back(fake_client):
    client = fake_client(
        {"m1": [RuntimeError("503 UNAVAILABLE"), RuntimeError("503 UNAVAILABLE")], "m2": [["ok"]]}
    )
    assert "".join(run(client)) == "ok"
    assert client.models.calls == ["m1", "m1", "m2"]


def test_key_error_is_raised_immediately(fake_client):
    client = fake_client({"m1": [RuntimeError("API key not valid")], "m2": [["never"]]})
    with pytest.raises(RuntimeError, match="API key not valid"):
        run(client)
    assert client.models.calls == ["m1"]


def test_last_error_is_raised_when_everything_fails(fake_client):
    client = fake_client(
        {
            "m1": [RuntimeError("busy"), RuntimeError("busy")],
            "m2": [RuntimeError("busy"), RuntimeError("429 RESOURCE_EXHAUSTED")],
        }
    )
    with pytest.raises(RuntimeError) as excinfo:
        run(client)
    assert "quota" in friendly_error(excinfo.value)


def test_stream_that_breaks_midway_gets_a_note_and_no_retry(fake_client):
    client = fake_client({"m1": [(["partial"], RuntimeError("connection reset"))], "m2": [["x"]]})
    out = run(client)
    assert out == ["partial", INTERRUPTED_NOTE]
    assert client.models.calls == ["m1"]


def test_empty_streams_end_with_a_plain_message(fake_client):
    client = fake_client({"m1": [[], []], "m2": [[], []]})
    assert run(client) == [NO_RESPONSE_NOTE]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("400 API_KEY_INVALID", "key was not accepted"),
        ("PERMISSION_DENIED", "key was not accepted"),
        ("429 Too Many Requests", "quota"),
        ("503 UNAVAILABLE", "busy"),
        ("Request timed out", "busy"),
    ],
)
def test_friendly_error_mapping(text, expected):
    assert expected in friendly_error(RuntimeError(text))


def test_friendly_error_truncates_unknown_details():
    message = friendly_error(RuntimeError("x" * 500))
    assert message.startswith("Something went wrong")
    assert len(message) < 200


def test_is_key_error():
    assert is_key_error(RuntimeError("UNAUTHENTICATED"))
    assert not is_key_error(RuntimeError("503"))
