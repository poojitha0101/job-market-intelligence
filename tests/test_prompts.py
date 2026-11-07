from job_assistant.prompts import (
    GENERAL_MARK,
    build_prompt,
    build_system_instruction,
    split_marker,
)


def test_split_marker_detects_and_strips_marker():
    assert split_marker("[[general]] SQL is a query language.") == (True, "SQL is a query language.")


def test_split_marker_hides_partial_marker_while_streaming():
    assert split_marker("[[gen") == (False, "")
    assert split_marker("") == (False, "")


def test_split_marker_passes_normal_text_through():
    assert split_marker("Hello there") == (False, "Hello there")


def test_split_marker_marker_only():
    assert split_marker(GENERAL_MARK) == (True, "")


def test_system_instruction_uses_real_listing_count():
    text = build_system_instruction(784)
    assert "784" in text
    assert "{n_listings}" not in text
    assert "[[general]]" in text  # the prompt and GENERAL_MARK must agree


def test_system_instruction_follows_corpus_size():
    assert "12 job listings" in build_system_instruction(12)


def test_build_prompt_without_history_or_listings():
    prompt = build_prompt("hi", [], "")
    assert "(no earlier messages)" in prompt
    assert "(none for this message)" in prompt
    assert prompt.endswith("USER MESSAGE: hi\n")


def test_build_prompt_keeps_only_recent_history():
    history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    prompt = build_prompt("now", history, "[Job 1] x")
    assert "q19" in prompt
    assert "q0" not in prompt
    assert "[Job 1] x" in prompt
