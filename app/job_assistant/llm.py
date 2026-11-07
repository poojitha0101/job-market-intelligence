"""Gemini calls: query embedding, streamed answers with model fallback, error text.

google-genai is imported inside the functions that need it so the rest of the
package, and the tests, work without the SDK installed.
"""

import time
from typing import Any, Iterator

import numpy as np

from .config import (
    ATTEMPTS_PER_MODEL,
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    GEN_MODELS,
    REQUEST_TIMEOUT_MS,
    RETRY_PAUSE_SECONDS,
)

INTERRUPTED_NOTE = "\n\n(The answer was interrupted. Please ask again.)"
NO_RESPONSE_NOTE = "The model did not respond. Wait a minute and ask again."


def make_client(api_key: str):
    """Create a Gemini client with a request timeout."""
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


def embed_query(client, query: str) -> np.ndarray:
    """Embed a search query with the same model and size as the stored listings."""
    from google.genai import types

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIMENSIONS
        ),
    )
    return np.array(result.embeddings[0].values, dtype="float32")


def is_key_error(error: Exception) -> bool:
    """True when the error means the API key was rejected, so retrying is pointless."""
    text = str(error)
    return any(
        s in text
        for s in ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "UNAUTHENTICATED")
    )


def friendly_error(error: Exception) -> str:
    """Map an API exception to a short message a visitor can act on."""
    text = str(error)
    if is_key_error(error):
        return "That API key was not accepted. Check the key in the sidebar and try again."
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "Your Gemini free quota is used up for now. Wait a minute and ask again."
    if "503" in text or "UNAVAILABLE" in text or "timed out" in text.lower():
        return "The model is busy right now. Wait a moment and ask again."
    return "Something went wrong while answering. Please try again. Details: " + text[:120]


def _generation_config(system_instruction: str):
    from google.genai import types

    return types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.4)


def stream_answer(
    client,
    prompt: str,
    system_instruction: str,
    models: list[str] | None = None,
    attempts: int = ATTEMPTS_PER_MODEL,
    pause: float = RETRY_PAUSE_SECONDS,
    config: Any = None,
) -> Iterator[str]:
    """Yield the answer in pieces, trying each model in turn.

    A rejected API key is raised straight away. If every attempt fails before
    any text arrives, the last error is raised so the caller can show a
    specific message (quota, busy, and so on). If a stream breaks after text
    has started, a short note is appended instead of retrying, because a retry
    would repeat text the reader has already seen.
    """
    models = models or GEN_MODELS
    config = config if config is not None else _generation_config(system_instruction)
    last_error: Exception | None = None
    for model_name in models:
        for _ in range(attempts):
            started = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=model_name, contents=prompt, config=config
                ):
                    if chunk.text:
                        started = True
                        yield chunk.text
                if started:
                    return
            except Exception as error:
                if is_key_error(error):
                    raise
                last_error = error
                if started:
                    yield INTERRUPTED_NOTE
                    return
            time.sleep(pause)
    if last_error is not None:
        raise last_error
    yield NO_RESPONSE_NOTE
