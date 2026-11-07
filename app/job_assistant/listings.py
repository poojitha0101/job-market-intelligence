"""Turning retrieved listings into prompt text and into citations for the UI."""

import re

import pandas as pd


def count_unique(jobs: pd.DataFrame, column: str) -> int | None:
    """Count distinct non-empty values in a column, or None if it is absent."""
    if column not in jobs.columns:
        return None
    values = jobs[column].astype(str).str.strip()
    return int((values != "").groupby(values).any().sum())


def format_listings(hits: pd.DataFrame) -> str:
    """Render retrieved rows as numbered [Job n] blocks for the model."""
    text = ""
    for i, (_, row) in enumerate(hits.iterrows(), start=1):
        text += (
            f"[Job {i}] Title: {row['title']} | Company: {row['company.display_name']} "
            f"| Location: {row.get('location.display_name', 'N/A')}\n"
            f"Description: {row['description']}\n\n"
        )
    return text


def source_lines(hits: pd.DataFrame | None, reply: str) -> list[str]:
    """Markdown lines for the jobs a reply actually cites as [Job n].

    A listing link is added only when the data has a redirect_url column and
    the value is an http(s) URL.
    """
    if hits is None or "[Job" not in reply:
        return []
    numbers = sorted({int(n) for n in re.findall(r"Job (\d+)", reply)})
    lines = []
    for n in numbers:
        if 1 <= n <= len(hits):
            row = hits.iloc[n - 1]
            place = row.get("location.display_name", "")
            line = f"**[Job {n}]** {row['title']}, {row['company.display_name']}"
            if place:
                line += f" ({place})"
            url = str(row.get("redirect_url", ""))
            if url.startswith("http"):
                line += f" - [View listing]({url})"
            lines.append(line)
    return lines
