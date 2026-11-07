import pandas as pd

from job_assistant.listings import count_unique, format_listings, source_lines


def test_format_listings_numbers_from_one(jobs):
    text = format_listings(jobs)
    assert text.startswith("[Job 1] Title: Data Analyst | Company: Acme | Location: Hyderabad, Telangana")
    assert "[Job 3] Title: BI Developer" in text
    assert "Description: Power BI and DAX" in text


def test_source_lines_only_cited_jobs(jobs):
    lines = source_lines(jobs, "Job 2 fits [Job 2], and also [Job 1].")
    assert len(lines) == 2
    assert lines[0].startswith("**[Job 1]** Data Analyst, Acme (Hyderabad, Telangana)")
    assert lines[1].startswith("**[Job 2]** ML Engineer, Globex")


def test_source_lines_ignores_out_of_range_numbers(jobs):
    assert source_lines(jobs, "See [Job 9] and [Job 0].") == []


def test_source_lines_empty_when_no_hits_or_no_citation(jobs):
    assert source_lines(None, "[Job 1]") == []
    assert source_lines(jobs, "No citations here.") == []


def test_source_lines_adds_link_only_for_http_urls(jobs):
    jobs = jobs.assign(redirect_url=["https://example.com/a", "javascript:alert(1)", ""])
    lines = source_lines(jobs, "[Job 1] [Job 2] [Job 3]")
    assert "[View listing](https://example.com/a)" in lines[0]
    assert "View listing" not in lines[1]
    assert "View listing" not in lines[2]


def test_source_lines_works_without_optional_columns(jobs):
    assert "View listing" not in source_lines(jobs, "[Job 1]")[0]


def test_count_unique(jobs):
    assert count_unique(jobs, "company.display_name") == 2  # blank is not counted
    assert count_unique(jobs, "missing_column") is None


def test_count_unique_deduplicates():
    frame = pd.DataFrame({"c": ["a", "a", " a ", "b", ""]})
    assert count_unique(frame, "c") == 2
