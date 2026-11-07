# India Data & AI Job Market Assistant

A Streamlit chat app that answers questions about a sample of 784 Data and AI job listings in India, collected from the Adzuna API. Each answer points back to the listings it used.

- Live app: https://india-job-market-assistant.streamlit.app/
- Project overview (PDF): https://drive.google.com/file/d/1emgFfmqUzvZY93wY0g4QioUP4azlbtCj/view?usp=sharing

## Executive summary

**Problem.** Job boards filter by keyword, but the questions a job seeker actually has are looser: which roles need both Python and SQL, what Power BI roles ask for, who is hiring analysts. Answering those means opening listings one at a time and reading descriptions.

**Objective.** Let someone ask in plain English (or Telugu) and get a short answer that cites the specific listings behind it. When the listings do not contain the answer, the assistant says so instead of filling the gap.

**Scope.** The data is a sample, not the Indian job market. According to the project overview, 800 raw records were collected and 784 remained after removing duplicate job IDs. The assistant only sees the 8 closest listings per question, so it cannot count or rank across the sample. Totals and trends were meant for a Power BI dashboard described in the project overview.

**What this repository contains.** The assistant, its data files and its tests. The collection, cleaning, SQL and Power BI work described in the overview is not in this repository, and neither is the code that produced the embeddings (see [Limitations](#limitations)).

## Architecture and pipeline

```
question
   |
   |-- greeting or thanks? --> skip retrieval
   v
embed the question            Gemini embedding model, 768 dimensions
   v
cosine similarity             against data/job_embeddings.npy (784 x 768)
   v
top 8 listings                formatted as [Job 1] ... [Job 8]
   v
prompt                        system rules + last 6 messages + listings + question
   v
Gemini generation             streamed, falls back across models
   v
answer + "Jobs used"          only the [Job n] the answer cites
```

### Layout

```
app/
  app.py                  Streamlit page and chat flow
  job_assistant/
    config.py             paths, model names, limits, environment overrides
    retrieval.py          corpus loading, cosine search, small-talk check
    listings.py           listing text for the prompt, citation lines for the UI
    prompts.py            system prompt, prompt assembly, [[general]] marker
    llm.py                Gemini client, query embedding, streaming with fallback
  static/style.css        page styling
data/
  app_jobs.csv            784 listings: title, company, location, description
  job_embeddings.npy      784 x 768 float32, same row order as the CSV
tests/                    pytest suite (logic, data integrity)
.streamlit/config.toml    theme
.env.example              optional local configuration
```

`job_assistant` does not import Streamlit, and imports the Gemini SDK only inside the functions that call it, so the logic is testable without either installed.

### Technology

| Tool | Role | Why |
|---|---|---|
| Streamlit | Chat UI and hosting | One Python file per page, built-in chat widgets, free hosting for a portfolio project |
| Gemini embeddings | Turn listings and questions into vectors | Same provider as the answer model, so one API key covers both |
| NumPy | Similarity search | 784 vectors is one matrix-vector product; see decision 1 |
| Gemini generation models | Write the answer from retrieved listings | Free tier lets visitors bring their own key |
| pandas | Load and format the CSV | Listings are a small table |
| pytest | Tests | Logic and data checks run in about a tenth of a second |

## Quick start

Requires Python 3.10 or newer and a free Gemini API key from https://aistudio.google.com/apikey.

```bash
git clone https://github.com/poojitha0101/job-market-intelligence.git
cd job-market-intelligence

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app/app.py
```

Paste the API key into the sidebar. The key lives in the browser session and is not stored.

To avoid pasting it every time during local development:

```bash
cp .env.example .env               # then set GEMINI_API_KEY in .env
```

Run the tests:

```bash
pip install -r requirements-dev.txt
pytest
```

### Configuration

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Used when the sidebar field is empty. Local use only: on a public deployment it would spend your quota for every visitor. |
| `GEMINI_MODELS` | Comma-separated answer models, tried in order. Defaults are in `app/job_assistant/config.py`. |

Retrieval depth, history length, timeouts and retry counts are constants in `config.py`.

## Engineering decisions

1. **Brute-force search in NumPy, no vector database.** At 784 rows the search is a single matrix-vector product. A database would add a service to run and deploy without changing the result. The cost is that it does not scale: past tens of thousands of listings this needs an index.

2. **Listings are embedded ahead of time; only the question is embedded at runtime.** One embedding call per question, and startup is a file read. At 768 dimensions the vector file is 2.4 MB, small enough to commit next to the CSV. The constraint is that the question must use the same model and dimension as the stored vectors, which `tests/test_data.py` checks for dimension and normalisation.

3. **Grounding through the prompt.** The system prompt tells the model to answer from the supplied listings only, cite `[Job n]`, say "in this sample", and decline to count or rank. This is cheap and works well, but it relies on the model following instructions. The "Jobs used" list shows which listings the answer cited; it does not check that those listings support the claim.

4. **Only the top 8 listings reach the model.** This keeps the prompt short and the answers focused. It also means questions like "how many jobs mention PySpark" cannot be answered here. The prompt tells the model to say so and point to the dashboard.

5. **Small talk skips retrieval.** A whole-message regex catches greetings and thanks, which saves an embedding call and keeps unrelated listings out of the prompt. It is deliberately strict: "hi, which jobs need SQL?" is treated as a real question. New phrasings of small talk will go through retrieval until added to the pattern.

6. **Short follow-ups borrow the previous question for retrieval.** A message of six words or fewer, such as "and in Hyderabad?", is searched together with the last user question. This avoids a second model call to rewrite the query. It is wrong when the follow-up changes topic.

7. **Bring your own API key.** No server-side secret and no cost to the project owner. Visitors need a Google account to get a key, and quota errors are theirs.

8. **Model fallback with bounded retries.** Each answer tries up to three models, two attempts each, with a 2 second pause. A rejected key fails immediately. A stream that breaks after text has started is not retried, since a retry would repeat text the reader has already seen. If every attempt fails, the last error is raised so the user sees a specific message (quota, busy) rather than a generic one. Worst case before giving up is roughly three minutes, from the 30 second request timeout.

9. **General questions are kept free of listing references.** For "what is RAG" the model is told to answer from general knowledge and start with `[[general]]`. The app strips that marker from the streamed text.

10. **Package inside `app/`.** Streamlit puts the script's folder on the import path, so `job_assistant` resolves without changing how the app is launched or deployed. The entry point is still `app/app.py`.

## Limitations

- **Sample, not market.** Answers describe 784 listings and say so.
- **Truncated descriptions.** Descriptions are cut at about 500 characters (the longest in the file is exactly 500), so skills mentioned later in a posting are invisible to the assistant.
- **Missing columns.** The shipped CSV has no listing URL and no role column. The "View listing" link and the "roles" figure in the header are supported by the code but never appear until those columns are added.
- **Possible duplicates.** 70 rows repeat the title, company and location of another row. They may be separate postings. The CSV has no job ID, so this cannot be settled from the repository.
- **Embeddings are not reproducible from this repo.** There is no embedding script, and the text embedded for each listing is not recorded. Loading checks that the row counts match, but cannot detect rows that were reordered or edited.
- **No quality evaluation.** There is no labelled question set for retrieval hit rate or answer faithfulness. The tests cover logic with stubbed model calls, not model behaviour.
- **UI untested.** The Streamlit page itself has no automated tests, and the default Gemini model names were carried over unchanged and not re-verified.
- **Loose dependency bounds.** `requirements.txt` sets minimum versions only, with no lock file.
- **Analysis not included.** Data collection, SQL analysis and the Power BI dashboard are not in this repository.

### Next steps

- Add `scripts/build_embeddings.py`, record the exact text embedded per listing, and add job ID and URL columns to the CSV.
- Build a set of 30 to 50 labelled questions and measure how often the right listing appears in the top 8.
- Combine keyword and vector search, so exact terms such as tool names are not blurred by embeddings.
- Answer count and ranking questions with fixed SQL queries over the SQLite data instead of redirecting to the dashboard.
- Add a lock file and a CI job that runs `pytest`.
- Add the collection, cleaning and SQL work under an `analysis/` folder so the repository matches the project overview.

## Author

Poojitha Nallamaru. A personal portfolio project for practising data analytics and retrieval-based question answering.
