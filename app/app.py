"""Streamlit entry point for the India Data & AI Job Market Assistant.

Run from the repository root:  streamlit run app/app.py
The retrieval and prompting logic lives in the job_assistant package.
"""

import streamlit as st
from dotenv import load_dotenv

from job_assistant import config
from job_assistant.listings import count_unique, format_listings, source_lines
from job_assistant.llm import embed_query, friendly_error, make_client, stream_answer
from job_assistant.prompts import build_prompt, build_system_instruction, split_marker
from job_assistant.retrieval import build_search_query, is_small_talk, load_corpus, search_jobs

load_dotenv()

st.set_page_config(page_title="India Data & AI Job Market Assistant", layout="centered")

USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/smart_toy:"
STARTER_QUESTIONS = [
    "Which jobs need both Python and SQL?",
    "What skills do Power BI jobs ask for?",
    "Which jobs mention LLM or RAG?",
    "Which companies are hiring Data Analysts?",
    "Explain what RAG means",
    "What does this project cover?",
]


# ------------------------------------------------------------------ setup
def inject_styles() -> None:
    """Apply the page styling from static/style.css."""
    css = config.STYLE_CSS.read_text(encoding="utf-8")
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


@st.cache_resource
def get_corpus():
    """Load listings and embeddings once per server process."""
    return load_corpus()


# ------------------------------------------------------------------ page sections
def render_header(jobs) -> None:
    """Title, one-line description and headline counts."""
    stat_items = [(f"{len(jobs):,}", "job listings")]
    for column, label in (
        ("search_role", "roles"),
        ("company.display_name", "companies"),
        ("location.display_name", "locations"),
    ):
        n = count_unique(jobs, column)
        if n:
            stat_items.append((f"{n:,}", label))
    stats_html = "".join(f'<div class="stat"><b>{v}</b><span>{l}</span></div>' for v, l in stat_items)
    st.markdown(
        '<div class="hero"><h1>India Data &amp; AI Job Market Assistant</h1>'
        '<p class="lede">Ask about roles, skills, companies and locations. '
        "Answers come from a sample of Adzuna job listings, not the whole Indian job market.</p>"
        f'<div class="stats">{stats_html}</div></div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Sidebar with the API key field. Returns the key to use, possibly empty."""
    with st.sidebar:
        st.header("Gemini API key")
        typed_key = st.text_input("Paste your key", type="password")
        st.markdown(
            "Get a free key at [Google AI Studio](https://aistudio.google.com/apikey). "
            "It stays in your browser session and is not saved."
        )
        st.divider()
        st.subheader("About this assistant")
        st.markdown(
            "- Job answers come only from the collected sample\n"
            "- Descriptions may be cut off at about 500 characters\n"
            "- Salary is available only for some listings\n"
            "- Counts and averages are in the Power BI dashboard"
        )
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.editing = False
            st.rerun()
    return typed_key or config.env_api_key()


def show_sources(lines: list[str] | None) -> None:
    """Collapsible list of the jobs an answer cited."""
    if lines:
        with st.expander("Jobs used for this answer"):
            for line in lines:
                st.markdown(line)


def render_history() -> None:
    """Replay earlier messages after a Streamlit rerun."""
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else BOT_AVATAR):
            st.markdown(m["content"])
            show_sources(m.get("sources"))


def render_starters(container) -> None:
    """Example question buttons, shown only before the first message."""
    with container.container():
        st.markdown('<p class="starter-label">Try asking</p>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, text in enumerate(STARTER_QUESTIONS):
            if cols[i % 2].button(text, key=f"starter_{i}", use_container_width=True):
                st.session_state.queued = text
                st.rerun()


# ------------------------------------------------------------------ answering
def answer_question(question, api_key, jobs, doc_vecs, system_instruction) -> None:
    """Show the question, retrieve listings, stream the answer, record both."""
    st.session_state.editing = False
    history = list(st.session_state.messages)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        if not api_key:
            reply = "Paste your Gemini API key in the sidebar first, then ask again."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            return
        try:
            client = make_client(api_key)
            hits = None
            listings_text = ""
            if not is_small_talk(question):
                with st.spinner("Searching the listings"):
                    query_vec = embed_query(client, build_search_query(question, history))
                    hits = search_jobs(jobs, doc_vecs, query_vec)
                listings_text = format_listings(hits)
            prompt = build_prompt(question, history, listings_text)
            box = st.empty()
            raw = ""
            for piece in stream_answer(client, prompt, system_instruction):
                raw += piece
                box.markdown(split_marker(raw)[1] + " ▌")
            _, reply = split_marker(raw)
            sources = source_lines(hits, reply)
            box.markdown(reply)
            show_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "sources": sources}
            )
        except Exception as error:
            reply = friendly_error(error)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})


def last_user_index(messages: list[dict]) -> int | None:
    """Index of the most recent user message, or None."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    return None


def render_edit_controls() -> None:
    """Let the visitor rewrite and resend their last question."""
    msgs = st.session_state.messages
    last_i = last_user_index(msgs)
    if last_i is None:
        return
    if not st.session_state.get("editing"):
        if st.button("Edit last question", key="edit_btn"):
            st.session_state.edit_nonce = st.session_state.get("edit_nonce", 0) + 1
            st.session_state.editing = True
            st.rerun()
        return
    nonce = st.session_state.get("edit_nonce", 0)
    new_text = st.text_area(
        "Edit your last question",
        value=msgs[last_i]["content"],
        key=f"edit_text_{nonce}",
        max_chars=config.MAX_QUESTION_CHARS,
    )
    send_col, cancel_col, _ = st.columns([2, 1, 3])
    if send_col.button("Send edited question", key="edit_send"):
        cleaned = new_text.strip()
        if cleaned:
            st.session_state.messages = msgs[:last_i]
            st.session_state.queued = cleaned
        st.session_state.editing = False
        st.rerun()
    if cancel_col.button("Cancel", key="edit_cancel"):
        st.session_state.editing = False
        st.rerun()


# ------------------------------------------------------------------ page
def main() -> None:
    inject_styles()
    jobs, doc_vecs = get_corpus()
    system_instruction = build_system_instruction(len(jobs))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_header(jobs)
    api_key = render_sidebar()
    if not api_key:
        st.info("Paste your free Gemini API key in the sidebar to start chatting.")

    render_history()

    starter_box = st.empty()
    if not st.session_state.messages:
        render_starters(starter_box)

    typed = st.chat_input("Ask about jobs, skills, or this project")
    question = typed or st.session_state.pop("queued", None)
    if question:
        question = question.strip()[: config.MAX_QUESTION_CHARS]
    if question:
        starter_box.empty()
        answer_question(question, api_key, jobs, doc_vecs, system_instruction)

    render_edit_controls()


main()
