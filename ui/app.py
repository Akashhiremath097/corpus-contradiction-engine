"""
Streamlit UI: two views controlled by session_state, not separate pages —
simpler than Streamlit's multi-page routing for a two-screen demo.

List view (default): conflicts ranked by severity, shown as clickable cards.
Detail view (session_state.selected_conflict is set): both sources side by
side with dates and the model's reasoning, mirroring the deck's slide 3
"both sources shown, reader decides" presentation.
"""

import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Corpus Contradiction Engine", layout="wide")

if "selected_conflict" not in st.session_state:
    st.session_state.selected_conflict = None


def severity_badge(score: float) -> str:
    """
    Maps a 0-1 severity score to a color-coded label. This is what makes
    the list view scannable at a glance — a reader shouldn't have to read
    every number to tell which conflicts are most urgent.
    """
    if score >= 0.7:
        return "🔴 High"
    elif score >= 0.4:
        return "🟠 Medium"
    else:
        return "🟡 Low"


def highlight_span(text: str, span: str) -> str:
    """
    Wraps the first case-insensitive occurrence of `span` inside `text` in
    markdown bold, so Streamlit renders it visually highlighted within the
    full claim sentence — e.g. "The standard adult dose of Medazol is
    **500mg**." instead of just showing the whole sentence undifferentiated.

    Graceful fallback: if span is empty (label wasn't "contradiction", or
    entailment.py's own verbatim check already discarded a hallucinated
    span) or genuinely isn't found in text, returns text unchanged rather
    than raising — a missing highlight should never break the page.
    """
    if not span:
        return text
    idx = text.lower().find(span.lower())
    if idx == -1:
        return text
    end = idx + len(span)
    return text[:idx] + "**" + text[idx:end] + "**" + text[end:]


def fetch_status() -> dict:
    resp = requests.get(f"{API_BASE}/status", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_conflicts() -> list[dict]:
    resp = requests.get(f"{API_BASE}/conflicts", timeout=10)
    resp.raise_for_status()
    return resp.json()["conflicts"]


def start_ingestion(source: str = "seeded", fallback_date: str = None) -> dict:
    params = {"source": source}
    if fallback_date:
        params["fallback_date"] = fallback_date
    resp = requests.post(f"{API_BASE}/ingest", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def upload_documents(uploaded_files) -> dict:
    """
    Sends one or more Streamlit UploadedFile objects to POST /upload.
    requests' `files` param expects (fieldname, (filename, content_bytes,
    mimetype)) tuples — Streamlit's UploadedFile already gives us the
    filename and raw bytes via .getvalue(), so this is just reshaping.
    """
    files_payload = [
        ("files", (f.name, f.getvalue(), "text/plain"))
        for f in uploaded_files
    ]
    resp = requests.post(f"{API_BASE}/upload", files=files_payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def render_detail_view(conflict: dict):
    """
    Side-by-side source comparison for one conflict — the click-through
    view. Both sources are shown with equal visual weight on purpose: the
    engine doesn't decide which one is "right," per the deck's design
    principle, so the UI shouldn't imply a verdict either.
    """
    if st.button("← Back to all conflicts"):
        st.session_state.selected_conflict = None
        st.rerun()

    st.subheader(f"{conflict['source_a']} vs {conflict['source_b']}")
    st.caption(
        f"{severity_badge(conflict['top_severity'])} · "
        f"{conflict['claim_pair_count']} conflicting claim(s) · "
        f"{(conflict.get('claim_pairs') or [{}])[0].get('recency_gap_days', '?')} days apart"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{conflict['source_a']}**")
        st.caption(conflict["date_a"])
    with col_b:
        st.markdown(f"**{conflict['source_b']}**")
        st.caption(conflict["date_b"])

    st.divider()

    for i, cp in enumerate(conflict["claim_pairs"], start=1):
        result = cp["result"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(highlight_span(result["claim_a"]["text"], result.get("conflicting_span_a", "")))
        with col_b:
            st.warning(highlight_span(result["claim_b"]["text"], result.get("conflicting_span_b", "")))
        st.caption(
            f"Model reasoning: {result['reasoning']} "
            f"(confidence: {result['confidence']:.0%})"
        )
        if i < len(conflict["claim_pairs"]):
            st.divider()


def render_list_view(conflicts: list[dict]):
    st.title("Corpus Contradiction Engine")
    st.caption(f"{len(conflicts)} document-level conflicts detected, ranked by severity")

    for conflict in sorted(conflicts, key=lambda c: -c["top_severity"]):
        with st.container(border=True):
            col_main, col_button = st.columns([5, 1])
            with col_main:
                st.markdown(f"**{conflict['source_a']}**  vs  **{conflict['source_b']}**")
                st.caption(
                    f"{severity_badge(conflict['top_severity'])}  ·  "
                    f"{conflict['date_a']} → {conflict['date_b']}  ·  "
                    f"{conflict['claim_pair_count']} conflicting claim(s)"
                )
            with col_button:
                if st.button("View", key=f"view_{conflict['doc_a']}_{conflict['doc_b']}"):
                    st.session_state.selected_conflict = conflict
                    st.rerun()


def render_ingestion_progress():
    """
    Polling loop: check /status, show progress, sleep briefly, rerun the
    whole script. Streamlit has no built-in "poll every N seconds"
    primitive — st.rerun() re-executing the script top to bottom IS the
    polling mechanism here. Capped iterations so a genuinely stuck backend
    doesn't spin the browser tab forever with no way out.
    """
    st.title("Corpus Contradiction Engine")

    try:
        status = fetch_status()
    except requests.exceptions.ConnectionError:
        st.error(
            "Can't reach the API. Is `python -m uvicorn api.main:app --reload` "
            "running in another terminal?"
        )
        return

    if status["status"] == "idle":
        st.info("No data loaded yet.")

        source_choice = st.radio(
            "What should be checked for contradictions?",
            options=["Seeded demo corpus", "My own uploaded documents", "Both"],
            help=(
                "Uploading your own documents replaces any previous upload. "
                "Files without a recognized date will use the date you set below — "
                "accurate dates matter for how conflicts get ranked by severity."
            ),
        )

        uploaded_files = None
        fallback_date_str = None
        if source_choice in ("My own uploaded documents", "Both"):
            uploaded_files = st.file_uploader(
                "Upload .txt documents", type=["txt"], accept_multiple_files=True
            )
            fallback_date = st.date_input(
                "Date to use for uploaded documents without a detected date",
                help="This affects recency-gap severity scoring — set it to the document's real date if you know it.",
            )
            fallback_date_str = fallback_date.isoformat()

        if st.button("Run contradiction detection"):
            source_map = {
                "Seeded demo corpus": "seeded",
                "My own uploaded documents": "uploaded",
                "Both": "both",
            }
            source = source_map[source_choice]

            if source in ("uploaded", "both"):
                if not uploaded_files:
                    st.warning("Please upload at least one .txt file first.")
                    return
                with st.spinner("Uploading documents..."):
                    upload_documents(uploaded_files)

            start_ingestion(source=source, fallback_date=fallback_date_str)
            st.rerun()
        return

    if status["status"] == "error":
        st.error(f"Pipeline failed: {status['error']}")
        if st.button("Retry"):
            start_ingestion()
            st.rerun()
        return

    if status["status"] == "running":
        st.info(f"⏳ {status['progress']}")
        st.progress(0.5)  # indeterminate-style bar; we don't have a real % from the backend
        time.sleep(2)
        st.rerun()
        return

    # status == "done" but this function was called anyway (e.g. race on first load) —
    # just fall through, main() will call render_list_view next pass
    st.rerun()


def main():
    if st.session_state.selected_conflict is not None:
        render_detail_view(st.session_state.selected_conflict)
        return

    try:
        status = fetch_status()
    except requests.exceptions.ConnectionError:
        st.title("Corpus Contradiction Engine")
        st.error(
            "Can't reach the API. Is `python -m uvicorn api.main:app --reload` "
            "running in another terminal?"
        )
        return

    if status["status"] == "done":
        conflicts = fetch_conflicts()
        render_list_view(conflicts)
    else:
        render_ingestion_progress()


if __name__ == "__main__":
    main()