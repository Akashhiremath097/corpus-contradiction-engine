"""
FastAPI layer: wires the core pipeline (ingestion -> embedding -> claims ->
pairing -> entailment -> scoring) behind HTTP endpoints for the Streamlit UI.

Ingestion runs in a background THREAD, not FastAPI's built-in BackgroundTasks.
This matters: BackgroundTasks runs after the response is sent, but still on
the same single event loop FastAPI uses to handle every other request. Since
our pipeline makes blocking calls (Groq API, FAISS, sentence-transformers —
none of it is async), running it via BackgroundTasks would freeze the entire
server for the ~1-2 minutes the pipeline takes, including the /status
endpoint the UI is supposed to be polling during that time. A real thread
runs genuinely in parallel, so /status stays responsive throughout.
"""

import threading
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.ingestion import load_corpus
from core.claim_extraction import extract_claims_from_corpus
from core.pairing import get_candidate_pairs
from core.entailment import classify_candidates, get_contradictions
from core.scoring import score_all, group_by_document_pair

app = FastAPI(title="Corpus Contradiction Engine")

# Streamlit runs on a different port (8501) than FastAPI (8000) — browsers
# block cross-origin requests by default, so without this the UI simply
# can't reach the API at all. allow_origins=["*"] is fine for a hackathon
# demo running locally; a real deployment would restrict this to the
# actual frontend's origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CORPUS_DIR = "data/corpus"

# Single global in-memory state — deliberately simple. This is a one-user
# hackathon demo, not a multi-tenant service, so there's no need for job
# IDs or per-request tracking. A lock guards it because the background
# thread writes to it while request handlers read from it concurrently;
# without the lock, a request could theoretically read the dict mid-write
# and see an inconsistent combination of old and new fields.
_state_lock = threading.Lock()
_state = {
    "status": "idle",       # idle | running | done | error
    "progress": "",
    "error": None,
    "conflicts": None,      # populated once status == "done"
}


def _set_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)


def _get_state() -> dict:
    with _state_lock:
        return dict(_state)


def _run_pipeline():
    """
    The actual pipeline, run inside a background thread. Every stage
    updates _state["progress"] so /status has something meaningful to
    report while the user waits.
    """
    try:
        _set_state(status="running", progress="Loading and chunking corpus...", error=None)
        chunks = load_corpus(CORPUS_DIR)

        _set_state(progress=f"Extracting claims from {len(chunks)} chunks...")
        claims = extract_claims_from_corpus(chunks)

        _set_state(progress=f"Finding candidate pairs among {len(claims)} claims...")
        candidates = get_candidate_pairs(claims)

        _set_state(progress=f"Classifying {len(candidates)} candidate pairs...")
        results = classify_candidates(candidates)
        contradictions = get_contradictions(results)

        _set_state(progress=f"Scoring {len(contradictions)} contradictions...")
        scored = score_all(contradictions)
        grouped = group_by_document_pair(scored)

        conflicts = _serialize_grouped_conflicts(grouped)
        _set_state(status="done", progress="Complete.", conflicts=conflicts)

    except Exception as e:
        # Whatever fails (bad API key, network issue, malformed response),
        # the polling endpoint needs to be able to report it instead of
        # the thread just dying silently and /status hanging on "running"
        # forever.
        _set_state(status="error", progress="", error=str(e))


def _serialize_grouped_conflicts(grouped: dict) -> list[dict]:
    """
    Converts the grouped (doc_pair -> [ScoredContradiction]) structure into
    plain dicts/lists that FastAPI can return as JSON. dataclasses.asdict
    handles the nested Claim/EntailmentResult/ScoredContradiction structure
    automatically since they're all dataclasses.
    """
    conflicts = []
    for (doc_a, doc_b), scored_list in grouped.items():
        top = max(scored_list, key=lambda sc: sc.severity_score)
        conflicts.append({
            "doc_a": doc_a,
            "doc_b": doc_b,
            "source_a": top.result.claim_a.source,
            "source_b": top.result.claim_b.source,
            "date_a": top.result.claim_a.date,
            "date_b": top.result.claim_b.date,
            "top_severity": top.severity_score,
            "claim_pair_count": len(scored_list),
            "claim_pairs": [asdict(sc) for sc in scored_list],
        })
    conflicts.sort(key=lambda c: -c["top_severity"])
    return conflicts


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def start_ingestion():
    """
    Kicks off the pipeline in a background thread and returns immediately.
    If a run is already in progress, returns its current status instead of
    starting a second overlapping run — two threads both hitting the Groq
    API and writing to the same _state dict at once would be a real bug,
    not just wasted API calls.
    """
    state = _get_state()
    if state["status"] == "running":
        return {"message": "Ingestion already running.", "status": "running", "progress": state["progress"]}

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()
    return {"message": "Ingestion started.", "status": "running"}


@app.get("/status")
def get_status():
    """Polled by the UI to show progress and know when /conflicts is ready."""
    state = _get_state()
    return {
        "status": state["status"],
        "progress": state["progress"],
        "error": state["error"],
    }


@app.get("/conflicts")
def list_conflicts():
    """
    Returns the ranked list of document-level conflicts. 425 (Too Early)
    is the semantically correct status code here — it means exactly
    "the request isn't invalid, you're just asking before the resource
    exists yet," which is exactly this situation before /ingest finishes.
    """
    state = _get_state()
    if state["status"] != "done":
        raise HTTPException(
            status_code=425,
            detail=f"Ingestion not complete (status: {state['status']}). Call POST /ingest first.",
        )
    return {"conflicts": state["conflicts"]}


@app.get("/conflicts/{doc_a}/{doc_b}")
def get_conflict_detail(doc_a: str, doc_b: str):
    """
    Full detail for one document pair — every contradicting claim-pair
    between them, not just the top one. This is what the Streamlit
    click-through detail view calls.
    """
    state = _get_state()
    if state["status"] != "done":
        raise HTTPException(status_code=425, detail="Ingestion not complete.")

    key = tuple(sorted([doc_a, doc_b]))
    for conflict in state["conflicts"]:
        if tuple(sorted([conflict["doc_a"], conflict["doc_b"]])) == key:
            return conflict

    raise HTTPException(status_code=404, detail=f"No conflict found between {doc_a} and {doc_b}.")