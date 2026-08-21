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

import os
import re
import threading
from dataclasses import asdict
from datetime import date

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from core.ingestion import load_corpus
from core.claim_extraction import extract_claims_from_corpus
from core.pairing import get_candidate_pairs
from core.reranking import rerank_candidates
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

CORPUS_DIR = "data/corpus"       # the seeded demo corpus, built by create_corpus.py
UPLOADED_DIR = "data/uploaded"   # user-uploaded documents, populated via POST /upload

# Set ENABLE_RERANKING=false on memory-constrained hosts (e.g. Render's
# free 512MB tier) to skip loading the cross-encoder reranker model
# entirely. The reranker is a quality/efficiency improvement on top of an
# already-working pipeline (verified 8/8 recall, 0 false positives
# without it) — not a required stage. On a host tight enough that two
# loaded transformer models risk an OOM kill, dropping this one is a
# reasonable trade: the pipeline still runs correctly, just without the
# extra pruning step before Groq. Defaults to true (enabled) for local
# development, where memory isn't the constraint.
ENABLE_RERANKING = os.environ.get("ENABLE_RERANKING", "true").lower() == "true"

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


def _run_pipeline(source: str = "seeded", fallback_date: str = None):
    """
    The actual pipeline, run inside a background thread. Every stage
    updates _state["progress"] so /status has something meaningful to
    report while the user waits.

    source picks which corpus directory(ies) to run against:
      "seeded"   — only the built-in demo corpus (default, existing behavior)
      "uploaded" — only whatever the user has uploaded via POST /upload
      "both"     — combines them
    A company checking their own policies almost certainly wants "uploaded"
    only — mixing in the fictional demo docs would pollute their real results.
    """
    try:
        if source == "seeded":
            corpus_dirs = [CORPUS_DIR]
        elif source == "uploaded":
            corpus_dirs = [UPLOADED_DIR]
        else:  # "both"
            corpus_dirs = [CORPUS_DIR, UPLOADED_DIR]

        _set_state(status="running", progress="Loading and chunking corpus...", error=None)
        chunks = load_corpus(corpus_dirs, fallback_date=fallback_date)

        _set_state(progress=f"Extracting claims from {len(chunks)} chunks...")
        claims = extract_claims_from_corpus(chunks)

        _set_state(progress=f"Finding candidate pairs among {len(claims)} claims...")
        candidates = get_candidate_pairs(claims)

        if ENABLE_RERANKING:
            _set_state(progress=f"Reranking {len(candidates)} candidates with cross-encoder...")
            reranked = rerank_candidates(candidates)
        else:
            # Cross-encoder skipped (ENABLE_RERANKING=false) — the pipeline
            # still works correctly with the FAISS-only candidates, just
            # without the extra pruning pass. This is exactly the pipeline
            # shape that was already verified at 8/8 recall, 0 false
            # positives before the reranker was added.
            reranked = candidates

        _set_state(progress=f"Classifying {len(reranked)} candidate pairs...")
        results = classify_candidates(reranked)
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
def start_ingestion(source: str = "seeded", fallback_date: str = None):
    """
    Kicks off the pipeline in a background thread and returns immediately.
    If a run is already in progress, returns its current status instead of
    starting a second overlapping run — two threads both hitting the Groq
    API and writing to the same _state dict at once would be a real bug,
    not just wasted API calls.

    source: "seeded" (default) | "uploaded" | "both"
    fallback_date: used for any uploaded file that has no TITLE/SOURCE/DATE
        header (i.e. any real-world document) — defaults to today if omitted.
    """
    if source not in ("seeded", "uploaded", "both"):
        raise HTTPException(status_code=400, detail="source must be one of: seeded, uploaded, both")

    if source in ("uploaded", "both") and not os.path.isdir(UPLOADED_DIR):
        raise HTTPException(
            status_code=400,
            detail="No uploaded documents found. Call POST /upload first.",
        )

    state = _get_state()
    if state["status"] == "running":
        return {"message": "Ingestion already running.", "status": "running", "progress": state["progress"]}

    effective_fallback_date = fallback_date or date.today().isoformat()
    thread = threading.Thread(
        target=_run_pipeline,
        kwargs={"source": source, "fallback_date": effective_fallback_date},
        daemon=True,
    )
    thread.start()
    return {"message": "Ingestion started.", "status": "running", "source": source}


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Accepts one or more .txt files and saves them into UPLOADED_DIR,
    replacing whatever was there before — each upload batch is a fresh
    set, not an accumulation across multiple visits, since a stale
    document silently left over from an earlier test would otherwise
    quietly show up in later results.

    Files are saved as-is; header format (or lack of it) is handled at
    ingestion time by _parse_document's fallback, not here — this endpoint
    just needs to get bytes onto disk safely.
    """
    os.makedirs(UPLOADED_DIR, exist_ok=True)

    # Clear previous uploads before saving the new batch
    for existing in os.listdir(UPLOADED_DIR):
        os.remove(os.path.join(UPLOADED_DIR, existing))

    saved = []
    skipped = []
    for upload in files:
        if not upload.filename.endswith(".txt"):
            skipped.append(upload.filename)
            continue

        # Sanitize the filename — strip anything that isn't alphanumeric,
        # dash, underscore, or dot, so a malicious or malformed filename
        # can't be used for a path-traversal write (e.g. "../../evil.txt").
        safe_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", upload.filename)
        dest_path = os.path.join(UPLOADED_DIR, safe_name)

        content = await upload.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        saved.append(safe_name)

    return {"saved": saved, "skipped_non_txt": skipped}


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