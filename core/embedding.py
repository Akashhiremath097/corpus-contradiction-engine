"""
Embedding: turns Chunks into vectors and builds a FAISS index over them.

This is also where the same index gets reused for candidate pairing later —
per the architecture, retrieval and contradiction-pairing share one index,
so there's no second piece of infrastructure to build or keep in sync.
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from core.ingestion import Chunk

# all-MiniLM-L6-v2: small (~80MB), fast on CPU, 384-dim vectors.
# Chosen over larger models specifically because there's no GPU requirement
# and encoding a few thousand chunks needs to take seconds, not minutes,
# during a live demo or a re-index after adding a document.
MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = os.path.join("data", "index", "corpus.faiss")
MAPPING_PATH = os.path.join("data", "index", "chunk_mapping.json")

_model = None  # lazy-loaded singleton — see note below


def get_model() -> SentenceTransformer:
    """
    Loads the embedding model once and reuses it. Loading a SentenceTransformer
    is not free (~1-2s plus the one-time download), so a module-level singleton
    avoids reloading it on every function call — this matters once the API
    server is handling multiple requests.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_index(chunks: list[Chunk]) -> tuple[faiss.Index, list[str]]:
    """
    Encodes every chunk's text and builds a FAISS index over the vectors.

    Returns the index itself, plus chunk_ids in the same order the vectors
    were added — FAISS only knows integer positions (0, 1, 2, ...), not our
    string chunk_ids, so this ordered list is what lets us translate a FAISS
    search result ("closest vector is at position 7") back into
    "chunk hr_leave_policy_2023__chunk0".
    """
    model = get_model()
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # Normalize vectors so inner product == cosine similarity. Cosine is what
    # we actually want here (semantic closeness regardless of text length),
    # and IndexFlatIP (inner product) is faster than a similarity computed
    # after the fact on raw L2 vectors.
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunk_ids


def save_index(index: faiss.Index, chunk_ids: list[str]) -> None:
    """
    Persists the index and its chunk_id mapping to disk so later steps
    (or a re-run) don't need to re-embed everything from scratch.
    Both *.faiss and this mapping are in .gitignore — they're generated
    artifacts, not source, and regenerating them is one script run.
    """
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)


def load_index() -> tuple[faiss.Index, list[str]]:
    """Loads a previously saved index + chunk_id mapping from disk."""
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    return index, chunk_ids


def embed_query(text: str) -> np.ndarray:
    """
    Embeds a single piece of text (e.g. one extracted claim) for a
    similarity search against the index. Normalized the same way the
    index vectors are, so the comparison is apples-to-apples.
    """
    model = get_model()
    vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(vec)
    return vec


if __name__ == "__main__":
    from core.ingestion import load_corpus

    chunks = load_corpus("data/corpus")
    print(f"Embedding {len(chunks)} chunks (first run downloads the model, ~80MB)...")
    index, chunk_ids = build_index(chunks)
    print(f"Built index with {index.ntotal} vectors, dimension {index.d}")

    save_index(index, chunk_ids)
    print(f"Saved index to {INDEX_PATH}")

    # Sanity check: does the dosage guideline pull its own conflict partner
    # as a top neighbour? If not, the seeded conflict pairs may not be
    # semantically close enough for candidate pairing to find them later.
    query_chunk = next(c for c in chunks if c.chunk_id == "hc_dosage_guideline_2020__chunk0")
    query_vec = embed_query(query_chunk.text)
    scores, indices = index.search(query_vec, k=3)
    print(f"\nTop-3 neighbours of '{query_chunk.chunk_id}':")
    for score, idx in zip(scores[0], indices[0]):
        print(f"  {chunk_ids[idx]}  (similarity={score:.3f})")