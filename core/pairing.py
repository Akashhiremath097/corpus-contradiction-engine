"""
Candidate pairing: given a list of extracted claims, finds which pairs of
claims are worth checking for contradiction.

This is the O(n^2) -> O(n*k) step from the deck's scalability slide, made
real: instead of comparing every claim to every other claim, we embed all
claims, build a FAISS index over them, and for each claim only look at its
top-k nearest neighbours. High semantic similarity is a prerequisite for
conflict, not evidence against it — two claims about completely different
topics can't contradict each other, so there's no reason to run the
expensive entailment check on them.
"""

from itertools import count

import faiss
import numpy as np

from core.claim_extraction import Claim
from core.embedding import get_model

# k is small here because our seeded corpus only has ~47 claims. At real
# scale (the deck's 50,000-chunk example) k=20 is what keeps ~1M candidate
# pairs instead of ~1.25B. At 47 claims, k=20 would just return almost every
# claim as a "neighbour" — a smaller k is what actually filters anything at
# this size.
DEFAULT_K = 5

# Below this cosine similarity, two claims are treated as "not even worth
# checking" — this is the threshold-gating mechanism from the deck: cheap
# filtering happens here, before any claim pair reaches the expensive
# LLM-based entailment step.
DEFAULT_SIMILARITY_THRESHOLD = 0.5


def build_claim_index(claims: list[Claim]) -> tuple[faiss.Index, list[str]]:
    """
    Same mechanics as embedding.build_index, applied to claims instead of
    chunks. Claims and chunks are different granularity — one chunk often
    yields 2-3 claims — so pairing needs its own index built at the claim
    level; contradiction detection compares individual assertions, not
    whole chunks.
    """
    model = get_model()  # reuses the same singleton embedding.py already loaded
    texts = [c.text for c in claims]
    claim_ids = [c.claim_id for c in claims]

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, claim_ids


def get_candidate_pairs(
    claims: list[Claim],
    k: int = DEFAULT_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[tuple[Claim, Claim, float]]:
    """
    Returns (claim_a, claim_b, similarity) tuples for every pair worth
    checking for contradiction.

    Three filtering rules matter here:
    1. A claim's own top-k neighbours always include itself (similarity
       1.0) — that's excluded, comparing a claim to itself is meaningless.
    2. FAISS search is directional (A's neighbours include B, and
       separately B's neighbours include A) — without dedup, every real
       pair would be checked twice. We keep pairs as a set keyed by sorted
       claim_id, so (A, B) and (B, A) collapse into one entry.
    3. Claims from the SAME document are excluded. A document can't
       contradict itself — when claim extraction splits one sentence into
       multiple claim fragments, those fragments are often each other's
       nearest neighbour by construction, not because they disagree.
       Contradiction is a cross-source concept; same-doc pairs are noise.
    """
    index, claim_ids = build_claim_index(claims)
    claims_by_id = {c.claim_id: c for c in claims}
    model = get_model()

    all_texts = [c.text for c in claims]
    query_vectors = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(query_vectors)

    seen_pairs = set()
    candidates: list[tuple[Claim, Claim, float]] = []

    # k+1 because each claim's own vector is always its closest match
    scores, indices = index.search(query_vectors, k + 1)

    for row_idx, claim_id in enumerate(claim_ids):
        for score, neighbour_idx in zip(scores[row_idx], indices[row_idx]):
            neighbour_id = claim_ids[neighbour_idx]

            if neighbour_id == claim_id:
                continue  # skip self-match
            if score < similarity_threshold:
                continue  # below the threshold-gating floor

            claim_obj = claims_by_id[claim_id]
            neighbour_obj = claims_by_id[neighbour_id]
            if claim_obj.doc_id == neighbour_obj.doc_id:
                continue  # skip same-document pairs — see docstring above

            pair_key = tuple(sorted([claim_id, neighbour_id]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            claim_a = claims_by_id[pair_key[0]]
            claim_b = claims_by_id[pair_key[1]]
            candidates.append((claim_a, claim_b, float(score)))

    return candidates


def verify_against_answer_key(
    candidates: list[tuple[Claim, Claim, float]],
    answer_key_path: str = "data/answer_key.json",
) -> None:
    """
    Cross-checks candidate pairs against the seeded answer key. For each
    known contradiction pair (doc_a, doc_b), reports whether at least one
    claim-level candidate pair spans those two documents — that's the real
    test of whether candidate pairing actually found it, versus just
    eyeballing a sorted list and hoping the right pairs are near the top.
    """
    import json

    with open(answer_key_path, "r", encoding="utf-8") as f:
        answer_key = json.load(f)

    # Build a lookup: for each (doc_a, doc_b) pair in our candidates,
    # what's the best similarity we found spanning those two docs?
    doc_pair_best_score: dict[tuple[str, str], float] = {}
    for claim_a, claim_b, score in candidates:
        key = tuple(sorted([claim_a.doc_id, claim_b.doc_id]))
        if key not in doc_pair_best_score or score > doc_pair_best_score[key]:
            doc_pair_best_score[key] = score

    print("\n--- Verification against seeded answer key ---")
    found_count = 0
    for entry in answer_key:
        doc_a = entry["doc_a"].replace(".txt", "")
        doc_b = entry["doc_b"].replace(".txt", "")
        key = tuple(sorted([doc_a, doc_b]))
        label = entry["label"]

        best_score = doc_pair_best_score.get(key)
        found = best_score is not None

        if label == "contradiction":
            status = "FOUND" if found else "MISSED"
            if found:
                found_count += 1
            print(f"  [{status:6}] {doc_a} <-> {doc_b}  "
                  f"(best similarity: {best_score:.3f})" if found else
                  f"  [{status:6}] {doc_a} <-> {doc_b}  (no candidate pair generated)")
        else:  # no_conflict — check it didn't accidentally surface
            status = "correctly absent" if not found else f"unexpectedly paired ({best_score:.3f})"
            print(f"  [distractor] {doc_a} <-> {doc_b}  -> {status}")

    total_seeded = sum(1 for e in answer_key if e["label"] == "contradiction")
    print(f"\nSeeded conflicts found by candidate pairing: {found_count}/{total_seeded}")
    if found_count < total_seeded:
        print("Missed pairs won't reach entailment classification later — "
              "consider lowering DEFAULT_SIMILARITY_THRESHOLD or rewording that seed pair.")


if __name__ == "__main__":
    from core.ingestion import load_corpus
    from core.claim_extraction import extract_claims_from_corpus

    chunks = load_corpus("data/corpus")
    print(f"Extracting claims from {len(chunks)} chunks...")
    claims = extract_claims_from_corpus(chunks)
    print(f"Total claims: {len(claims)}\n")

    naive_pairs = len(claims) * (len(claims) - 1) // 2
    candidates = get_candidate_pairs(claims)

    print(f"Naive pairwise comparisons would be: {naive_pairs}")
    print(f"Candidate pairs after neighbour-bounded search: {len(candidates)}")
    print(f"Reduction: {naive_pairs / max(len(candidates), 1):.1f}x fewer comparisons\n")

    print("Sample candidate pairs (sorted by similarity, highest first):")
    for claim_a, claim_b, score in sorted(candidates, key=lambda x: -x[2])[:10]:
        print(f"  [{score:.3f}] {claim_a.source} vs {claim_b.source}")
        print(f"      A: {claim_a.text}")
        print(f"      B: {claim_b.text}")

    verify_against_answer_key(candidates)