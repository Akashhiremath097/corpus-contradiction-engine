"""
Cross-encoder reranking: a second, sharper cheap filter between candidate
pairing and Groq entailment classification.

FAISS's bi-encoder cosine similarity (already built in pairing.py) is good
for retrieval — "do these two texts embed close together" — but it's a
blunt instrument for contradiction specifically. A cross-encoder reads
both texts TOGETHER (not as two separately-embedded vectors) and, when
fine-tuned for natural language inference, can score "how likely is this
pair to actually be a contradiction" far more precisely — before a single
Groq call is spent confirming it.

This is the deck's own "cheap filters run first, the expensive model runs
only on what survives" principle, applied twice: cosine similarity is the
first filter, this cross-encoder is the second, and Groq remains the
final, authoritative classifier with full reasoning.
"""

from core.claim_extraction import Claim

# A compact cross-encoder fine-tuned specifically for NLI (entailment /
# neutral / contradiction), not just general similarity. Small variant
# chosen for the same reason as the MiniLM bi-encoder in embedding.py —
# this needs to score every candidate pair in seconds on a CPU.
RERANKER_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# This model's output label order, per its published config.
LABEL_ORDER = ["contradiction", "entailment", "neutral"]

# Deliberately permissive, not 0.5. This stage exists only to cut pairs
# that are almost certainly NOT contradictions — pruning the obviously
# irrelevant, not making the final call. Groq, with full source context
# and reasoning, remains the authoritative classifier. A high threshold
# here would risk discarding a real contradiction before Groq ever sees
# it, which is a much worse failure than letting a few weak candidates
# through to Groq.
DEFAULT_CONTRADICTION_THRESHOLD = 0.15

_reranker = None


def get_reranker():
    """Lazy singleton, same pattern as the embedding model and Groq client."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def score_pairs(pairs_text: list[tuple[str, str]]) -> list[list[float]]:
    """
    Runs the cross-encoder over a batch of (text_a, text_b) pairs and
    returns softmax probabilities in LABEL_ORDER for each. Split out from
    rerank_candidates so the probability math can be tested independently
    of loading the actual model (which needs a network download).
    """
    import torch

    model = get_reranker()
    raw_scores = model.predict(pairs_text)  # shape: (n, 3) raw logits
    probs = torch.nn.functional.softmax(torch.tensor(raw_scores), dim=1)
    return probs.tolist()


def rerank_candidates(
    candidates: list[tuple[Claim, Claim, float]],
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
) -> list[tuple[Claim, Claim, float]]:
    """
    Filters candidate pairs by cross-encoder contradiction probability,
    then sorts survivors by that probability descending.

    Returns the SAME 3-tuple shape pairing.py produces — (claim_a, claim_b,
    similarity) — not a 4-tuple with the cross-encoder score attached.
    This matters: entailment.classify_candidates already expects 3-tuples,
    and keeping the shape identical means this stage can be dropped into
    the pipeline (or removed) without touching classify_candidates at all.

    Sorting by contradiction probability (highest first) means that if a
    Groq rate limit or timeout forces the pipeline to stop partway through
    classification, the most promising candidates were already processed
    first — a partial run degrades gracefully instead of randomly.
    """
    if not candidates:
        return []

    pairs_text = [(a.text, b.text) for a, b, _ in candidates]
    probs = score_pairs(pairs_text)
    contradiction_idx = LABEL_ORDER.index("contradiction")

    scored = []
    for (claim_a, claim_b, similarity), prob_row in zip(candidates, probs):
        contradiction_prob = prob_row[contradiction_idx]
        if contradiction_prob >= contradiction_threshold:
            scored.append((claim_a, claim_b, similarity, contradiction_prob))

    scored.sort(key=lambda x: -x[3])

    # Strip the cross-encoder score back off before returning — see docstring.
    return [(a, b, sim) for a, b, sim, _prob in scored]


if __name__ == "__main__":
    from core.ingestion import load_corpus
    from core.claim_extraction import extract_claims_from_corpus
    from core.pairing import get_candidate_pairs

    chunks = load_corpus("data/corpus")
    claims = extract_claims_from_corpus(chunks)
    candidates = get_candidate_pairs(claims)
    print(f"{len(candidates)} candidate pairs before reranking (first run downloads the reranker model)...")

    reranked = rerank_candidates(candidates)
    print(f"{len(reranked)} candidate pairs survive reranking "
          f"({len(candidates) - len(reranked)} pruned before reaching Groq)\n")

    print("Reranked pairs (highest contradiction probability first):")
    for claim_a, claim_b, sim in reranked[:10]:
        print(f"  {claim_a.source} vs {claim_b.source}  (faiss_similarity={sim:.3f})")