"""
Severity scoring: ranks confirmed contradictions by how urgent they are,
per the deck's formula — confidence x recency gap x retrieval frequency.

Not every contradiction is equally important. A high-confidence conflict
between a 2019 and a 2024 source (clearly superseded, large gap) is more
actionable than a lower-confidence conflict between two sources three
weeks apart. This is what lets the UI show "most severe first" instead of
an unordered list.
"""

from dataclasses import dataclass
from datetime import date

from core.entailment import EntailmentResult

# Retrieval frequency would, in a live system, come from query logs — how
# often each source chunk actually gets pulled into real answers. That data
# doesn't exist in a seeded offline demo, so it's stubbed at a constant.
# This keeps the formula's shape matching the deck exactly (three factors,
# not two) while being honest that this factor isn't measured here — a
# production deployment would replace this constant with a real counter.
STUBBED_RETRIEVAL_FREQUENCY = 1.0

# Recency gaps beyond this many days are treated as "maximally stale" for
# normalization purposes — without a cap, one extreme outlier (e.g. a
# 20-year-old source) would compress every other gap toward zero on a
# 0-1 scale, making them all look equally low-severity by comparison.
RECENCY_GAP_CAP_DAYS = 365 * 5  # 5 years


@dataclass
class ScoredContradiction:
    result: EntailmentResult
    recency_gap_days: int
    severity_score: float  # 0.0 - 1.0, higher = more severe / urgent


def _recency_gap_days(date_a: str, date_b: str) -> int:
    """Absolute difference in days between two ISO-format date strings."""
    da = date.fromisoformat(date_a)
    db = date.fromisoformat(date_b)
    return abs((da - db).days)


def _normalize_recency_gap(gap_days: int) -> float:
    """
    Maps a raw day-gap onto 0-1 using the cap above. Normalizing matters
    because confidence is already 0-1 — multiplying a 0-1 confidence by a
    raw day-count (which could be in the thousands) would let recency gap
    dominate the score regardless of how confident the model actually was.
    """
    return min(gap_days / RECENCY_GAP_CAP_DAYS, 1.0)


def score_contradiction(result: EntailmentResult) -> ScoredContradiction:
    gap_days = _recency_gap_days(result.claim_a.date, result.claim_b.date)
    normalized_recency = _normalize_recency_gap(gap_days)

    severity = result.confidence * normalized_recency * STUBBED_RETRIEVAL_FREQUENCY

    return ScoredContradiction(
        result=result,
        recency_gap_days=gap_days,
        severity_score=severity,
    )


def score_all(contradictions: list[EntailmentResult]) -> list[ScoredContradiction]:
    """Scores every confirmed contradiction and returns them sorted, most severe first."""
    scored = [score_contradiction(r) for r in contradictions]
    return sorted(scored, key=lambda s: -s.severity_score)


def group_by_document_pair(scored: list[ScoredContradiction]) -> dict[tuple[str, str], list[ScoredContradiction]]:
    """
    Groups claim-level scored contradictions by their source document pair.
    A single doc-pair (e.g. Dosage Guideline vs Dosage Protocol) can have
    multiple contradicting claim-pairs underneath it — this is what the
    Streamlit UI will actually display: one entry per document conflict,
    not one entry per claim-pair, since the latter would show near-duplicate
    rows for what a user experiences as a single policy conflict.
    """
    groups: dict[tuple[str, str], list[ScoredContradiction]] = {}
    for sc in scored:
        key = tuple(sorted([sc.result.claim_a.doc_id, sc.result.claim_b.doc_id]))
        groups.setdefault(key, []).append(sc)
    return groups


if __name__ == "__main__":
    from core.ingestion import load_corpus
    from core.claim_extraction import extract_claims_from_corpus
    from core.pairing import get_candidate_pairs
    from core.entailment import classify_candidates, get_contradictions

    chunks = load_corpus("data/corpus")
    print(f"Extracting claims from {len(chunks)} chunks...")
    claims = extract_claims_from_corpus(chunks)

    candidates = get_candidate_pairs(claims)
    print(f"{len(candidates)} candidate pairs, classifying...")
    results = classify_candidates(candidates)

    contradictions = get_contradictions(results)
    print(f"\n{len(contradictions)} claim-pairs classified as contradiction\n")

    scored = score_all(contradictions)
    grouped = group_by_document_pair(scored)

    print(f"Grouped into {len(grouped)} document-level conflicts, ranked by severity:\n")

    # rank doc-pairs by their highest-severity claim pair
    doc_pair_ranked = sorted(
        grouped.items(),
        key=lambda kv: -max(sc.severity_score for sc in kv[1])
    )

    for (doc_a, doc_b), scs in doc_pair_ranked:
        top = max(scs, key=lambda sc: sc.severity_score)
        print(f"[severity={top.severity_score:.3f}] {top.result.claim_a.source} vs {top.result.claim_b.source}")
        print(f"    gap: {top.recency_gap_days} days | confidence: {top.result.confidence:.2f}")
        print(f"    conflicting claims:")
        print(f"      A: {top.result.claim_a.text}")
        print(f"      B: {top.result.claim_b.text}")
        print()