"""
Entailment classification: the heart of the pipeline, per the deck (slide 5,
stage 3) — everything else is standard RAG plumbing, this is the genuinely
novel step. For each candidate pair from pairing.py, ask the model whether
the two claims agree, contradict, or are merely related-but-neutral.
"""

import json

from groq import Groq
from dataclasses import dataclass

from core.claim_extraction import Claim, call_groq_with_retry, _strip_markdown_fences

VALID_LABELS = {"contradiction", "entailment", "neutral"}

ENTAILMENT_PROMPT = """You are checking whether two claims from different documents contradict each other.

Claim A (source: {source_a}, dated {date_a}): "{text_a}"
Claim B (source: {source_b}, dated {date_b}): "{text_b}"

Classify the relationship between Claim A and Claim B as exactly one of:
- "contradiction": the two claims cannot both be true at the same time (different specific values, incompatible rules, or opposing requirements)
- "entailment": one claim implies or restates the other; they agree or are consistent
- "neutral": the claims are topically related but neither confirms nor contradicts the other

Be careful: two claims about the same TOPIC are not automatically a contradiction. Only classify as "contradiction" if they genuinely cannot both be true — e.g. different numbers for the same specific rule, or directly opposing requirements. A claim that adds detail or context to another, without conflicting, is "entailment" or "neutral", not "contradiction".

If the label is "contradiction", also identify the SHORTEST exact substring from each claim that captures the actual point of disagreement — e.g. just "500mg" and "250 mg administered twice daily", not the whole sentence. These must be copied VERBATIM, character-for-character, from the claim text given above — do not paraphrase or fix formatting. If the label is not "contradiction", leave both span fields as empty strings.

Respond with ONLY a JSON object in this exact form, nothing else. No markdown fences, no preamble.
{{"label": "contradiction" | "entailment" | "neutral", "confidence": <0.0 to 1.0>, "reasoning": "<one brief sentence>", "conflicting_span_a": "<verbatim substring of Claim A, or empty string>", "conflicting_span_b": "<verbatim substring of Claim B, or empty string>"}}"""


@dataclass
class EntailmentResult:
    claim_a: Claim
    claim_b: Claim
    label: str
    confidence: float
    reasoning: str
    similarity: float  # carried over from candidate pairing — used later in severity scoring
    conflicting_span_a: str = ""  # verbatim substring of claim_a.text pinpointing the conflict, if label == "contradiction"
    conflicting_span_b: str = ""  # verbatim substring of claim_b.text, same purpose


def classify_pair(claim_a: Claim, claim_b: Claim, similarity: float) -> EntailmentResult:
    """
    One Groq call per candidate pair. temperature=0.1 for the same reason
    as claim extraction: this is a classification task, not a creative one,
    and low temperature keeps the label consistent if you rerun the pipeline.
    """
    prompt = ENTAILMENT_PROMPT.format(
        source_a=claim_a.source, date_a=claim_a.date, text_a=claim_a.text,
        source_b=claim_b.source, date_b=claim_b.date, text_b=claim_b.text,
    )

    response = call_groq_with_retry(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    cleaned = _strip_markdown_fences(raw)

    try:
        parsed = json.loads(cleaned)
        label = parsed["label"]
        confidence = float(parsed["confidence"])
        reasoning = parsed.get("reasoning", "")
        span_a = parsed.get("conflicting_span_a", "")
        span_b = parsed.get("conflicting_span_b", "")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(
            f"Model returned unparseable entailment response for "
            f"{claim_a.claim_id} vs {claim_b.claim_id}. "
            f"Raw response (first 200 chars): {raw[:200]}"
        ) from e

    if label not in VALID_LABELS:
        # Defensive fallback: an unexpected label (e.g. the model invents
        # "partial_contradiction") is treated as neutral rather than
        # crashing the whole pipeline over one bad response.
        label = "neutral"

    # Verbatim check: the model is asked for an exact substring, but LLMs
    # sometimes paraphrase or fix punctuation even when told not to. A
    # span that isn't actually IN the source text can't be highlighted
    # meaningfully, so it's discarded here rather than passed downstream
    # for the UI to fail on — the UI's highlight function does its own
    # verbatim check too, but catching it here means EntailmentResult
    # never claims to have a span it can't actually back up.
    if span_a and span_a.lower() not in claim_a.text.lower():
        span_a = ""
    if span_b and span_b.lower() not in claim_b.text.lower():
        span_b = ""

    return EntailmentResult(
        claim_a=claim_a,
        claim_b=claim_b,
        label=label,
        confidence=confidence,
        reasoning=reasoning,
        similarity=similarity,
        conflicting_span_a=span_a,
        conflicting_span_b=span_b,
    )


def classify_candidates(
    candidates: list[tuple[Claim, Claim, float]]
) -> list[EntailmentResult]:
    """Runs entailment classification across every candidate pair, with progress."""
    results = []
    for claim_a, claim_b, similarity in candidates:
        result = classify_pair(claim_a, claim_b, similarity)
        results.append(result)
        print(f"  [{result.label:13}] {claim_a.source} vs {claim_b.source} "
              f"(confidence={result.confidence:.2f})")
    return results


def get_contradictions(results: list[EntailmentResult]) -> list[EntailmentResult]:
    """Filters to just the pairs classified as contradiction — what scoring.py acts on."""
    return [r for r in results if r.label == "contradiction"]


def verify_against_answer_key(
    results: list[EntailmentResult],
    answer_key_path: str = "data/answer_key.json",
) -> None:
    """
    Two-sided check, both matter for the demo's credibility line:
    1. Recall — did every seeded conflict doc-pair get AT LEAST ONE claim
       pair classified as "contradiction"?
    2. False positives — did any doc-pair classified as "contradiction"
       correspond to a seeded no_conflict pair, or to a pair not in the
       answer key at all (an unplanned false alarm)?
    """
    with open(answer_key_path, "r", encoding="utf-8") as f:
        answer_key = json.load(f)

    seeded_conflicts = {
        tuple(sorted([e["doc_a"].replace(".txt", ""), e["doc_b"].replace(".txt", "")]))
        for e in answer_key if e["label"] == "contradiction"
    }
    seeded_no_conflict = {
        tuple(sorted([e["doc_a"].replace(".txt", ""), e["doc_b"].replace(".txt", "")]))
        for e in answer_key if e["label"] == "no_conflict"
    }

    # doc-pairs the model actually flagged as contradiction
    flagged_doc_pairs = {
        tuple(sorted([r.claim_a.doc_id, r.claim_b.doc_id]))
        for r in results if r.label == "contradiction"
    }

    print("\n--- Recall: seeded conflicts ---")
    found = 0
    for pair in seeded_conflicts:
        hit = pair in flagged_doc_pairs
        found += hit
        print(f"  [{'FOUND' if hit else 'MISSED':6}] {pair[0]} <-> {pair[1]}")
    print(f"Recall: {found}/{len(seeded_conflicts)} seeded conflicts correctly flagged")

    print("\n--- False positives ---")
    false_positives = flagged_doc_pairs - seeded_conflicts
    if not false_positives:
        print("  None — every flagged pair matches a seeded conflict.")
    else:
        for pair in false_positives:
            tag = "SHOULD BE NO-CONFLICT" if pair in seeded_no_conflict else "unplanned"
            print(f"  [{tag}] {pair[0]} <-> {pair[1]}")
    print(f"False positive count: {len(false_positives)}")


if __name__ == "__main__":
    from core.ingestion import load_corpus
    from core.claim_extraction import extract_claims_from_corpus
    from core.pairing import get_candidate_pairs

    chunks = load_corpus("data/corpus")
    print(f"Extracting claims from {len(chunks)} chunks...")
    claims = extract_claims_from_corpus(chunks)

    print(f"\nFinding candidate pairs...")
    candidates = get_candidate_pairs(claims)
    print(f"{len(candidates)} candidate pairs to classify\n")

    print("Classifying pairs (one Groq call each)...")
    results = classify_candidates(candidates)

    contradictions = get_contradictions(results)
    print(f"\nTotal pairs classified: {len(results)}")
    print(f"Classified as contradiction: {len(contradictions)}")

    verify_against_answer_key(results)